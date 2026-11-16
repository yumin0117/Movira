/* Inference runs away from the UI thread. Media never leaves this browser. */
importScripts('./assets/ort.webgpu.min.js','./pose-core.js');
ort.env.wasm.wasmPaths=new URL('./assets/',self.location.href).href;
ort.env.wasm.numThreads=1;
ort.env.wasm.proxy=false;
let session,footSession;
const surface=new OffscreenCanvas(640,640);
const context=surface.getContext('2d',{willReadFrequently:true});
const footSurface=new OffscreenCanvas(256,256);
const footContext=footSurface.getContext('2d',{willReadFrequently:true});
async function loadModel() {
  const response=await fetch(new URL('./assets/yolov8n-pose.onnx',self.location.href));
  if(!response.ok) throw Error('모델을 불러올 수 없습니다. 인터넷 연결을 확인하고 다시 선택해 주세요.');
  const reader=response.body?.getReader();
  let model;
  if(reader) {
    const chunks=[];let received=0;const size=Number(response.headers.get('content-length'))||13484153;
    while(true) { const {done,value}=await reader.read(); if(done)break; chunks.push(value);received+=value.length;self.postMessage({type:'progress',percent:Math.min(99,Math.round(received/size*100))}); }
    model=new Uint8Array(received);let offset=0;
    for(const chunk of chunks){model.set(chunk,offset);offset+=chunk.length;}
  } else model=new Uint8Array(await response.arrayBuffer());
  self.postMessage({type:'progress',percent:100});
  if(self.navigator.gpu) {
    try { session=await ort.InferenceSession.create(model,{executionProviders:['webgpu','wasm'],graphOptimizationLevel:'all'}); }
    catch(error) { /* GPU support varies by device. Use the same local model on CPU. */ }
  }
  if(!session)session=await ort.InferenceSession.create(model,{executionProviders:['wasm'],graphOptimizationLevel:'all'});
  self.postMessage({type:'progress',label:'발끝 분석 모델 준비 중',percent:100});
  const footResponse=await fetch(new URL('./assets/pose-feet.onnx',self.location.href));
  if(!footResponse.ok)throw Error('발끝 분석 모델을 불러올 수 없습니다. 다시 선택해 주세요.');
  const footModel=new Uint8Array(await footResponse.arrayBuffer());
  if(self.navigator.gpu) {
    try { footSession=await ort.InferenceSession.create(footModel,{executionProviders:['webgpu','wasm'],graphOptimizationLevel:'all'}); }
    catch(error) { /* Keep full-body processing available on CPU. */ }
  }
  if(!footSession)footSession=await ort.InferenceSession.create(footModel,{executionProviders:['wasm'],graphOptimizationLevel:'all'});
}
async function refineFeet(bitmap,pose) {
  pose.feet=[];
  if(!footSession||Math.max(pose.joints[15].confidence,pose.joints[16].confidence)<.2)return;
  const crop=MoviraPose.footCrop(pose),scale=256/crop.side;
  footContext.setTransform(1,0,0,1,0,0);footContext.fillStyle='#000';footContext.fillRect(0,0,256,256);
  footContext.setTransform(scale*crop.cos,scale*crop.sin,-scale*crop.sin,scale*crop.cos,128-scale*crop.cos*crop.cx+scale*crop.sin*crop.cy,128-scale*crop.sin*crop.cx-scale*crop.cos*crop.cy);
  footContext.drawImage(bitmap,0,0);footContext.setTransform(1,0,0,1,0,0);
  const rgba=footContext.getImageData(0,0,256,256).data,input=new Float32Array(256*256*3);
  for(let i=0;i<256*256;i++){input[i*3]=rgba[i*4]/255;input[i*3+1]=rgba[i*4+1]/255;input[i*3+2]=rgba[i*4+2]/255;}
  const tensor=new ort.Tensor('float32',input,[1,256,256,3]);
  const results=await footSession.run({[footSession.inputNames[0]]:tensor});
  const outputs=Object.values(results),landmarks=outputs.find(output=>output.data.length===195),presence=outputs.find(output=>output.data.length===1);
  if(landmarks&&presence)pose.feet=MoviraPose.decodeFeet(landmarks.data,presence.data[0],crop,pose).filter(foot=>[foot.heel,foot.toe].every(point=>point.x>=0&&point.x<bitmap.width&&point.y>=0&&point.y<bitmap.height));
  tensor.dispose();for(const result of outputs)result.dispose();
}
self.onmessage=async ({data})=>{
  const {id,type,bitmap}=data;
  try {
    if(type==='init') { if(!session||!footSession) await loadModel();self.postMessage({id,type:'ready'});return; }
    if(type!=='infer') return;
    if(!session) throw Error('모델 준비를 먼저 완료해 주세요.');
    const started=performance.now(), t=MoviraPose.letterbox(bitmap.width,bitmap.height);
    context.fillStyle='rgb(114,114,114)';context.fillRect(0,0,640,640);
    context.drawImage(bitmap,t.left,t.top,t.scaledWidth,t.scaledHeight);
    const rgba=context.getImageData(0,0,640,640).data;
    const plane=640*640,input=new Float32Array(plane*3);
    for(let i=0;i<plane;i++){input[i]=rgba[4*i]/255;input[plane+i]=rgba[4*i+1]/255;input[plane*2+i]=rgba[4*i+2]/255;}
    const tensor=new ort.Tensor('float32',input,[1,3,640,640]);
    const results=await session.run({[session.inputNames[0]]:tensor});
    const output=results[session.outputNames[0]];
    const poses=MoviraPose.decode(output.data,output.dims,t);
    tensor.dispose();for(const result of Object.values(results)) result.dispose();
    for(const pose of poses)await refineFeet(bitmap,pose);
    self.postMessage({id,type:'result',poses,ms:performance.now()-started,bitmap},[bitmap]);
  } catch(error) {
    if(bitmap) bitmap.close();
    self.postMessage({id,type:'error',message:error.message||'분석 중 오류가 발생했습니다.'});
  }
};
