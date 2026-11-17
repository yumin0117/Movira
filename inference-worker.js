/* Original worker. All imported JavaScript files below are Movira source. */
importScripts('./gpu-graph.js?v=hands2','./vision-core.js?v=hands2','./calibration.js?v=hands2','./motion.js?v=hands2','./hand-net.js?v=hands2','./hands.js?v=hands2','./vision-runtime.js?v=hands2','./calibration-trainer.js?v=hands2');
let runtime=null,queue=Promise.resolve(),cancelled=false;
const surface=new OffscreenCanvas(1,1),context=surface.getContext('2d',{willReadFrequently:true});
async function ensure(){if(!runtime){self.postMessage({type:'progress',label:'인식 가중치 불러오는 중 · 첫 실행 약 46 MB',progress:0});runtime=await MoviraRuntime.create((label,progress)=>self.postMessage({type:'progress',label,progress}));try{const response=await fetch('./calibration.json');if(response.ok)runtime.calibration=MoviraCalibration.Calibration.import(await response.json());}catch{/* Raw public weights remain usable if the optional adaptation is unavailable. */}}return runtime;}
function frame(bitmap){surface.width=bitmap.width;surface.height=bitmap.height;context.drawImage(bitmap,0,0);return {pixels:context.getImageData(0,0,bitmap.width,bitmap.height).data,width:bitmap.width,height:bitmap.height};}
async function handle(data){const {id,type}=data;let bitmap=data.bitmap;try{
  const engine=await ensure();
  if(type==='init'){self.postMessage({id,type:'ready',calibration:engine.calibration.export(),handAvailable:Boolean(engine.hands)});return;}
  if(type==='calibration'){engine.calibration=MoviraCalibration.Calibration.import(data.calibration);self.postMessage({id,type:'ready',calibration:engine.calibration.export()});return;}
  if(type==='infer'){const started=performance.now(),poses=await engine.infer(frame(bitmap),{threshold:data.threshold,detail:data.detail,video:data.video,time:data.time,stream:data.stream,hands:data.hands,refine:data.refine});self.postMessage({id,type:'result',poses,ms:performance.now()-started,bitmap},[bitmap]);bitmap=null;return;}
  if(type==='train'){
    cancelled=false;const records=[];
    for(let i=0;i<data.records.length;i++){if(cancelled)throw Error('학습을 중단했습니다.');const r=data.records[i],blob=await fetch(r.dataURL).then(response=>response.blob()),picture=await createImageBitmap(blob);let proposals;try{proposals=await engine.proposals(frame(picture),true,.02);}finally{picture.close();}records.push({name:r.name,width:r.width,height:r.height,people:r.people,ignoreBoxes:r.ignoreBoxes||[],proposals,split:i<Math.floor(data.records.length*.8)?'train':'validation'});self.postMessage({type:'training',label:`학습 사진 분석 · ${i+1}/${data.records.length}`,progress:(i+1)/data.records.length*.7});}
    const trained=await MoviraTrainer.train(records,{epochs:data.epochs,cancel:()=>cancelled,progress:p=>self.postMessage({type:'training',label:`보정 학습 · ${p.epoch}/${p.epochs} · 손실 ${p.loss.toFixed(4)}`,progress:.7+.3*p.epoch/p.epochs})});if(cancelled)throw Error('학습을 중단했습니다.');engine.calibration=MoviraCalibration.Calibration.import(trained.calibration);self.postMessage({id,type:'trained',...trained});return;
  }
}catch(error){if(bitmap)bitmap.close();self.postMessage({id,type:'error',message:error.message||'분석에 실패했습니다.'});}}
self.onmessage=({data})=>{if(data.type==='stop'){cancelled=true;return;}queue=queue.then(()=>handle(data));};
