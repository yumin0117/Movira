'use strict';
const $=id=>document.getElementById(id),canvas=$('canvas'),ctx=canvas.getContext('2d'),video=$('video');
function checkbox(id,text){const label=document.createElement('label'),input=document.createElement('input');label.className='check';input.type='checkbox';input.id=id;label.append(input,document.createTextNode(text));return label;}
$('showJoints').closest('label').after(checkbox('showInferred','가려진 관절 추정 표시 · 점선'));$('showInferred').checked=true;
$('skipJoint').after(checkbox('occludedPoint','선택 관절이 가림 · 정답 위치를 알 때'));
$('confidence').closest('label').firstChild.textContent='인물 검출 기준 ';
$('confidence').max='99.9';$('confidence').step='.1';
const lineLabel=document.createElement('label');lineLabel.textContent='선 두께';const lineInput=document.createElement('input');lineInput.id='lineWidth';lineInput.type='range';lineInput.min='1';lineInput.max='7';lineInput.value='3';lineLabel.append(lineInput);$('confidence').closest('label').after(lineLabel);
const occlusionNote=document.createElement('p');occlusionNote.textContent='점선과 속이 빈 점은 가려진 관절의 추정입니다. 실제로 보이는 위치와 구분하며, 완전히 가려진 사람은 찾지 못할 수 있습니다.';$('validation').before(occlusionNote);
const scratch=document.createElement('canvas'),pixelsContext=scratch.getContext('2d',{willReadFrequently:true});
const analyzedFrame=document.createElement('canvas');let hasAnalyzedFrame=false;
let media=null,isVideo=false,mediaURL=null,model=null,records=[],active=null,selected=0,jointIndex=0,placingBox=false,gesture=null,poses=[],generation=0,running=false,stop=false,lastDetection=0;
function status(message){$('status').textContent=message;}
function time(value){return `${Math.floor(value/60)||0}:${String(Math.floor(value%60)||0).padStart(2,'0')}`;}
function download(name,data,type='application/json'){const blob=data instanceof Blob?data:new Blob([JSON.stringify(data)],{type});const url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),2000);}
function currentPerson(){return active?.people[selected];}
function refresh(){
  $('jointSelect').replaceChildren(...Movira.joints.map((name,i)=>new Option(`${i+1}. ${name}`,String(i))));$('jointSelect').value=String(jointIndex);
  $('personSelect').replaceChildren(...(active?.people.length?active.people.map((_,i)=>new Option(`사람 ${i+1}`,String(i))):[new Option('아직 없음','0')]));$('personSelect').value=String(selected);
  $('addPerson').disabled=!active||$('mode').value!=='label';$('removePerson').disabled=!currentPerson()||$('mode').value!=='label';$('skipJoint').disabled=!currentPerson()||$('mode').value!=='label';
  $('datasetCount').textContent=`학습 사진 ${records.length}장 · 인물 ${records.reduce((n,r)=>n+r.people.length,0)}명`;
  const person=currentPerson();$('labelHelp').textContent=person?`${person.known.filter(Boolean).length}/${Movira.joints.length}개 확인 · ${Movira.joints[jointIndex]} 위치를 클릭하거나 점을 드래그하세요.`:'사람 영역을 드래그한 뒤 각 관절을 클릭하세요.';
  $('occludedPoint').disabled=!person?.points[jointIndex]||$('mode').value!=='label';$('occludedPoint').checked=Boolean(person?.points[jointIndex]?.occluded);
  $('gallery').replaceChildren(...records.map(record=>{const b=document.createElement('button'),img=document.createElement('img');img.src=record.dataURL;img.alt='';b.append(img,document.createTextNode(record.name));b.classList.toggle('active',active===record);b.addEventListener('click',()=>showRecord(record));return b;}));
}
function draw(){
  if(!media)return;const w=isVideo?video.videoWidth:media.naturalWidth,h=isVideo?video.videoHeight:media.naturalHeight;if(!w||!h)return;
  const ratio=Math.min(1,1280/w,900/h);canvas.width=Math.round(w*ratio);canvas.height=Math.round(h*ratio);ctx.fillStyle='#0c1418';ctx.fillRect(0,0,canvas.width,canvas.height);
  if($('showOriginal').checked)ctx.drawImage(isVideo&&hasAnalyzedFrame?analyzedFrame:media,0,0,canvas.width,canvas.height);
  const labels=$('mode').value==='label',people=labels?(active?.people||[]).map(p=>({...p,points:p.points.map((point,i)=>point?{...point,confidence:p.known[i]?1:0}:null)})):poses;
  const threshold=labels ? .1 : Number($('confidence').value)/100;ctx.lineWidth=Math.max(1,Number($('lineWidth').value)*canvas.width/850);ctx.lineCap='round';ctx.lineJoin='round';
  const visible=p=>p&&p.confidence>=threshold&&(labels||!p.occluded||$('showInferred').checked);
  for(const [index,person] of people.entries()){
    ctx.strokeStyle=labels&&index===selected?'#ffcc77':'#c7f57c';
    if(labels){const b=person.box;ctx.setLineDash([5,5]);ctx.strokeRect(b.x*ratio,b.y*ratio,b.w*ratio,b.h*ratio);ctx.setLineDash([]);}
    if($('showBones').checked)for(const [a,b] of Movira.bones)if(visible(person.points[a])&&visible(person.points[b])){ctx.setLineDash(person.points[a].occluded||person.points[b].occluded?[5,5]:[]);ctx.beginPath();ctx.moveTo(person.points[a].x*ratio,person.points[a].y*ratio);ctx.lineTo(person.points[b].x*ratio,person.points[b].y*ratio);ctx.stroke();}ctx.setLineDash([]);
    if($('showJoints').checked)for(const [k,p] of person.points.entries())if(visible(p)){ctx.beginPath();ctx.arc(p.x*ratio,p.y*ratio,labels&&index===selected&&k===jointIndex?6:3,0,Math.PI*2);ctx.fillStyle=labels&&index===selected&&k===jointIndex?'#ffcc77':'#83e8df';if(p.occluded)ctx.stroke();else ctx.fill();}
  }
  if(gesture?.type==='box'){ctx.strokeStyle='#ffcc77';ctx.strokeRect(gesture.start.x*ratio,gesture.start.y*ratio,(gesture.end.x-gesture.start.x)*ratio,(gesture.end.y-gesture.start.y)*ratio);}
  $('peopleCount').textContent=`${people.length}명`;$('empty').hidden=true;canvas.hidden=false;$('savePng').disabled=false;
}
function framePixels(){const w=isVideo?video.videoWidth:media.naturalWidth,h=isVideo?video.videoHeight:media.naturalHeight;scratch.width=w;scratch.height=h;pixelsContext.drawImage(media,0,0);return {data:pixelsContext.getImageData(0,0,w,h).data,width:w,height:h};}
function infer(){
  if(!media||$('mode').value==='label'){draw();return;}
  if(!model){poses=[];draw();status('자체 학습 모델이 없습니다. 학습 데이터를 열고 학습하거나 자체 모델 JSON을 선택하세요.');return;}
  const frame=framePixels(),start=performance.now();if(isVideo){analyzedFrame.width=frame.width;analyzedFrame.height=frame.height;analyzedFrame.getContext('2d').drawImage(scratch,0,0);hasAnalyzedFrame=true;}poses=Movira.detect(model,frame.data,frame.width,frame.height,Number($('confidence').value)/100);draw();status(`실험용 자동 인식 · ${poses.length}명 · ${(performance.now()-start).toFixed(0)} ms · 학습 범위 밖 사진에서는 크게 틀릴 수 있습니다.`);
}
function tick(now){if(isVideo&&media===video&&!video.paused){if(now-lastDetection>180){lastDetection=now;infer();}$('seek').value=String(video.currentTime/video.duration*1000);$('time').textContent=time(video.currentTime);}requestAnimationFrame(tick);}
requestAnimationFrame(tick);
async function imageFrom(dataURL){const img=new Image();img.src=dataURL;await img.decode();if(img.naturalWidth*img.naturalHeight>20000000)throw Error('사진은 2천만 화소 이하로 준비해 주세요.');return img;}
async function showRecord(record){const token=++generation;video.pause();const image=await imageFrom(record.dataURL);if(token!==generation)return;isVideo=false;active=record;selected=0;media=image;$('fileName').textContent=record.name;$('viewerTitle').textContent=record.name;$('playback').hidden=true;$('mode').value='label';$('capture').disabled=false;poses=[];refresh();draw();status('정답 편집 모드 · 사람이 옆으로 눕거나 팔을 들어도 실제 관절 위치에 표시하세요.');}
async function openMedia(file){
  const token=++generation;video.pause();active=null;poses=[];placingBox=false;hasAnalyzedFrame=false;if(mediaURL)URL.revokeObjectURL(mediaURL);mediaURL=URL.createObjectURL(file);$('mode').value='detect';
  try{isVideo=file.type.startsWith('video/');if(isVideo){await new Promise((resolve,reject)=>{video.onloadeddata=resolve;video.onerror=()=>reject(Error('브라우저가 이 영상 형식을 재생하지 못했습니다. MP4(H.264) 또는 WebM을 사용하세요.'));video.src=mediaURL;video.loop=$('loop').checked;video.load();});if(token!==generation)return;media=video;$('duration').textContent=time(video.duration);$('playback').hidden=false;}else{const img=await imageFrom(mediaURL);if(token!==generation)return;media=img;$('playback').hidden=true;}
    $('viewerTitle').textContent=file.name;$('fileName').textContent=file.name;$('capture').disabled=false;refresh();infer();
  }catch(error){if(token===generation)status(error.message);}
}
$('mediaInput').addEventListener('change',e=>{if(e.target.files[0])openMedia(e.target.files[0]);e.target.value='';});
document.addEventListener('dragover',e=>e.preventDefault());document.addEventListener('drop',e=>{e.preventDefault();if(e.dataTransfer.files[0])openMedia(e.dataTransfer.files[0]);});
$('capture').addEventListener('click',async()=>{if(!media)return;video.pause();const frame=framePixels(),record={name:`${$('fileName').textContent} @ ${isVideo?time(video.currentTime):'사진'}`,width:frame.width,height:frame.height,dataURL:scratch.toDataURL('image/png'),people:[]};records.push(record);await showRecord(record);placingBox=true;status('사람 전체를 감싸는 사각형을 드래그하세요.');});
$('mode').addEventListener('change',()=>{video.pause();if($('mode').value==='label'&&!active){status('현재 프레임을 학습 사진으로 추가한 뒤 정답을 표시하세요.');}refresh();infer();});
$('addPerson').addEventListener('click',()=>{placingBox=true;status('사람 전체를 감싸는 영역을 드래그하세요.');});
$('removePerson').addEventListener('click',()=>{if(currentPerson()){active.people.splice(selected,1);selected=Math.max(0,selected-1);refresh();draw();}});
$('personSelect').addEventListener('change',e=>{selected=Number(e.target.value);refresh();draw();});$('jointSelect').addEventListener('change',e=>{jointIndex=Number(e.target.value);refresh();draw();});
$('occludedPoint').addEventListener('change',()=>{const point=currentPerson()?.points[jointIndex];if(point){point.occluded=$('occludedPoint').checked;draw();}});
function nextJoint(){const person=currentPerson();if(person){const next=person.known.findIndex((known,i)=>i>jointIndex&&!known);jointIndex=next<0?Math.min(jointIndex+1,Movira.joints.length-1):next;}refresh();draw();}
$('skipJoint').addEventListener('click',()=>{const person=currentPerson();if(person){person.points[jointIndex]=null;person.known[jointIndex]=true;nextJoint();}});
function position(e){const bounds=canvas.getBoundingClientRect(),w=active?.width||canvas.width,h=active?.height||canvas.height;return {x:Math.max(0,Math.min(w-1,(e.clientX-bounds.left)/bounds.width*w)),y:Math.max(0,Math.min(h-1,(e.clientY-bounds.top)/bounds.height*h))};}
canvas.addEventListener('pointerdown',e=>{if($('mode').value!=='label'||!active||running)return;const p=position(e);canvas.setPointerCapture(e.pointerId);if(placingBox){gesture={type:'box',start:p,end:p};return;}const person=currentPerson();if(!person)return;const tolerance=12*active.width/canvas.getBoundingClientRect().width;let nearest=-1,dist=tolerance;person.points.forEach((point,i)=>{if(point){const d=Math.hypot(point.x-p.x,point.y-p.y);if(d<dist){nearest=i;dist=d;}}});if(nearest>=0){jointIndex=nearest;gesture={type:'joint',index:nearest};}else{person.points[jointIndex]=p;person.known[jointIndex]=true;gesture={type:'new',index:jointIndex};}draw();});
canvas.addEventListener('pointermove',e=>{if(!gesture)return;const p=position(e);if(gesture.type==='box')gesture.end=p;else if(currentPerson())currentPerson().points[gesture.index]=p;draw();});
canvas.addEventListener('pointerup',()=>{if(!gesture)return;if(gesture.type==='box'){const a=gesture.start,b=gesture.end;if(Math.abs(a.x-b.x)>=12&&Math.abs(a.y-b.y)>=12){active.people.push({box:{x:Math.min(a.x,b.x),y:Math.min(a.y,b.y),w:Math.abs(a.x-b.x),h:Math.abs(a.y-b.y)},points:Array(Movira.joints.length).fill(null),known:Array(Movira.joints.length).fill(false)});selected=active.people.length-1;jointIndex=0;placingBox=false;}}else if(gesture.type==='new')nextJoint();gesture=null;refresh();draw();});canvas.addEventListener('pointercancel',()=>{gesture=null;draw();});
for(const id of ['showOriginal','showBones','showJoints','showInferred'])$(id).addEventListener('change',draw);$('confidence').addEventListener('input',()=>{$('confidenceText').textContent=`${$('confidence').value}%`;infer();});
$('lineWidth').addEventListener('input',draw);
$('play').addEventListener('click',async()=>{if(!isVideo)return;if(video.paused){$('mode').value='detect';refresh();await video.play();$('play').textContent='Ⅱ 일시정지';}else{video.pause();$('play').textContent='▶ 재생';}});
$('seek').addEventListener('input',()=>{if(isVideo&&Number.isFinite(video.duration))video.currentTime=Number($('seek').value)/1000*video.duration;});video.addEventListener('seeked',infer);video.addEventListener('ended',()=>{$('play').textContent='▶ 재생';});$('loop').addEventListener('change',()=>video.loop=$('loop').checked);
$('savePng').addEventListener('click',()=>canvas.toBlob(blob=>{if(blob)download('movira-result.png',blob);},'image/png'));
$('saveData').addEventListener('click',()=>download('movira-dataset.json',{format:'movira-dataset-v1',joints:Movira.joints,records:records.map(({name,width,height,dataURL,people,source,ignoreBoxes})=>({name,width,height,dataURL,people,source,ignoreBoxes}))}));
async function readJSON(file){if(file.size>150*1024*1024)throw Error('JSON 파일은 150 MB 이하로 준비해 주세요.');return JSON.parse(await file.text());}
async function importData(data){
  if(data?.format!=='movira-dataset-v1'||!Array.isArray(data.records)||data.records.length>500)throw Error('Movira 학습 데이터 형식이 아닙니다.');
  const incoming=[];
  for(const r of data.records){
    if(typeof r.dataURL!=='string'||!/^data:image\/(png|jpeg|webp);base64,/.test(r.dataURL)||!Array.isArray(r.people))throw Error('이미지와 정답 형식을 확인해 주세요.');const img=await imageFrom(r.dataURL);
    if(img.naturalWidth!==r.width||img.naturalHeight!==r.height)throw Error('사진과 정답 좌표의 크기가 다릅니다.');
    const validPoint=p=>p===null||p&&Number.isFinite(p.x)&&Number.isFinite(p.y)&&p.x>=0&&p.x<r.width&&p.y>=0&&p.y<r.height;
    for(const p of r.people){if(!p.box||!['x','y','w','h'].every(k=>Number.isFinite(p.box[k]))||p.box.w<=0||p.box.h<=0||!Array.isArray(p.points)||p.points.length!==Movira.joints.length||!p.points.every(validPoint)||!Array.isArray(p.known)||p.known.length!==Movira.joints.length||!p.known.every(k=>typeof k==='boolean'))throw Error('관절 정답 좌표가 올바르지 않습니다.');}
    const ignoreBoxes=Array.isArray(r.ignoreBoxes)?r.ignoreBoxes.filter(b=>b&&['x','y','w','h'].every(k=>Number.isFinite(b[k]))&&b.w>0&&b.h>0):undefined;
    incoming.push({name:String(r.name||'학습 사진'),width:r.width,height:r.height,dataURL:r.dataURL,people:r.people,source:r.source||null,ignoreBoxes});
  }
  records=incoming;refresh();if(records.length)await showRecord(records[0]);status(`학습 사진 ${records.length}장을 열었습니다. 수정 후 데이터 저장을 눌러 보관하세요.`);
}
$('dataInput').addEventListener('change',async e=>{try{if(e.target.files[0])await importData(await readJSON(e.target.files[0]));}catch(error){status(error.message);}e.target.value='';});
function modelInfo(){
  if(Number.isFinite(model?.meta?.recommendedThreshold)){$('confidence').value=String(model.meta.recommendedThreshold*100);$('confidenceText').textContent=`${$('confidence').value}%`;}
  if(model&&Number($('progress').value)===0)$('trainStatus').textContent='자체 학습 모델을 불러왔습니다. 이 화면에서 새 학습은 아직 실행하지 않았습니다.';
  $('saveModel').disabled=!model;$('modelStatus').textContent=model?`자체 ${model instanceof Movira.ConvModel?'CNN':'MLP'} · ${model.steps.toLocaleString()}회 갱신 · 학습 관절 ${model.coverage.filter(n=>n>=3).length}/${Movira.joints.length}종`:'학습 모델 없음';
  const v=model?.meta?.validation;$('validation').textContent=v?`별도 사진 검증: ${v.samples}개 영역 · 사람/배경 분류 ${(100*v.classificationAccuracy).toFixed(1)}% · 관절 평균 오차 ${v.normalizedJointError===null?'측정 없음':(100*v.normalizedJointError).toFixed(2)+'% (영역 크기 정규화)'} · 정답 사람 영역을 제공한 평가이며 전체 자동 검출 정확도는 아닙니다.`:'검증 결과 없음';
  const full=model?.meta?.fullImageTest;if(full)$('validation').textContent+=` 전체 사진 시험: 검출 정밀도 ${(full.precision*100).toFixed(1)}%, 재현율 ${(full.recall*100).toFixed(1)}%. 낮은 수치는 추가 개발이 필요하다는 뜻입니다.`;
}
$('modelInput').addEventListener('change',async e=>{try{if(e.target.files[0]){model=Movira.Model.import(await readJSON(e.target.files[0]));modelInfo();infer();}}catch(error){status(error.message);}e.target.value='';});$('saveModel').addEventListener('click',()=>{if(model)download('movira-own-model.json',model.export());});
async function prepare(records){
  const samples=[];
  for(const record of records){const img=await imageFrom(record.dataURL);scratch.width=record.width;scratch.height=record.height;pixelsContext.drawImage(img,0,0);const pixels=pixelsContext.getImageData(0,0,record.width,record.height).data;
    for(const person of record.people){if(!person.points.some((p,i)=>p&&person.known[i]))continue;const box=Movira.crop(person.box);for(const mirrored of [false,true])samples.push({input:Movira.features(pixels,record.width,record.height,box,mirrored),positive:true,points:Movira.pointsTarget(person,box,mirrored),box:Movira.boxTarget(person,box,mirrored)});}
    const ignore=record.ignoreBoxes||record.people.map(p=>p.box);let negatives=0;for(const box of Movira.windows(record.width,record.height))if(negatives<Math.max(2,record.people.length*2)&&ignore.every(b=>Movira.overlap(box,b)<.02)){samples.push({input:Movira.features(pixels,record.width,record.height,box),positive:false,points:[]});negatives++;}
  }
  return samples;
}
$('stopTrain').addEventListener('click',()=>stop=true);
$('train').addEventListener('click',async()=>{
  if(running)return;if(records.length<2){status('서로 다른 학습 사진을 최소 2장 준비하세요. 실제 일반화 성능에는 훨씬 많은 사진이 필요합니다.');return;}
  running=true;stop=false;video.pause();$('train').disabled=true;$('stopTrain').disabled=false;
  try{status('학습 영역과 정답 좌표를 준비하고 있습니다.');const split=Math.max(1,Math.floor(records.length*.8)),trainRecords=records.slice(0,split),valRecords=records.slice(split),samples=await prepare(trainRecords),validation=await prepare(valRecords);
    if(!samples.some(s=>s.positive)||!validation.some(s=>s.positive))throw Error('학습·검증 사진 모두에 관절 정답이 필요합니다.');
    const candidate=new Movira.ConvModel(),rng=Movira.random(3261),epochs=Math.max(1,Math.min(500,Number($('epochs').value)||60));
    for(const record of trainRecords)for(const person of record.people)person.points.forEach((p,k)=>{if(p&&person.known[k])candidate.coverage[k]++;});
    for(let epoch=0;epoch<epochs&&!stop;epoch++){
      for(let i=samples.length-1;i>0;i--){const j=Math.floor(rng()*(i+1));[samples[i],samples[j]]=[samples[j],samples[i]];}
      let loss=0;for(let i=0;i<samples.length;i++){const sample=samples[i];loss+=candidate.train(sample.positive&&rng()<.4?Movira.occlude(sample,rng):sample,.0015/(1+epoch*.02));if(i%16===0)await new Promise(resolve=>setTimeout(resolve,0));}
      $('progress').value=(epoch+1)/epochs;$('trainStatus').textContent=`${epoch+1}/${epochs}회 · 손실 ${(loss/samples.length).toFixed(4)}`;await new Promise(resolve=>setTimeout(resolve,0));
    }
    candidate.meta={created:new Date().toISOString(),implementation:'Movira original CNN',trainImages:trainRecords.length,validationImages:valRecords.length,validation:Movira.evaluate(candidate,validation),experimental:true};model=candidate;modelInfo();status(stop?'학습을 중단했습니다. 현재까지 학습한 모델을 저장할 수 있습니다.':'자체 모델 학습을 마쳤습니다. 모델 저장 후, 학습에 쓰지 않은 사진으로 인식을 확인하세요.');
  }catch(error){status(error.message);}finally{running=false;$('train').disabled=false;$('stopTrain').disabled=true;}
});
// This locally produced model contains no upstream parameters. No media is sent.
if(location.protocol!=='file:')fetch('./own-model.json',{cache:'no-cache'}).then(r=>r.ok?r.json():null).then(data=>{if(data&&!model){model=Movira.Model.import(data);modelInfo();if(media)infer();}}).catch(()=>{status('기본 자체 모델을 열지 못했습니다. 모델 JSON을 직접 선택하거나 새로 학습해 주세요.');});
refresh();modelInfo();
