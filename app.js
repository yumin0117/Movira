'use strict';
const $=id=>document.getElementById(id),canvas=$('canvas'),ctx=canvas.getContext('2d'),video=$('video');
function checkbox(id,text){const label=document.createElement('label'),input=document.createElement('input');label.className='check';input.type='checkbox';input.id=id;label.append(input,document.createTextNode(text));return label;}
$('showJoints').closest('label').after(checkbox('showInferred','가려진 관절 추정 표시 · 점선'));$('showInferred').checked=true;$('showInferred').closest('label').hidden=true;
$('skipJoint').after(checkbox('occludedPoint','선택 관절이 가림 · 정답 위치를 알 때'));
$('occludedPoint').closest('label').hidden=true;
$('confidence').closest('label').firstChild.textContent='인식 민감도 ';
$('confidence').min='0';$('confidence').max='100';$('confidence').step='1';
$('confidence').closest('label').parentElement.querySelector('p.muted').textContent='높일수록 작은 사람·흐린 사람을 더 적극적으로 찾습니다. 100에서도 오검출이 생길 수 있습니다.';
let sensitivityChanged=false;
const lineLabel=document.createElement('label');lineLabel.textContent='선 두께';const lineInput=document.createElement('input');lineInput.id='lineWidth';lineInput.type='range';lineInput.min='1';lineInput.max='7';lineInput.value='3';lineLabel.append(lineInput);$('confidence').closest('label').after(lineLabel);
const detailLabel=checkbox('detail','작은 인물도 자세히 찾기');$('showOriginal').closest('label').before(detailLabel);$('detail').checked=true;
const handLabel=checkbox('showHands','몸과 손가락 함께 인식');$('showJoints').closest('label').after(handLabel);$('showHands').checked=true;const handStatus=document.createElement('p');handStatus.className='muted';handStatus.textContent='손은 사람별 손목에 연결해 함께 분석합니다. 작거나 흐린 손은 생략할 수 있습니다.';handLabel.after(handStatus);
const refineLabel=checkbox('refineBody','관절 위치 정밀 보정');$('detail').closest('label').after(refineLabel);$('refineBody').checked=true;
const videoSettings=document.createElement('section');videoSettings.hidden=true;videoSettings.innerHTML='<h2>03 · 영상 재생</h2><label>재생 방식<select id="videoMode"><option value="live">실시간 · 속도 우선</option><option value="precise">정밀 · 분석에 맞춰 재생</option></select></label><label>재생 속도<select id="playbackSpeed"><option value="0.5">0.5배</option><option value="1" selected>1배</option><option value="2">2배</option></select></label><p class="muted">빠른 동작이 끊기면 정밀 재생을 선택하세요. 분석을 기다리며 천천히 재생합니다. 속도 설정은 실시간 재생에 적용됩니다.</p>';$('confidence').closest('section').after(videoSettings);
const busyLayer=document.createElement('div');busyLayer.className='busy-layer';busyLayer.hidden=true;busyLayer.setAttribute('role','status');$('stage').append(busyLayer);
const scratch=document.createElement('canvas'),pixelsContext=scratch.getContext('2d',{willReadFrequently:true});
const analyzedFrame=document.createElement('canvas');let hasAnalyzedFrame=false;
const videoCapture=document.createElement('canvas'),videoCaptureContext=videoCapture.getContext('2d');
let media=null,isVideo=false,mediaURL=null,model=null,records=[],active=null,selected=0,jointIndex=0,placingBox=false,gesture=null,poses=[],generation=0,running=false,stop=false,lastDetection=0;
let worker=null,ready=null,nextRequest=0,inferring=false,needsInference=false,inFlight=Promise.resolve(),precisePlaying=false,preciseToken=0,precisionSeeking=false,lastDecoded=-1;const requests=new Map();
function rpc(type,extra={},transfer=[]){return new Promise((resolve,reject)=>{const id=++nextRequest;requests.set(id,{resolve,reject});worker.postMessage({id,type,...extra},transfer);});}
async function ensureGPU(){
  if(ready)return ready;
  if(location.protocol==='file:')throw Error('웹사이트 링크 또는 로컬 서버(node scripts/serve.cjs)로 열어 주세요.');
  worker=new Worker('./inference-worker.js?v=hands2');
  worker.onmessage=({data})=>{if(data.type==='progress'){busyLayer.hidden=false;busyLayer.textContent=data.label;return;}if(data.type==='training'){$('trainStatus').textContent=data.label;$('progress').value=data.progress;return;}const request=requests.get(data.id);if(!request){data.bitmap?.close();return;}requests.delete(data.id);if(data.type==='error')request.reject(Error(data.message));else request.resolve(data);};
  worker.onerror=()=>{for(const request of requests.values())request.reject(Error('분석 엔진을 실행하지 못했습니다. 최신 Chrome 또는 Edge에서 다시 시도해 주세요.'));requests.clear();worker?.terminate();worker=null;ready=null;busyLayer.hidden=true;};
  ready=rpc('init').then(async result=>{handStatus.textContent=result.handAvailable?'몸과 양손·손가락을 함께 분석합니다. 손가락은 초기 모델이며 작은 손·가림은 누락되거나 위치에 오차가 생길 수 있습니다.':'손가락 모델은 아직 검증 조건을 통과하지 못해 표시하지 않습니다.';if(model)await rpc('calibration',{calibration:model.export()});else model=MoviraCalibration.Calibration.import(result.calibration);modelInfo();busyLayer.hidden=true;return result;}).catch(error=>{ready=null;worker?.terminate();worker=null;busyLayer.hidden=true;throw error;});return ready;
}
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
  const threshold=labels ? .1 : .18;ctx.lineWidth=Math.max(1,Number($('lineWidth').value)*canvas.width/850);ctx.lineCap='round';ctx.lineJoin='round';
  const visible=p=>p&&p.confidence>=threshold&&(labels||!p.occluded||$('showInferred').checked);
  for(const [index,person] of people.entries()){
    ctx.strokeStyle=labels&&index===selected?'#ffcc77':'#c7f57c';
    if(labels){const b=person.box;ctx.setLineDash([5,5]);ctx.strokeRect(b.x*ratio,b.y*ratio,b.w*ratio,b.h*ratio);ctx.setLineDash([]);}
    if($('showBones').checked)for(const [a,b] of Movira.bones)if(visible(person.points[a])&&visible(person.points[b])){ctx.setLineDash(person.points[a].occluded||person.points[b].occluded?[5,5]:[]);ctx.beginPath();ctx.moveTo(person.points[a].x*ratio,person.points[a].y*ratio);ctx.lineTo(person.points[b].x*ratio,person.points[b].y*ratio);ctx.stroke();}ctx.setLineDash([]);
    if($('showJoints').checked)for(const [k,p] of person.points.entries())if(visible(p)){ctx.beginPath();ctx.arc(p.x*ratio,p.y*ratio,labels&&index===selected&&k===jointIndex?6:3,0,Math.PI*2);ctx.fillStyle=labels&&index===selected&&k===jointIndex?'#ffcc77':'#83e8df';if(p.occluded)ctx.stroke();else ctx.fill();}
  }
  for(const person of people)if($('showHands').checked)for(const hand of person.hands||[]){const visible=p=>p&&p.confidence>=hand.pointThreshold;ctx.strokeStyle='#83e8df';ctx.fillStyle='#e9ffff';if($('showBones').checked)for(const [a,b] of MoviraHands.bones){const p=hand.points[a],q=hand.points[b];if(!visible(p)||!visible(q))continue;ctx.beginPath();ctx.moveTo(p.x*ratio,p.y*ratio);ctx.lineTo(q.x*ratio,q.y*ratio);ctx.stroke();}if($('showJoints').checked)for(const p of hand.points)if(visible(p)){ctx.beginPath();ctx.arc(p.x*ratio,p.y*ratio,Math.max(1.2,2*canvas.width/850),0,Math.PI*2);ctx.fill();}}
  if(gesture?.type==='box'){ctx.strokeStyle='#ffcc77';ctx.strokeRect(gesture.start.x*ratio,gesture.start.y*ratio,(gesture.end.x-gesture.start.x)*ratio,(gesture.end.y-gesture.start.y)*ratio);}
  const handCount=$('showHands').checked?people.reduce((n,p)=>n+(p.hands?.length||0),0):0;$('peopleCount').textContent=`${people.length}명 · 손 ${handCount}개`;$('empty').hidden=true;canvas.hidden=false;$('savePng').disabled=false;
}
function framePixels(){const w=isVideo?video.videoWidth:media.naturalWidth,h=isVideo?video.videoHeight:media.naturalHeight;scratch.width=w;scratch.height=h;pixelsContext.drawImage(media,0,0);return {data:pixelsContext.getImageData(0,0,w,h).data,width:w,height:h};}
function infer(){if(inferring){if(!isVideo||video.paused&&!precisePlaying)needsInference=true;return inFlight;}inFlight=inferFrame();return inFlight;}
async function inferFrame(){
  if(!media||$('mode').value==='label'){draw();return;}if(inferring){needsInference=true;return;}if(running)return;
  inferring=true;needsInference=false;const token=generation;
  try{if(!worker){busyLayer.hidden=false;busyLayer.textContent='분석 준비 중 · 첫 실행에는 가중치를 불러옵니다.';}await ensureGPU();if(token!==generation)return;if(isVideo&&(video.readyState<2||!video.videoWidth))return;if(!isVideo||video.paused&&!precisePlaying){busyLayer.hidden=false;busyLayer.textContent='사람과 관절을 분석하는 중';}let snapshot=media;if(isVideo){videoCapture.width=video.videoWidth;videoCapture.height=video.videoHeight;videoCaptureContext.drawImage(video,0,0);snapshot=videoCapture;}const sampleTime=isVideo?video.currentTime:0;const bitmap=await createImageBitmap(snapshot);if(token!==generation){bitmap.close();return;}const result=await rpc('infer',{bitmap,threshold:MoviraMotion.sensitivityThreshold($('confidence').value),detail:$('detail').checked,video:isVideo,time:sampleTime,stream:token,hands:$('showHands').checked,refine:$('refineBody').checked},[bitmap]);if(token!==generation){result.bitmap.close();return;}if(isVideo){analyzedFrame.width=result.bitmap.width;analyzedFrame.height=result.bitmap.height;analyzedFrame.getContext('2d').drawImage(result.bitmap,0,0);hasAnalyzedFrame=true;}result.bitmap.close();poses=result.poses;busyLayer.hidden=true;draw();status(`분석 완료 · ${poses.length}명 · ${result.ms.toFixed(0)} ms · 분석 ${(1000/result.ms).toFixed(1)} FPS${precisePlaying?' · 정밀 재생':''}`);
  }catch(error){if(token===generation){busyLayer.hidden=true;status(error.message);if(isVideo){stopPrecision();video.pause();}}}finally{inferring=false;if(needsInference&&media&&!running&&!precisePlaying)setTimeout(infer,0);}
}
function tick(now){if(isVideo&&media===video&&!video.paused&&!precisePlaying){if(!video.requestVideoFrameCallback&&now-lastDetection>16&&lastDecoded!==video.currentTime){lastDetection=now;lastDecoded=video.currentTime;infer();}$('seek').value=String(video.currentTime/video.duration*1000);$('time').textContent=time(video.currentTime);}requestAnimationFrame(tick);}
requestAnimationFrame(tick);
if(video.requestVideoFrameCallback){const decoded=(now,metadata)=>{if(isVideo&&media===video&&!video.paused&&!precisePlaying&&metadata.mediaTime!==lastDecoded){lastDecoded=metadata.mediaTime;infer();}video.requestVideoFrameCallback(decoded);};video.requestVideoFrameCallback(decoded);}
function stopPrecision(){precisePlaying=false;preciseToken++;$('play').textContent='▶ 재생';}
async function seekVideo(target){if(Math.abs(video.currentTime-target)<.00001&&video.readyState>=2)return;precisionSeeking=true;try{await new Promise((resolve,reject)=>{const timer=setTimeout(()=>{video.removeEventListener('seeked',done);reject(Error('영상 위치 이동이 지연되고 있습니다. 다른 위치에서 다시 재생해 주세요.'));},8000);function done(){clearTimeout(timer);resolve();}video.addEventListener('seeked',done,{once:true});video.currentTime=target;});}finally{precisionSeeking=false;}}
async function playPrecision(){
  video.pause();precisePlaying=true;const token=++preciseToken;$('play').textContent='Ⅱ 일시정지';
  try{await inFlight;if(video.currentTime>=video.duration-.001){generation++;await seekVideo(0);}while(precisePlaying&&token===preciseToken&&media===video){needsInference=false;await infer();if(!precisePlaying||token!==preciseToken)break;let target=video.currentTime+1/30;if(target>video.duration-.0001){if(!$('loop').checked)break;generation++;target=0;}await seekVideo(target);}}
  catch(error){status(error.message);}finally{if(token===preciseToken)stopPrecision();}
}
async function imageFrom(dataURL){const img=new Image();img.src=dataURL;await img.decode();if(img.naturalWidth*img.naturalHeight>20000000)throw Error('사진은 2천만 화소 이하로 준비해 주세요.');return img;}
async function showRecord(record){stopPrecision();const token=++generation;video.pause();const image=await imageFrom(record.dataURL);if(token!==generation)return;isVideo=false;videoSettings.hidden=true;active=record;selected=0;media=image;$('fileName').textContent=record.name;$('viewerTitle').textContent=record.name;$('playback').hidden=true;$('mode').value='label';$('capture').disabled=false;poses=[];refresh();draw();status('정답 편집 모드 · 사람이 옆으로 눕거나 팔을 들어도 실제 관절 위치에 표시하세요.');}
async function openMedia(file){
  stopPrecision();lastDecoded=-1;const token=++generation;video.pause();media=null;active=null;poses=[];placingBox=false;hasAnalyzedFrame=false;if(mediaURL)URL.revokeObjectURL(mediaURL);mediaURL=URL.createObjectURL(file);$('mode').value='detect';
  canvas.hidden=true;$('empty').hidden=true;busyLayer.hidden=false;busyLayer.textContent='파일을 준비하는 중';
  try{isVideo=file.type.startsWith('video/');videoSettings.hidden=!isVideo;if(isVideo){await new Promise((resolve,reject)=>{video.onloadeddata=resolve;video.onerror=()=>reject(Error('브라우저가 이 영상 형식을 재생하지 못했습니다. MP4(H.264) 또는 WebM을 사용하세요.'));video.src=mediaURL;video.loop=$('loop').checked;video.load();});if(token!==generation)return;media=video;$('duration').textContent=time(video.duration);$('time').textContent='0:00';$('seek').value='0';video.playbackRate=Number($('playbackSpeed').value);$('playback').hidden=false;}else{const img=await imageFrom(mediaURL);if(token!==generation)return;media=img;$('playback').hidden=true;}
    $('viewerTitle').textContent=file.name;$('fileName').textContent=file.name;$('capture').disabled=false;refresh();infer();
  }catch(error){if(token===generation){busyLayer.hidden=true;status(error.message);}}
}
$('mediaInput').addEventListener('change',e=>{if(e.target.files[0])openMedia(e.target.files[0]);e.target.value='';});
document.addEventListener('dragover',e=>e.preventDefault());document.addEventListener('drop',e=>{e.preventDefault();if(e.dataTransfer.files[0])openMedia(e.dataTransfer.files[0]);});
$('capture').addEventListener('click',async()=>{if(!media)return;stopPrecision();video.pause();const frame=framePixels(),record={name:`${$('fileName').textContent} @ ${isVideo?time(video.currentTime):'사진'}`,width:frame.width,height:frame.height,dataURL:scratch.toDataURL('image/png'),people:[]};records.push(record);await showRecord(record);placingBox=true;status('사람 전체를 감싸는 사각형을 드래그하세요.');});
$('mode').addEventListener('change',()=>{stopPrecision();video.pause();if($('mode').value==='label'&&!active){status('현재 프레임을 학습 사진으로 추가한 뒤 정답을 표시하세요.');}refresh();infer();});
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
for(const id of ['showOriginal','showBones','showJoints','showInferred'])$(id).addEventListener('change',draw);$('confidence').addEventListener('input',()=>{sensitivityChanged=true;$('confidenceText').textContent=`${$('confidence').value}%`;generation++;infer();});$('detail').addEventListener('change',()=>{generation++;infer();});for(const id of ['showHands','refineBody'])$(id).addEventListener('change',()=>{generation++;infer();});
$('lineWidth').addEventListener('input',draw);
$('play').addEventListener('click',async()=>{if(!isVideo)return;try{if(precisePlaying){stopPrecision();return;}if(!video.paused){video.pause();return;}$('mode').value='detect';refresh();if($('videoMode').value==='precise'){playPrecision();return;}await video.play();$('play').textContent='Ⅱ 일시정지';}catch(error){status(error.message);}});
$('videoMode').addEventListener('change',()=>{stopPrecision();video.pause();});$('playbackSpeed').addEventListener('change',()=>video.playbackRate=Number($('playbackSpeed').value));
$('seek').addEventListener('input',()=>{if(isVideo&&Number.isFinite(video.duration)){stopPrecision();generation++;video.currentTime=Number($('seek').value)/1000*video.duration;}});
video.addEventListener('timeupdate',()=>{$('time').textContent=time(video.currentTime);if(Number.isFinite(video.duration)&&video.duration>0)$('seek').value=String(video.currentTime/video.duration*1000);});
video.addEventListener('seeked',()=>{if(!precisionSeeking||!precisePlaying)infer();});video.addEventListener('pause',()=>{if(!precisePlaying)$('play').textContent='▶ 재생';});video.addEventListener('ended',()=>{if(!precisePlaying){$('play').textContent='▶ 재생';infer();}});$('loop').addEventListener('change',()=>video.loop=$('loop').checked);
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
  if(model&&!sensitivityChanged){$('confidence').value=String(Math.round(MoviraMotion.thresholdSensitivity(model.threshold)));$('confidenceText').textContent=`${$('confidence').value}%`;}
  if(model&&Number($('progress').value)===0)$('trainStatus').textContent='공개 가중치에 직접 학습한 인물 판별 보정을 적용합니다. 새 데이터로 보정 학습을 추가할 수 있습니다.';
  $('saveModel').disabled=!model;$('modelStatus').textContent=model?`Movira 판별 보정 · ${model.steps.toLocaleString()}회 갱신 · ${model.meta.trainingExamples?.toLocaleString()||0}개 예제`:'기본 인식 모델 · 첫 파일 선택 시 준비';
  const test=model?.meta?.independentTest||model?.meta?.test,label=model?.meta?.independentTest?'별도 검증':'보정층 내부 분리 시험';
  $('validation').textContent=test?.images?`정지 사진 ${label} ${test.images}장 · 검출 정밀도 ${(test.precision*100).toFixed(1)}% · 재현율 ${(test.recall*100).toFixed(1)}%. 작은·중간·큰 인물 재현율: ${['small','medium','large'].map(k=>test.bySize[k].recall===null?'측정 없음':(test.bySize[k].recall*100).toFixed(1)+'%').join(' / ')}. 실제 결과는 영상과 인물 크기에 따라 달라집니다.`:'아주 작거나 흐린 사람은 놓칠 수 있습니다. 관절 위치에는 오차가 생길 수 있습니다.';
}
$('modelInput').addEventListener('change',async e=>{try{if(e.target.files[0]){model=MoviraCalibration.Calibration.import(await readJSON(e.target.files[0]));if(worker){await ensureGPU();await rpc('calibration',{calibration:model.export()});}modelInfo();generation++;infer();}}catch(error){status(error.message);}e.target.value='';});$('saveModel').addEventListener('click',()=>{if(model)download('movira-person-calibration.json',model.export());});
$('stopTrain').addEventListener('click',()=>{stop=true;worker?.postMessage({type:'stop'});});
$('train').addEventListener('click',async()=>{
  if(running)return;if(records.length<3){status('서로 다른 학습 사진을 최소 3장 준비하고 사람 영역 정답을 지정하세요.');return;}
  running=true;stop=false;generation++;stopPrecision();video.pause();$('train').disabled=true;$('stopTrain').disabled=false;for(const id of ['mediaInput','dataInput','modelInput'])$(id).disabled=true;
  try{await ensureGPU();busyLayer.hidden=true;status('사람 판별 보정을 학습합니다. 기본 관절 가중치는 변경하지 않습니다.');const trained=await rpc('train',{records,epochs:Math.max(1,Math.min(500,Number($('epochs').value)||160))});model=MoviraCalibration.Calibration.import(trained.calibration);modelInfo();status('인물 판별 보정 학습을 완료했습니다. 저장 버튼으로 보관할 수 있습니다.');
  }catch(error){status(error.message);}finally{running=false;$('train').disabled=false;$('stopTrain').disabled=true;}
  for(const id of ['mediaInput','dataInput','modelInput'])$(id).disabled=false;
});
// Only the small, locally trained score adaptation is read here. Media stays local.
if(location.protocol!=='file:')fetch('./calibration.json',{cache:'no-cache'}).then(r=>r.ok?r.json():null).then(data=>{if(data&&!model){model=MoviraCalibration.Calibration.import(data);modelInfo();}}).catch(()=>{});
window.addEventListener('beforeunload',()=>{worker?.terminate();if(mediaURL)URL.revokeObjectURL(mediaURL);});
refresh();modelInfo();
