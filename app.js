'use strict';
const $=id=>document.getElementById(id);
const canvas=$('resultCanvas'),ctx=canvas.getContext('2d'),video=$('sourceVideo');
let worker,initializing,ready=false,nextId=0;
const pending=new Map();
let source=null,kind=null,mediaURL=null,version=0,busy=false,needsAnalysis=false,timer=null;
let lastBitmap=null,lastPoses=[],lastMs=0,showedResult=false,toastTimer;
const settings={bones:true,joints:true,original:true,threshold:.35,width:3};
const allowedImage=/\.(jpe?g|png|webp|bmp|gif|avif)$/i,allowedVideo=/\.(mp4|webm|mov|m4v|avi|mkv)$/i;
function toast(message){$('toast').textContent=message;$('toast').hidden=false;clearTimeout(toastTimer);toastTimer=setTimeout(()=>$('toast').hidden=true,8000);}
function status(message){$('statusText').textContent=message;}
function loading(active,text){$('busyOverlay').hidden=!active;$('stage').setAttribute('aria-busy',String(active));if(text)$('busyText').textContent=text;}
function setEngine(text){$('engineValue').replaceChildren();const dot=document.createElement('i');$('engineValue').append(dot,document.createTextNode(text));}
function rpc(type,extra={},transfer=[]){return new Promise((resolve,reject)=>{const id=++nextId;pending.set(id,{resolve,reject});worker.postMessage({id,type,...extra},transfer);});}
async function ensureModel(){
  if(ready)return;
  if(initializing)return initializing;
  loading(true,'분석 모델 불러오는 중');setEngine('준비 중');
  if(!worker){
    worker=new Worker('./worker.js');
    worker.onmessage=({data})=>{
      if(data.type==='progress'){$('busyText').textContent=data.label||(data.percent===100?'분석 엔진 준비 중':`모델 불러오는 중 · ${data.percent}%`);return;}
      const request=pending.get(data.id);if(!request){data.bitmap?.close();return;}pending.delete(data.id);
      if(data.type==='error')request.reject(Error(data.message));else request.resolve(data);
    };
    worker.onerror=event=>{
      for(const request of pending.values())request.reject(Error('분석 엔진을 시작하지 못했습니다. 최신 Chrome 또는 Edge에서 다시 시도해 주세요.'));
      pending.clear();worker?.terminate();worker=null;ready=false;event.preventDefault();
    };
  }
  initializing=rpc('init').then(()=>{ready=true;setEngine('준비 완료');}).finally(()=>{initializing=null;});
  return initializing;
}
function formatTime(seconds){if(!Number.isFinite(seconds))return '0:00';return `${Math.floor(seconds/60)}:${String(Math.floor(seconds%60)).padStart(2,'0')}`;}
function updateControls(){
  $('playPause').textContent=video.paused?'▶':'Ⅱ';$('playPause').setAttribute('aria-label',video.paused?'재생':'일시정지');
  $('currentTime').textContent=formatTime(video.currentTime);
  if(Number.isFinite(video.duration)&&video.duration>0)$('timeline').value=String(video.currentTime/video.duration*1000);
}
function showPlayState(){updateControls();if(kind==='video'){setEngine(video.paused?'일시정지':'분석 중');status(video.paused?'일시정지 · 재생 버튼을 눌러 계속하세요.':'영상을 분석하고 있습니다.');}}
function visible(joint){return joint.confidence>=settings.threshold&&Number.isFinite(joint.x)&&Number.isFinite(joint.y);}
function render(){
  if(!lastBitmap)return;
  const scale=Math.min(1,1920/lastBitmap.width,1080/lastBitmap.height);
  const w=Math.round(lastBitmap.width*scale),h=Math.round(lastBitmap.height*scale);
  if(canvas.width!==w||canvas.height!==h){canvas.width=w;canvas.height=h;}
  ctx.fillStyle='#10191c';ctx.fillRect(0,0,w,h);
  if(settings.original)ctx.drawImage(lastBitmap,0,0,w,h);
  const line=Math.max(1.25,settings.width*w/850), radius=Math.max(2.5,line*1.35);
  ctx.lineWidth=line;ctx.lineCap='round';ctx.lineJoin='round';
  const drawLine=(a,b)=>{ctx.beginPath();ctx.moveTo(a.x*scale,a.y*scale);ctx.lineTo(b.x*scale,b.y*scale);ctx.stroke();};
  for(const pose of lastPoses){
    ctx.strokeStyle='#b9f875';
    if(settings.bones){
      for(const [a,b] of MoviraPose.bones)if(visible(pose.joints[a])&&visible(pose.joints[b]))drawLine(pose.joints[a],pose.joints[b]);
      const head=MoviraPose.head(pose.joints);
      if(head&&visible(head.neck)&&visible(head.face))drawLine(head.neck,head.face);
      const joints=pose.joints;
      if([5,6,11,12].every(i=>visible(joints[i])))drawLine({x:(joints[5].x+joints[6].x)/2,y:(joints[5].y+joints[6].y)/2},{x:(joints[11].x+joints[12].x)/2,y:(joints[11].y+joints[12].y)/2});
    }
    const head=MoviraPose.head(pose.joints);
    const displayedJoints=[...pose.joints.slice(5),...(head?[head.face,head.neck]:[])];
    for(const foot of pose.feet||[]){
      const ankle=pose.joints[foot.ankleIndex];
      if(visible(ankle)&&visible(foot.heel)&&visible(foot.toe)){
        if(settings.bones){ctx.strokeStyle='#b9f875';drawLine(ankle,foot.heel);drawLine(foot.heel,foot.toe);drawLine(foot.toe,ankle);}
        displayedJoints.push(foot.heel,foot.toe);
      }
    }
    if(settings.joints)for(const joint of displayedJoints)if(visible(joint)){
      ctx.beginPath();ctx.arc(joint.x*scale,joint.y*scale,radius+1,0,Math.PI*2);ctx.fillStyle='#122d20';ctx.fill();
      ctx.beginPath();ctx.arc(joint.x*scale,joint.y*scale,radius,0,Math.PI*2);ctx.fillStyle='#93f1d8';ctx.fill();
    }
  }
}
function schedule(){clearTimeout(timer);timer=setTimeout(requestAnalysis,kind==='video'&&!video.paused?8:0);}
function requestAnalysis(){needsAnalysis=true;if(!busy)analyze();}
async function analyze(){
  if(!source||busy)return;
  busy=true;needsAnalysis=false;const current=version;
  try{
    await ensureModel();if(current!==version)return;
    if(kind==='video'&&video.readyState<2)return;
    const bitmap=await createImageBitmap(source);if(current!==version){bitmap.close();return;}
    const result=await rpc('infer',{bitmap},[bitmap]);
    if(current!==version){result.bitmap.close();return;}
    lastBitmap?.close();lastBitmap=result.bitmap;lastPoses=result.poses;lastMs=result.ms;
    render();$('emptyState').hidden=true;canvas.hidden=false;$('downloadBtn').disabled=false;
    $('peopleCount').textContent=String(lastPoses.length);$('fpsValue').textContent=(1000/lastMs).toFixed(1);
    $('resolutionValue').textContent=`${lastBitmap.width} × ${lastBitmap.height}`;
    $('liveDot').classList.add('active');setEngine(kind==='image'?'분석 완료':video.paused?'일시정지':'분석 중');loading(false);showedResult=true;
    const footCount=lastPoses.reduce((count,pose)=>count+(pose.feet||[]).filter(foot=>visible(pose.joints[foot.ankleIndex])&&visible(foot.heel)&&visible(foot.toe)).length,0);
    if(kind==='image')status(lastPoses.length?`사진 분석 완료 · ${lastPoses.length}명 · 발 ${footCount}개 검출`:'사람을 찾지 못했습니다. 더 크고 선명한 사진을 선택해 주세요.');
    else{updateControls();if(video.paused)status('일시정지 · 재생 버튼을 눌러 계속하세요.');else status(lastPoses.length?'영상의 움직임을 분석하고 있습니다.':'영상을 분석 중입니다. 현재 프레임에서 사람을 찾지 못했습니다.');}
  }catch(error){
    if(current===version){loading(false);toast(error.message);status('분석하지 못했습니다. 파일을 다시 선택해 주세요.');setEngine('확인 필요');if(kind==='video')video.pause();}
  }finally{
    busy=false;
    if(source&&(needsAnalysis||(kind==='video'&&!video.paused&&!video.ended)))schedule();
  }
}
function resetMedia(){
  version++;clearTimeout(timer);video.pause();video.removeAttribute('src');video.load();source=null;
  if(mediaURL){URL.revokeObjectURL(mediaURL);mediaURL=null;}
  lastBitmap?.close();lastBitmap=null;lastPoses=[];showedResult=false;
  canvas.hidden=true;$('emptyState').hidden=false;$('downloadBtn').disabled=true;
  $('peopleCount').textContent='—';$('fpsValue').textContent='—';$('resolutionValue').textContent='—';
  $('liveDot').classList.remove('active');$('videoControls').hidden=true;loading(false);
}
async function openFile(file){
  if(!file)return;
  const image=file.type.startsWith('image/')||allowedImage.test(file.name);
  const movie=file.type.startsWith('video/')||allowedVideo.test(file.name);
  if(!image&&!movie){toast('사진 또는 동영상 파일을 선택해 주세요.');return;}
  resetMedia();const current=version;kind=image?'image':'video';mediaURL=URL.createObjectURL(file);
  $('fileInfo').hidden=false;$('fileName').textContent=file.name;$('fileName').title=file.name;
  $('fileDetails').textContent=`${image?'사진':'동영상'} · ${(file.size/1024/1024).toFixed(1)} MB`;
  $('viewerTitle').textContent=file.name;$('modeTag').textContent=image?'PHOTO':'VIDEO';status('파일을 여는 중입니다.');
  loading(true,'파일을 준비하는 중');
  try{
    if(image){
      const picture=new Image();picture.src=mediaURL;await picture.decode();if(current!==version)return;source=picture;requestAnalysis();
    }else{
      await new Promise((resolve,reject)=>{video.onloadeddata=resolve;video.onerror=()=>reject(Error('이 동영상을 재생할 수 없습니다. MP4(H.264) 또는 WebM 파일로 다시 시도해 주세요.'));video.src=mediaURL;video.loop=$('repeatBtn').getAttribute('aria-pressed')==='true';video.load();});
      if(current!==version)return;
      if(!Number.isFinite(video.duration)||video.duration<=0)throw Error('영상 길이를 읽을 수 없습니다. 다른 동영상 파일을 선택해 주세요.');
      source=video;$('videoControls').hidden=false;$('duration').textContent=formatTime(video.duration);updateControls();
      requestAnalysis();
    }
  }catch(error){if(current===version){loading(false);status('파일을 열지 못했습니다.');toast(error.message);setEngine('확인 필요');}}
}
function chooseFile(){$('fileInput').click();}
for(const id of ['dropZone','emptyUpload','changeFile'])$(id).addEventListener('click',chooseFile);
$('fileInput').addEventListener('change',event=>{openFile(event.target.files[0]);event.target.value='';});
for(const target of [$('dropZone'),$('stage')]){
  target.addEventListener('dragover',event=>{event.preventDefault();target.classList.add('dragging');});
  target.addEventListener('dragleave',()=>target.classList.remove('dragging'));
  target.addEventListener('drop',event=>{event.preventDefault();target.classList.remove('dragging');if(event.dataTransfer.files.length>1)toast('한 번에 파일 하나씩 분석합니다. 첫 번째 파일을 엽니다.');openFile(event.dataTransfer.files[0]);});
}
window.addEventListener('dragover',event=>event.preventDefault());window.addEventListener('drop',event=>event.preventDefault());
for(const [id,key] of [['showBones','bones'],['showJoints','joints'],['showOriginal','original']])$(id).addEventListener('change',event=>{settings[key]=event.target.checked;render();});
$('jointThreshold').addEventListener('input',event=>{settings.threshold=Number(event.target.value)/100;$('thresholdValue').textContent=`${event.target.value}%`;render();});
$('lineWidth').addEventListener('input',event=>{settings.width=Number(event.target.value);$('lineValue').textContent=event.target.value;render();});
$('resetSettings').addEventListener('click',()=>{Object.assign(settings,{bones:true,joints:true,original:true,threshold:.35,width:3});for(const id of ['showBones','showJoints','showOriginal'])$(id).checked=true;$('jointThreshold').value='35';$('thresholdValue').textContent='35%';$('lineWidth').value='3';$('lineValue').textContent='3';render();});
$('playPause').addEventListener('click',async()=>{
  if(!source||kind!=='video')return;
  if(video.paused){try{if(video.ended)video.currentTime=0;await video.play();showPlayState();requestAnalysis();}catch(error){toast('재생을 시작하지 못했습니다. 다시 눌러 주세요.');}}
  else{video.pause();showPlayState();}
});
$('timeline').addEventListener('input',event=>{if(kind!=='video'||!source)return;version++;video.currentTime=Number(event.target.value)/1000*video.duration;updateControls();});
video.addEventListener('seeked',()=>{if(source===video)requestAnalysis();});
video.addEventListener('ended',()=>{showPlayState();status('영상이 끝났습니다. 재생 버튼을 누르면 다시 시작합니다.');});
video.addEventListener('timeupdate',updateControls);
$('repeatBtn').addEventListener('click',()=>{const on=$('repeatBtn').getAttribute('aria-pressed')!=='true';$('repeatBtn').setAttribute('aria-pressed',String(on));video.loop=on;});
$('downloadBtn').addEventListener('click',()=>{if(!lastBitmap)return;canvas.toBlob(blob=>{if(!blob)return;const url=URL.createObjectURL(blob),link=document.createElement('a');link.href=url;link.download=`movira-${Date.now()}.png`;link.click();setTimeout(()=>URL.revokeObjectURL(url),1500);},'image/png');});
window.addEventListener('beforeunload',()=>{video.pause();worker?.terminate();lastBitmap?.close();if(mediaURL)URL.revokeObjectURL(mediaURL);});
if(!window.Worker||!window.OffscreenCanvas||!window.createImageBitmap){status('최신 Chrome 또는 Edge에서 열어 주세요.');toast('이 브라우저에서는 분석 기능을 지원하지 않습니다. 최신 Chrome 또는 Edge를 사용해 주세요.');}
