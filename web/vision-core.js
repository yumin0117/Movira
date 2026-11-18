/* Movira image geometry, detections, and native pose decoding. Original code. */
(function(scope){
'use strict';
const clamp=(n,l,h)=>Math.max(l,Math.min(h,n));
function bodyEvidence(person){const core=[0,1,2,3,8,9,10,11,12,13,14];return core.filter(k=>person.points[k]?.confidence>=.25).length>=3;}
function mergeBodyFragments(people){const out=[];for(const p of people){let merged=false;for(const q of out){const h=Math.min(p.box.h,q.box.h);if(p.points[0].confidence>=.3&&q.points[0].confidence>=.3)continue;const hips=[8,9,10];if(!hips.every(k=>p.points[k].confidence>=.25&&q.points[k].confidence>=.25&&Math.hypot(p.points[k].x-q.points[k].x,p.points[k].y-q.points[k].y)<h*.12))continue;if(iou(p.box,q.box)<.03)continue;const x=Math.min(p.box.x,q.box.x),y=Math.min(p.box.y,q.box.y),right=Math.max(p.box.x+p.box.w,q.box.x+q.box.w),bottom=Math.max(p.box.y+p.box.h,q.box.y+q.box.h);q.box={x,y,w:right-x,h:bottom-y};q.points=q.points.map((a,k)=>p.points[k].confidence>a.confidence?p.points[k]:a);q.confidence=Math.max(q.confidence,p.confidence);merged=true;break;}if(!merged)out.push(p);}return out;}
function sample(frame,x,y,c,border=0){
  const {pixels,width,height,bgra=false}=frame;x=Number(x);y=Number(y);const x0=Math.floor(x),y0=Math.floor(y),dx=x-x0,dy=y-y0,channel=bgra?2-c:c;
  const get=(xx,yy)=>xx<0||yy<0||xx>=width||yy>=height?border:pixels[(yy*width+xx)*4+channel];
  return (get(x0,y0)*(1-dx)+get(x0+1,y0)*dx)*(1-dy)+(get(x0,y0+1)*(1-dx)+get(x0+1,y0+1)*dx)*dy;
}
function detectorInput(frame,view,size=416){
  const ratio=Math.min(size/view.w,size/view.h),rw=Math.max(1,Math.floor(view.w*ratio)),rh=Math.max(1,Math.floor(view.h*ratio)),sx=rw/view.w,sy=rh/view.h,plane=size*size,input=new Float32Array(plane*3);input.fill(114);
  for(let y=0;y<rh;y++)for(let x=0;x<rw;x++){const px=view.x+(x+.5)/sx-.5,py=view.y+(y+.5)/sy-.5;for(let c=0;c<3;c++)input[c*plane+y*size+x]=sample(frame,px,py,2-c,114);}
  return {input,transform:{...view,sx,sy,size}};
}
function views(width,height,detail=true){
  const result=[{x:0,y:0,w:width,h:height,tile:false}];if(!detail||Math.min(width,height)<180)return result;
  const side=Math.min(Math.min(width,height),Math.max(128,Math.max(width,height)*.55));
  function positions(extent){if(extent<=side*1.1)return [Math.max(0,(extent-side)/2)];const values=[];for(let x=0;x<extent-side;x+=side*.72)values.push(x);const end=extent-side;if(values.length&&end-values.at(-1)<side*.2)values[values.length-1]=end;else values.push(end);return values;}
  for(const x of positions(width))for(const y of positions(height))result.push({x,y,w:side,h:side,tile:true});
  return result;
}
function intersection(a,b){return Math.max(0,Math.min(a.x+a.w,b.x+b.w)-Math.max(a.x,b.x))*Math.max(0,Math.min(a.y+a.h,b.y+b.h)-Math.max(a.y,b.y));}
function iou(a,b){const value=intersection(a,b);return value/Math.max(1e-8,a.w*a.h+b.w*b.h-value);}
function nms(candidates,threshold=.45){
  const sorted=[...candidates].sort((a,b)=>b.confidence-a.confidence),kept=[];for(const p of sorted)if(!kept.some(q=>iou(p.box,q.box)>threshold))kept.push(p);return kept;
}
function decodePeople(data,transform,imageWidth,imageHeight,minimum=.04,columns=6){
  const result=[];let row=0;for(const stride of [8,16,32]){const cells=transform.size/stride;for(let gy=0;gy<cells;gy++)for(let gx=0;gx<cells;gx++,row++){
    const at=row*columns,raw=data[at+4]*data[at+5];if(!Number.isFinite(raw)||raw<minimum)continue;
    const cx=(data[at]+gx)*stride,cy=(data[at+1]+gy)*stride,w=Math.exp(clamp(data[at+2],-12,12))*stride,h=Math.exp(clamp(data[at+3],-12,12))*stride;
    const left=transform.x+(cx-w/2)/transform.sx,top=transform.y+(cy-h/2)/transform.sy,right=left+w/transform.sx,bottom=top+h/transform.sy;
    const box={x:clamp(left,0,imageWidth),y:clamp(top,0,imageHeight)};box.w=clamp(right,0,imageWidth)-box.x;box.h=clamp(bottom,0,imageHeight)-box.y;if(box.w<8||box.h<14)continue;
    const edge=Math.min(cx-w/2,cy-h/2,transform.w*transform.sx-(cx+w/2),transform.h*transform.sy-(cy+h/2));
    result.push({box,raw,confidence:raw,tile:Boolean(transform.tile),viewFraction:transform.h/imageHeight,nearEdge:edge<5});
  }}return result;
}
function poseCrop(box){let w=box.w*1.25,h=box.h*1.25;if(w/h>.75)h=w/.75;else w=h*.75;return {cx:box.x+box.w/2,cy:box.y+box.h/2,w,h};}
function refinementRegions(person,width,height){const b=person.box,upper={x:b.x,y:b.y,w:b.w,h:b.h*.68},lower={x:b.x,y:b.y+b.h*.38,w:b.w,h:b.h*.62};const hip=person.points[8];if(hip?.confidence>.4){upper.h=Math.max(b.h*.4,Math.min(b.h*.8,hip.y-b.y+b.h*.07));lower.y=Math.max(b.y,b.y+Math.min(b.h*.5,hip.y-b.y-b.h*.1));lower.h=b.y+b.h-lower.y;}for(const region of [upper,lower]){region.x=Math.max(0,region.x);region.y=Math.max(0,region.y);region.w=Math.min(region.w,width-region.x);region.h=Math.min(region.h,height-region.y);}return [{box:upper,indices:[0,1,2,3,4,5,6,7],anchors:[0,1,2,3]},{box:lower,indices:[8,9,10,11,12,13,14,15,16,17,18],anchors:[8,9,10]}];}
function poseInput(frame,box){
  const crop=poseCrop(box),width=192,height=256,plane=width*height,result=new Float32Array(plane*3),mean=[123.675,116.28,103.53],std=[58.395,57.12,57.375];
  for(let y=0;y<height;y++)for(let x=0;x<width;x++){const px=crop.cx+(x-width/2)*crop.w/width,py=crop.cy+(y-height/2)*crop.h/height;for(let c=0;c<3;c++)result[c*plane+y*width+x]=(sample(frame,px,py,c)-mean[c])/std[c];}
  return {input:result,crop};
}
function decodePose(x,y,crop,imageWidth,imageHeight){
  if(x.length!==26*384||y.length!==26*512)throw Error('관절 모델의 출력 형식이 다릅니다.');
  const points=[];for(let k=0;k<26;k++){let ix=0,iy=0,mx=-Infinity,my=-Infinity;for(let j=0;j<384;j++)if(x[k*384+j]>mx){mx=x[k*384+j];ix=j;}for(let j=0;j<512;j++)if(y[k*512+j]>my){my=y[k*512+j];iy=j;}
    const p={x:crop.cx-crop.w/2+ix/384*crop.w,y:crop.cy-crop.h/2+iy/512*crop.h,confidence:clamp(Math.min(mx,my),0,1)};if(ix<2||ix>381||iy<2||iy>509||!Number.isFinite(p.x+p.y+p.confidence)||p.x<0||p.y<0||p.x>=imageWidth||p.y>=imageHeight)p.confidence=0;points.push(p);
  }
  const order=[0,18,5,6,7,8,9,10,19,11,12,13,14,15,16,24,25,20,21];return order.map(i=>points[i]);
}
function deduplicatePoses(poses){
  const result=[];for(const p of [...poses].sort((a,b)=>b.confidence-a.confidence)){
    const duplicate=result.some(q=>{const boxOverlap=iou(p.box,q.box),h=Math.min(p.box.h,q.box.h);let count=0,dist=0;for(const k of [0,1,8,9,10]){const a=p.points[k],b=q.points[k];if(a.confidence>=.25&&b.confidence>=.25){dist+=Math.hypot(a.x-b.x,a.y-b.y);count++;}}if(boxOverlap>=.12&&count>=3&&dist/count<h*.1)return true;let upperCount=0,upperDistance=0;for(const k of [0,1,2,3]){const a=p.points[k],b=q.points[k];if(a.confidence>=.35&&b.confidence>=.35){upperDistance+=Math.hypot(a.x-b.x,a.y-b.y);upperCount++;}}const sameFace=p.points[0].confidence>.35&&q.points[0].confidence>.35&&Math.hypot(p.points[0].x-q.points[0].x,p.points[0].y-q.points[0].y)<h*.025;if(sameFace&&upperCount>=3&&upperDistance/upperCount<h*.065)return true;const conflictingFaces=p.points[0].confidence>.35&&q.points[0].confidence>.35&&Math.hypot(p.points[0].x-q.points[0].x,p.points[0].y-q.points[0].y)>h*.08;if(conflictingFaces||!(p.nearEdge||q.nearEdge))return false;for(const chain of [[2,4,6],[3,5,7]]){let sum=0,valid=0;for(const k of chain){const a=p.points[k],b=q.points[k];if(a.confidence>.35&&b.confidence>.35){sum+=Math.hypot(a.x-b.x,a.y-b.y);valid++;}}if(valid===3&&sum/3<h*.06)return true;}return false;});if(!duplicate)result.push(p);
  }return result;
}
const api={bodyEvidence,mergeBodyFragments,sample,detectorInput,views,intersection,iou,nms,decodePeople,poseCrop,refinementRegions,poseInput,decodePose,deduplicatePoses};scope.MoviraVision=api;if(typeof module!=='undefined')module.exports=api;
})(globalThis);
