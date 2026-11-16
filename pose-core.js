/* Movira pose geometry. AGPL-3.0. Shared by the inference worker and tests. */
(() => {
  // Body only. The head has one central connection, never ear-to-shoulder edges.
  const bones = [[5,6],[5,7],[7,9],[6,8],[8,10],[5,11],[6,12],[11,12],[11,13],[13,15],[12,14],[14,16]];
  function letterbox(width, height, size = 640) {
    const ratio = Math.min(size / width, size / height);
    const scaledWidth = Math.round(width * ratio), scaledHeight = Math.round(height * ratio);
    return {width,height,scaledWidth,scaledHeight,left:Math.floor((size-scaledWidth)/2),top:Math.floor((size-scaledHeight)/2)};
  }
  function restore(x,y,t) { return {x:(x-t.left)*t.width/t.scaledWidth, y:(y-t.top)*t.height/t.scaledHeight}; }
  function iou(a,b) {
    const intersection = Math.max(0,Math.min(a.right,b.right)-Math.max(a.left,b.left)) * Math.max(0,Math.min(a.bottom,b.bottom)-Math.max(a.top,b.top));
    return intersection / Math.max(1e-6,(a.right-a.left)*(a.bottom-a.top)+(b.right-b.left)*(b.bottom-b.top)-intersection);
  }
  function decode(data,dims,t,confidence=.35,nms=.45) {
    if(dims.length!==3 || dims[0]!==1 || (dims[1]!==56 && dims[2]!==56)) throw Error('지원하지 않는 포즈 모델 출력입니다.');
    const first=dims[1]===56, count=first?dims[2]:dims[1];
    const value=(i,c)=>data[first?c*count+i:i*56+c];
    const candidates=[];
    for(let i=0;i<count;i++) {
      const score=value(i,4), x=value(i,0), y=value(i,1), w=value(i,2), h=value(i,3);
      if(!Number.isFinite(score)||score<confidence||!Number.isFinite(x+y+w+h)||w<=0||h<=0) continue;
      const a=restore(x-w/2,y-h/2,t), b=restore(x+w/2,y+h/2,t);
      const joints=Array.from({length:17},(_,k)=>({...restore(value(i,5+k*3),value(i,6+k*3),t),confidence:value(i,7+k*3)}));
      candidates.push({left:a.x,top:a.y,right:b.x,bottom:b.y,confidence:score,joints});
    }
    candidates.sort((a,b)=>b.confidence-a.confidence);
    const result=[];
    for(const pose of candidates) if(!result.some(kept=>iou(pose,kept)>nms)) result.push(pose);
    return result;
  }
  function head(joints) {
    const usable=j=>j&&Number.isFinite(j.x)&&Number.isFinite(j.y)&&Number.isFinite(j.confidence);
    const l=joints[5],r=joints[6];
    if(!usable(l)||!usable(r))return null;
    const neck={x:(l.x+r.x)/2,y:(l.y+r.y)/2,confidence:Math.min(l.confidence,r.confidence)};
    const nose=joints[0];
    let face=usable(nose)?nose:null;
    // The eyes are internal fallback measurements, not separate displayed joints.
    if((!face||face.confidence<.2)&&usable(joints[1])&&usable(joints[2]))face={x:(joints[1].x+joints[2].x)/2,y:(joints[1].y+joints[2].y)/2,confidence:Math.min(joints[1].confidence,joints[2].confidence)};
    return face?{neck,face}:null;
  }
  function footCrop(pose) {
    const hipA=pose.joints[11],hipB=pose.joints[12],sA=pose.joints[5],sB=pose.joints[6];
    const cx=(hipA.x+hipB.x)/2,cy=(hipA.y+hipB.y)/2;
    const dx=(sA.x+sB.x)/2-cx,dy=(sA.y+sB.y)/2-cy;
    const angle=Math.atan2(-dx,-dy),cos=Math.cos(angle),sin=Math.sin(angle);
    const radius=Math.max((pose.bottom-pose.top)*.55,cy-pose.top,pose.bottom-cy,(pose.right-pose.left)*.55)*1.12;
    return {cx,cy,cos,sin,side:Math.max(32,radius*2)};
  }
  function restoreFoot(x,y,crop) {
    const px=(x-128)*crop.side/256,py=(y-128)*crop.side/256;
    return {x:crop.cx+crop.cos*px+crop.sin*py,y:crop.cy-crop.sin*px+crop.cos*py};
  }
  function decodeFeet(data,presence,crop,pose) {
    const sigmoid=x=>1/(1+Math.exp(-x));
    if(presence<.5||data.length!==195)return [];
    const joint=index=>({...restoreFoot(data[index*5],data[index*5+1],crop),confidence:Math.min(sigmoid(data[index*5+3]),sigmoid(data[index*5+4]),presence)});
    const hipA=joint(23),hipB=joint(24);
    if(Math.hypot((hipA.x+hipB.x)/2-crop.cx,(hipA.y+hipB.y)/2-crop.cy)>crop.side*.08)return [];
    const feet=[];
    for(const [ankleIndex,mpAnkle,mpHeel,mpToe] of [[15,27,29,31],[16,28,30,32]]) {
      const anchor=pose.joints[ankleIndex],ankle=joint(mpAnkle),heel=joint(mpHeel),toe=joint(mpToe);
      const bound=(pose.bottom-pose.top)*.2;
      if(anchor.confidence<.2||!Number.isFinite(heel.x+heel.y+toe.x+toe.y))continue;
      if(Math.hypot(ankle.x-anchor.x,ankle.y-anchor.y)>bound||Math.hypot(heel.x-anchor.x,heel.y-anchor.y)>bound||Math.hypot(toe.x-anchor.x,toe.y-anchor.y)>bound)continue;
      // Reject displaced feet (common when another person's leg overlaps the crop).
      // A heel remains close to the ankle; a foot must not jump back up the shin.
      const knee=pose.joints[ankleIndex===15?13:14];
      const shin=knee.confidence>=.2?Math.hypot(anchor.x-knee.x,anchor.y-knee.y):(pose.bottom-pose.top)*.23;
      if(Math.hypot(heel.x-anchor.x,heel.y-anchor.y)>Math.max(9,shin*.3)||Math.hypot(toe.x-anchor.x,toe.y-anchor.y)>Math.max(14,shin*.7))continue;
      if(knee.confidence>=.2&&shin>1){
        const heelForward=((heel.x-anchor.x)*(anchor.x-knee.x)+(heel.y-anchor.y)*(anchor.y-knee.y))/shin;
        const toeForward=((toe.x-anchor.x)*(anchor.x-knee.x)+(toe.y-anchor.y)*(anchor.y-knee.y))/shin;
        if(heelForward < -shin*.2||toeForward < -shin*.25)continue;
      }
      heel.confidence=Math.min(heel.confidence,anchor.confidence);toe.confidence=Math.min(toe.confidence,anchor.confidence);
      feet.push({ankleIndex,heel,toe});
    }
    return feet;
  }
  const api={bones,letterbox,restore,iou,decode,head,footCrop,restoreFoot,decodeFeet};
  globalThis.MoviraPose=api;
  if(typeof module!=='undefined'&&module.exports) module.exports=api;
})();
