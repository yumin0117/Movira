const test=require('node:test');
const assert=require('node:assert/strict');
const {letterbox,restore,decode,bones,head,footCrop,restoreFoot,decodeFeet}=require('../web/pose-core.js');
test('letterbox restores widescreen and odd-sized images',()=>{
  assert.deepEqual(restore(0,140,letterbox(1280,720)),{x:0,y:0});
  assert.deepEqual(restore(640,500,letterbox(1280,720)),{x:1280,y:720});
  const t=letterbox(777,431);assert.equal(restore(t.left+t.scaledWidth,t.top+t.scaledHeight,t).y,431);
});
function fixture(){
  const data=new Float32Array(56*4);
  for(let i=0;i<4;i++){
    data[i]=i===2?450:100+i;data[4+i]=300;data[8+i]=100;data[12+i]=300;data[16+i]=i===3?.1:.9-i*.05;
    for(let k=0;k<17;k++){data[(5+k*3)*4+i]=data[i];data[(6+k*3)*4+i]=300;data[(7+k*3)*4+i]=.9;}
  }return data;
}
test('keeps several people while removing duplicate and weak boxes',()=>{
  const people=decode(fixture(),[1,56,4],letterbox(640,640));
  assert.equal(people.length,2);assert.equal(people[1].joints[0].x,450);assert.equal(people[0].joints.length,17);
});
test('supports both raw output orientations',()=>{
  const data=fixture(),transposed=new Float32Array(data.length);
  for(let i=0;i<4;i++)for(let c=0;c<56;c++)transposed[i*56+c]=data[c*4+i];
  assert.deepEqual(decode(transposed,[1,4,56],letterbox(640,640)),decode(data,[1,56,4],letterbox(640,640)));
});
test('rejects end-to-end output and ignores nonfinite candidates',()=>{
  assert.throws(()=>decode(new Float32Array(300*57),[1,300,57],letterbox(640,640)));
  const data=fixture();data[16]=NaN;data[17]=NaN;assert.equal(decode(data,[1,56,4],letterbox(640,640)).length,1);
});
test('head has one neck-to-face segment without shoulder-to-ear edges',()=>{
  assert.ok(bones.every(edge=>edge.every(index=>index>=5)));
  const pose=decode(fixture(),[1,56,4],letterbox(640,640))[0];
  const result=head(pose.joints);assert.ok(result);assert.equal(result.face,pose.joints[0]);
  assert.equal(result.neck.x,(pose.joints[5].x+pose.joints[6].x)/2);
});
function feetFixture(){
  const joints=Array.from({length:17},()=>({x:100,y:150,confidence:.9}));
  Object.assign(joints[5],{x:80,y:100});Object.assign(joints[6],{x:120,y:100});
  Object.assign(joints[11],{x:85,y:180});Object.assign(joints[12],{x:115,y:180});
  Object.assign(joints[15],{x:85,y:320});Object.assign(joints[16],{x:115,y:320});
  const pose={left:50,right:150,top:40,bottom:350,joints},crop=footCrop(pose),data=new Float32Array(195);
  const set=(index,x,y)=>{
    const dx=x-crop.cx,dy=y-crop.cy,scale=256/crop.side;
    data[index*5]=128+(crop.cos*dx-crop.sin*dy)*scale;
    data[index*5+1]=128+(crop.sin*dx+crop.cos*dy)*scale;
    data[index*5+3]=8;data[index*5+4]=8;
  };
  set(23,85,180);set(24,115,180);set(27,85,320);set(29,80,325);set(31,103,333);
  set(28,115,320);set(30,112,325);set(32,135,333);
  return {pose,crop,data};
}
test('foot landmarks recover heel and toe coordinates from the actual model output',()=>{
  const {pose,crop,data}=feetFixture(),feet=decodeFeet(data,.95,crop,pose);
  assert.equal(feet.length,2);assert.ok(Math.abs(feet[0].heel.x-80)<.001);assert.ok(Math.abs(feet[0].toe.y-333)<.001);
  assert.deepEqual(restoreFoot(128,128,crop),{x:crop.cx,y:crop.cy});
});
test('feet are omitted for absent poses, hidden ankles or mismatched people',()=>{
  const {pose,crop,data}=feetFixture();assert.equal(decodeFeet(data,.1,crop,pose).length,0);
  pose.joints[15].confidence=.05;assert.equal(decodeFeet(data,.95,crop,pose).length,1);
  data[23*5]=1000;assert.equal(decodeFeet(data,.95,crop,pose).length,0);
});
test('rejects heel and toe predictions floating far above an ankle',()=>{
  const {pose,crop,data}=feetFixture();
  pose.joints[13]={x:85,y:240,confidence:.9};
  for(const index of [29,31])data[index*5+1]-=60*256/crop.side;
  assert.equal(decodeFeet(data,.95,crop,pose).length,1);
});
