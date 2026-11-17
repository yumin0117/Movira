/* Movira's own experimental learner. No imported implementation or weights. */
(function(scope){
  'use strict';
  const joints=['얼굴','목','왼쪽 어깨','오른쪽 어깨','왼쪽 팔꿈치','오른쪽 팔꿈치','왼쪽 손목','오른쪽 손목','골반 중심','왼쪽 고관절','오른쪽 고관절','왼쪽 무릎','오른쪽 무릎','왼쪽 발목','오른쪽 발목','왼쪽 뒤꿈치','오른쪽 뒤꿈치','왼쪽 발끝','오른쪽 발끝'];
  const flip=[0,1,3,2,5,4,7,6,8,10,9,12,11,14,13,16,15,18,17];
  const bones=[[0,1],[1,2],[1,3],[2,4],[4,6],[3,5],[5,7],[1,8],[8,9],[8,10],[9,11],[11,13],[10,12],[12,14],[13,15],[15,17],[17,13],[14,16],[16,18],[18,14]];
  const W=24,H=32,I=W*H,P=5,O=P+joints.length*3;
  function random(seed=7061){let state=seed>>>0;return ()=>{state^=state<<13;state^=state>>>17;state^=state<<5;return (state>>>0)/4294967296;};}
  function sigmoid(x){return 1/(1+Math.exp(-Math.max(-30,Math.min(30,x))));}
  function features(pixels,width,height,box,mirrored=false,bgra=false){
    const out=new Float32Array(I);let sum=0,squared=0;
    for(let y=0;y<H;y++)for(let x=0;x<W;x++){
      const col=mirrored?W-1-x:x;
      const ix=Math.floor(box.x+(col+.5)*box.w/W),iy=Math.floor(box.y+(y+.5)*box.h/H);
      let value=0;
      if(ix>=0&&ix<width&&iy>=0&&iy<height){const offset=(iy*width+ix)*4;value=(pixels[offset+(bgra?2:0)]*.299+pixels[offset+1]*.587+pixels[offset+(bgra?0:2)]*.114)/255;}
      out[y*W+x]=value;sum+=value;squared+=value*value;
    }
    const mean=sum/I,scale=Math.max(.08,Math.sqrt(Math.max(0,squared/I-mean*mean)));
    for(let i=0;i<I;i++)out[i]=(out[i]-mean)/scale;
    return out;
  }
  function crop(box,padding=1.15){
    let w=box.w*padding,h=box.h*padding;if(w/h>W/H)h=w*H/W;else w=h*W/H;
    return {x:box.x+box.w/2-w/2,y:box.y+box.h/2-h/2,w,h};
  }
  function overlap(a,b){const size=Math.max(0,Math.min(a.x+a.w,b.x+b.w)-Math.max(a.x,b.x))*Math.max(0,Math.min(a.y+a.h,b.y+b.h)-Math.max(a.y,b.y));return size/Math.max(1,a.w*a.h+b.w*b.h-size);}
  class Model{
    constructor(hidden=64,seed=7061){
      if(!Number.isInteger(hidden)||hidden<4||hidden>128)throw Error('지원하지 않는 모델 크기입니다.');
      this.hidden=hidden;this.w1=new Float32Array(I*hidden);this.b1=new Float32Array(hidden);this.w2=new Float32Array(hidden*O);this.b2=new Float32Array(O);this.coverage=Array(joints.length).fill(0);this.steps=0;this.meta={};
      const rng=random(seed);for(let i=0;i<this.w1.length;i++)this.w1[i]=(rng()*2-1)*Math.sqrt(3/I);for(let i=0;i<this.w2.length;i++)this.w2[i]=(rng()*2-1)*Math.sqrt(3/hidden);
      this.velocity=[this.w1,this.b1,this.w2,this.b2].map(a=>new Float32Array(a.length));
    }
    encode(input){return {encoded:input};}
    forward(input){
      const state=this.encode(input);input=state.encoded;
      const hidden=new Float32Array(this.hidden),out=new Float32Array(O);
      for(let j=0;j<this.hidden;j++){let z=this.b1[j],start=j*I;for(let i=0;i<I;i++)z+=this.w1[start+i]*input[i];hidden[j]=Math.tanh(z);}
      for(let k=0;k<O;k++){let z=this.b2[k],start=k*this.hidden;for(let j=0;j<this.hidden;j++)z+=this.w2[start+j]*hidden[j];out[k]=sigmoid(z);}
      return {...state,hidden,out};
    }
    train(sample,rate=.015){
      const state=this.forward(sample.input),{hidden,out,encoded}=state,dy=new Float32Array(O),dh=new Float32Array(this.hidden);
      const positive=sample.positive?1:0;dy[0]=out[0]-positive;
      let loss=-positive*Math.log(Math.max(1e-7,out[0]))-(1-positive)*Math.log(Math.max(1e-7,1-out[0]));
      if(positive&&sample.box)for(let k=0;k<4;k++){const error=out[k+1]-sample.box[k];loss+=2*error*error;dy[k+1]=4*error*out[k+1]*(1-out[k+1]);}
      const count=Math.max(1,sample.points?.filter(p=>p!==undefined).length||0);
      if(positive)for(let k=0;k<joints.length;k++){
        const p=sample.points[k];if(p===undefined)continue;
        const base=P+k*3,v=p?(p[2]===0?0:1):0;dy[base+2]=.5/count*(out[base+2]-v);
        loss+=.5/count*(-v*Math.log(Math.max(1e-7,out[base+2]))-(1-v)*Math.log(Math.max(1e-7,1-out[base+2])));
        if(p)for(let c=0;c<2;c++){const weight=v?1:.35,error=out[base+c]-p[c];loss+=weight*4/count*error*error;dy[base+c]=weight*8/count*error*out[base+c]*(1-out[base+c]);}
      }
      for(let k=0;k<O;k++)for(let j=0;j<this.hidden;j++)dh[j]+=dy[k]*this.w2[k*this.hidden+j];
      let featureGradient;
      if(this.backpropEncoding){featureGradient=new Float32Array(I);for(let j=0;j<this.hidden;j++){const d=dh[j]*(1-hidden[j]*hidden[j]);for(let i=0;i<I;i++)featureGradient[i]+=d*this.w1[j*I+i];}}
      const update=(array,velocity,index,gradient)=>{const g=Math.max(-3,Math.min(3,gradient));velocity[index]=.85*velocity[index]+g;array[index]-=rate*velocity[index];};
      for(let k=0;k<O;k++){for(let j=0;j<this.hidden;j++)update(this.w2,this.velocity[2],k*this.hidden+j,dy[k]*hidden[j]);update(this.b2,this.velocity[3],k,dy[k]);}
      for(let j=0;j<this.hidden;j++){const d=dh[j]*(1-hidden[j]*hidden[j]);for(let i=0;i<I;i++)update(this.w1,this.velocity[0],j*I+i,d*encoded[i]);update(this.b1,this.velocity[1],j,d);}
      if(featureGradient)this.backpropEncoding(state,featureGradient,rate);
      this.steps++;return loss;
    }
    export(){return {format:'movira-own-mlp-v2',width:W,height:H,joints:[...joints],hidden:this.hidden,steps:this.steps,coverage:this.coverage,meta:this.meta,w1:Array.from(this.w1),b1:Array.from(this.b1),w2:Array.from(this.w2),b2:Array.from(this.b2)};}
    static import(data){
      if(data?.format==='movira-own-conv-v2')return ConvModel.import(data);
      if(data?.format!=='movira-own-mlp-v2'||data.width!==W||data.height!==H||JSON.stringify(data.joints)!==JSON.stringify(joints))throw Error('현재 Movira 모델 형식이 아닙니다. 이 버전에서 다시 학습한 모델을 사용하세요.');
      const model=new Model(data.hidden);
      for(const key of ['w1','b1','w2','b2']){if(!Array.isArray(data[key])||data[key].length!==model[key].length||data[key].some(n=>!Number.isFinite(n)||Math.abs(n)>1000))throw Error('모델 가중치가 손상되었습니다.');model[key].set(data[key]);}
      if(!Array.isArray(data.coverage)||data.coverage.length!==joints.length||data.coverage.some(n=>!Number.isInteger(n)||n<0))throw Error('학습 범위 정보가 올바르지 않습니다.');
      model.coverage=[...data.coverage];model.steps=Number.isInteger(data.steps)&&data.steps>=0?data.steps:0;model.meta=data.meta&&typeof data.meta==='object'?data.meta:{};return model;
    }
  }
  // Two spatial layers, written here rather than imported from a neural library.
  // Channel-first, 3x3 kernel, stride 2, one pixel of zero padding, ReLU.
  function convolution(input,inW,inH,inC,outC,weights,bias){
    const outW=Math.ceil(inW/2),outH=Math.ceil(inH/2),output=new Float32Array(outW*outH*outC);
    for(let c=0;c<outC;c++)for(let y=0;y<outH;y++)for(let x=0;x<outW;x++){
      let value=bias[c];
      for(let ic=0;ic<inC;ic++)for(let ky=0;ky<3;ky++)for(let kx=0;kx<3;kx++){
        const ix=x*2+kx-1,iy=y*2+ky-1;if(ix>=0&&ix<inW&&iy>=0&&iy<inH)value+=input[(ic*inH+iy)*inW+ix]*weights[((c*inC+ic)*3+ky)*3+kx];
      }
      output[(c*outH+y)*outW+x]=Math.max(0,value);
    }
    return output;
  }
  function convolutionGradient(input,output,gradient,inW,inH,inC,outC,weights){
    const outW=Math.ceil(inW/2),outH=Math.ceil(inH/2),dx=new Float32Array(input.length),dw=new Float32Array(weights.length),db=new Float32Array(outC);
    for(let c=0;c<outC;c++)for(let y=0;y<outH;y++)for(let x=0;x<outW;x++){
      const at=(c*outH+y)*outW+x,g=output[at]>0?gradient[at]:0;if(!g)continue;db[c]+=g;
      for(let ic=0;ic<inC;ic++)for(let ky=0;ky<3;ky++)for(let kx=0;kx<3;kx++){
        const ix=x*2+kx-1,iy=y*2+ky-1;if(ix<0||ix>=inW||iy<0||iy>=inH)continue;
        const wi=((c*inC+ic)*3+ky)*3+kx,pi=(ic*inH+iy)*inW+ix;dw[wi]+=g*input[pi];dx[pi]+=g*weights[wi];
      }
    }
    return {dx,dw,db};
  }
  class ConvModel extends Model{
    constructor(hidden=64,seed=7061){
      super(hidden,seed);this.c1=new Float32Array(8*9);this.cb1=new Float32Array(8);this.c2=new Float32Array(16*8*9);this.cb2=new Float32Array(16);
      const rng=random(seed+17);for(let i=0;i<this.c1.length;i++)this.c1[i]=(rng()*2-1)*Math.sqrt(6/9);for(let i=0;i<this.c2.length;i++)this.c2[i]=(rng()*2-1)*Math.sqrt(6/72);
      this.convVelocity=[this.c1,this.cb1,this.c2,this.cb2].map(a=>new Float32Array(a.length));
    }
    encode(input){const first=convolution(input,W,H,1,8,this.c1,this.cb1),encoded=convolution(first,W/2,H/2,8,16,this.c2,this.cb2);return {raw:input,first,encoded};}
    backpropEncoding(state,gradient,rate){
      const second=convolutionGradient(state.first,state.encoded,gradient,W/2,H/2,8,16,this.c2),first=convolutionGradient(state.raw,state.first,second.dx,W,H,1,8,this.c1);
      for(const [index,[array,g]] of [[this.c1,first.dw],[this.cb1,first.db],[this.c2,second.dw],[this.cb2,second.db]].entries())for(let i=0;i<array.length;i++){this.convVelocity[index][i]=.85*this.convVelocity[index][i]+Math.max(-3,Math.min(3,g[i]));array[i]-=rate*this.convVelocity[index][i];}
    }
    export(){return {...super.export(),format:'movira-own-conv-v2',c1:Array.from(this.c1),cb1:Array.from(this.cb1),c2:Array.from(this.c2),cb2:Array.from(this.cb2)};}
    static import(data){
      const dense=Model.import({...data,format:'movira-own-mlp-v2'}),model=new ConvModel(data.hidden);
      for(const key of ['w1','b1','w2','b2'])model[key].set(dense[key]);model.coverage=dense.coverage;model.steps=dense.steps;model.meta=dense.meta;
      for(const key of ['c1','cb1','c2','cb2']){if(!Array.isArray(data[key])||data[key].length!==model[key].length||data[key].some(n=>!Number.isFinite(n)||Math.abs(n)>1000))throw Error('합성곱 가중치가 손상되었습니다.');model[key].set(data[key]);}
      return model;
    }
  }
  function pointsTarget(person,box,mirrored=false){
    return joints.map((_,k)=>{const index=mirrored?flip[k]:k,point=person.points[index];if(person.known&&person.known[index]===false||point===undefined)return undefined;if(point===null)return null;const x=(point.x-box.x)/box.w,y=(point.y-box.y)/box.h;return [mirrored?1-x:x,y,point.occluded?0:1];});
  }
  function occlude(sample,rng=random(8)){
    if(!sample.positive)return sample;
    const w=.2+rng()*.2,h=.15+rng()*.25,x=rng()*(1-w),y=rng()*(1-h),input=Float32Array.from(sample.input);
    for(let row=0;row<H;row++)for(let col=0;col<W;col++)if((col+.5)/W>=x&&(col+.5)/W<x+w&&(row+.5)/H>=y&&(row+.5)/H<y+h)input[row*W+col]=0;
    const points=sample.points.map(p=>p?([p[0],p[1],p[0]>=x&&p[0]<=x+w&&p[1]>=y&&p[1]<=y+h?0:(p[2]===0?0:1)]):p);
    return {...sample,input,points,erased:{x,y,w,h}};
  }
  function boxTarget(person,box,mirrored=false){const p=person.box,left=(p.x-box.x)/box.w,right=(p.x+p.w-box.x)/box.w;return [mirrored?1-right:left,(p.y-box.y)/box.h,mirrored?1-left:right,(p.y+p.h-box.y)/box.h];}
  function windows(width,height){
    const result=[];for(const fraction of [1,.72,.5,.35]){
      const h=Math.max(24,height*fraction),w=h*W/H,stepX=w*.45,stepY=h*.4;
      const xs=[0],ys=[0];for(let x=stepX;x<Math.max(0,width-w);x+=stepX)xs.push(x);if(width>w)xs.push(width-w);
      for(let y=stepY;y<Math.max(0,height-h);y+=stepY)ys.push(y);if(height>h)ys.push(height-h);
      for(const x of xs)for(const y of ys)result.push({x,y,w,h});
    }
    return result;
  }
  function samePerson(a,b){
    if(overlap(a.box,b.box)<=.3)return false;
    let distance=0,count=0;for(let k=0;k<joints.length;k++){const p=a.points[k],q=b.points[k];if(p.confidence>.2&&q.confidence>.2){distance+=Math.hypot(p.x-q.x,p.y-q.y);count++;}}
    const scale=Math.max(1,Math.hypot(a.box.w,a.box.h),Math.hypot(b.box.w,b.box.h));
    return count>=4&&distance/count<scale*.07;
  }
  function detect(model,pixels,width,height,threshold=.75){
    if(!model||model.steps===0)return [];
    const found=[];
    for(const box of windows(width,height)){
      const {out}=model.forward(features(pixels,width,height,box));if(out[0]<threshold)continue;
      const points=joints.map((_,k)=>({x:box.x+out[P+k*3]*box.w,y:box.y+out[P+1+k*3]*box.h,confidence:model.coverage[k]>=3?out[0]:0,visibility:out[P+2+k*3],occluded:out[P+2+k*3]<.5}));
      const refined={x:Math.max(0,box.x+out[1]*box.w),y:Math.max(0,box.y+out[2]*box.h)};
      refined.w=Math.min(width,box.x+out[3]*box.w)-refined.x;refined.h=Math.min(height,box.y+out[4]*box.h)-refined.y;
      if(refined.w<5||refined.h<10)continue;
      found.push({box:refined,confidence:out[0],points});
    }
    found.sort((a,b)=>b.confidence-a.confidence);const kept=[];
    // Two people can overlap. Only suppress a candidate whose joint geometry
    // also agrees, rather than deleting a person solely because boxes overlap.
    for(const person of found)if(!kept.some(p=>samePerson(p,person)))kept.push(person);
    return kept;
  }
  function evaluate(model,samples){
    let count=0,error=0,correct=0,positives=0,retained=0,hidden=0,hiddenError=0,visible=0,visibleError=0;
    for(const sample of samples){const {out}=model.forward(sample.input);correct+=Number((out[0]>=.5)===sample.positive);if(!sample.positive)continue;positives++;if(out[0]>=.75)retained++;for(let k=0;k<joints.length;k++){const p=sample.points[k];if(p){const e=Math.hypot(out[P+k*3]-p[0],out[P+1+k*3]-p[1]);error+=e;count++;if(p[2]===0){hidden++;hiddenError+=e;}else{visible++;visibleError+=e;}}}}
    return {samples:samples.length,classificationAccuracy:samples.length?correct/samples.length:null,labeledJoints:count,normalizedJointError:count?error/count:null,personCrops:positives,personRecallAt075:positives?retained/positives:null,visibleJoints:visible,visibleJointError:visible?visibleError/visible:null,occludedJoints:hidden,occludedJointError:hidden?hiddenError/hidden:null};
  }
  const api={joints,bones,flip,W,H,I,O,random,features,crop,overlap,Model,ConvModel,pointsTarget,boxTarget,occlude,windows,samePerson,detect,evaluate};scope.Movira=api;if(typeof module!=='undefined')module.exports=api;
})(globalThis);
