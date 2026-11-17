/* A small trainable residual on detector scores. This training code is original. */
(function(scope){
'use strict';
const sigmoid=x=>1/(1+Math.exp(-Math.max(-35,Math.min(35,x)))),logit=p=>Math.log(Math.max(1e-7,p)/Math.max(1e-7,1-p));
function features(p,width,height){return [1,logit(p.raw)/4,Math.log(Math.max(.01,p.box.h/height))/4,Math.log(Math.max(.01,p.box.w/width))/4,Math.abs(Math.log(Math.max(.05,p.box.w/p.box.h)))/2,p.tile?1:0,p.nearEdge?1:0];}
class Calibration{
  constructor(){this.weights=new Float64Array(7);this.blend=1;this.threshold=.35;this.steps=0;this.meta={};}
  score(p,width,height){const x=features(p,width,height);let residual=0;for(let i=0;i<7;i++)residual+=x[i]*this.weights[i];return sigmoid(logit(p.raw)+this.blend*residual);}
  step(batch,rate=.04,regularization=.01){const gradient=new Float64Array(7);let loss=0;for(const sample of batch){let z=logit(sample.raw);for(let i=0;i<7;i++)z+=sample.x[i]*this.weights[i];const predicted=sigmoid(z),error=predicted-sample.target;loss-=sample.target*Math.log(Math.max(1e-9,predicted))+(1-sample.target)*Math.log(Math.max(1e-9,1-predicted));for(let i=0;i<7;i++)gradient[i]+=error*sample.x[i];}for(let i=0;i<7;i++)this.weights[i]-=rate*(gradient[i]/batch.length+regularization*this.weights[i]);this.steps++;return loss/batch.length;}
  export(){return {format:'movira-score-calibration-v1',weights:Array.from(this.weights),blend:this.blend,threshold:this.threshold,steps:this.steps,meta:this.meta};}
  static import(data){if(data?.format!=='movira-score-calibration-v1'||!Array.isArray(data.weights)||data.weights.length!==7||data.weights.some(n=>!Number.isFinite(n)||Math.abs(n)>100))throw Error('판별 보정 모델 형식이 올바르지 않습니다.');const c=new Calibration();c.weights.set(data.weights);c.blend=Number.isFinite(data.blend)?Math.max(0,Math.min(1,data.blend)):1;c.threshold=Number.isFinite(data.threshold)?Math.max(.05,Math.min(.99,data.threshold)):.35;c.steps=Number.isInteger(data.steps)?data.steps:0;c.meta=data.meta||{};return c;}
}
const api={Calibration,features,sigmoid,logit};scope.MoviraCalibration=api;if(typeof module!=='undefined')module.exports=api;
})(globalThis);
