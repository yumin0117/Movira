/* Supervised score adaptation, implemented independently for Movira. */
(function(scope){
'use strict';
function random(seed=4909){let state=seed>>>0;return ()=>{state^=state<<13;state^=state>>>17;state^=state<<5;return (state>>>0)/4294967296;};}
function sizeOf(box,height){const fraction=box.h/height;return fraction<.2?'small':fraction<.5?'medium':'large';}
function samples(records){const result=[];for(const r of records)for(const p of r.proposals){let best=0;for(const truth of r.people)best=Math.max(best,MoviraVision.iou(p.box,truth.box));if(best>=.5)result.push({x:MoviraCalibration.features(p,r.width,r.height),raw:p.raw,target:1});else if(best<.15&&!(r.ignoreBoxes||[]).some(b=>MoviraVision.iou(p.box,b)>.2))result.push({x:MoviraCalibration.features(p,r.width,r.height),raw:p.raw,target:0});}return result;}
function metrics(records,calibration,threshold){
  const bins={small:{tp:0,fn:0},medium:{tp:0,fn:0},large:{tp:0,fn:0}};let tp=0,fp=0,fn=0,ignored=0;
  for(const r of records){const predicted=MoviraVision.nms(r.proposals.map(p=>({...p,confidence:calibration.score(p,r.width,r.height)})).filter(p=>p.confidence>=threshold),.45),used=new Set();
    for(const p of predicted){let index=-1,best=0;for(let i=0;i<r.people.length;i++){const score=MoviraVision.iou(p.box,r.people[i].box);if(!used.has(i)&&score>best){best=score;index=i;}}if(best>=.5){used.add(index);tp++;bins[sizeOf(r.people[index].box,r.height)].tp++;}else if((r.ignoreBoxes||[]).some(b=>MoviraVision.iou(p.box,b)>.2))ignored++;else fp++;}
    for(let i=0;i<r.people.length;i++)if(!used.has(i)){fn++;bins[sizeOf(r.people[i].box,r.height)].fn++;}
  }
  const precision=tp+fp?tp/(tp+fp):0,recall=tp+fn?tp/(tp+fn):0;for(const b of Object.values(bins))b.recall=b.tp+b.fn?b.tp/(b.tp+b.fn):null;
  return {images:records.length,tp,fp,fn,ignored,precision,recall,f1:precision+recall?2*precision*recall/(precision+recall):0,bySize:bins,sizeDefinition:'person bbox height / image height: small<0.2; medium<0.5; large>=0.5'};
}
async function train(records,options={}){
  const epochCount=options.epochs||160,rng=random(),training=records.filter(r=>r.split==='train'),validation=records.filter(r=>r.split==='validation'),testing=records.filter(r=>r.split==='test'),data=samples(training),positives=data.filter(s=>s.target).length;
  if(!positives||positives===data.length)throw Error('학습에는 사람 정답과 배경 오검출 예제가 모두 필요합니다.');if(!validation.length)throw Error('서로 다른 검증 사진이 필요합니다.');
  const candidate=new MoviraCalibration.Calibration(),baseline=new MoviraCalibration.Calibration(),thresholds=[.12,.16,.2,.25,.3,.35,.4,.45,.5,.6,.7,.8];let lastLoss=0;
  for(let epoch=0;epoch<epochCount;epoch++){
    if(options.cancel?.())break;for(let i=data.length-1;i>0;i--){const j=Math.floor(rng()*(i+1));[data[i],data[j]]=[data[j],data[i]];}
    let sum=0,count=0;for(let i=0;i<data.length;i+=64){const batch=data.slice(i,i+64);sum+=candidate.step(batch,.04/(1+epoch*.015),.008);count++;}lastLoss=sum/count;
    options.progress?.({phase:'학습',epoch:epoch+1,epochs:epochCount,loss:lastLoss});await new Promise(resolve=>setTimeout(resolve,0));
  }
  let best=null,baseBest=null;
  for(const blend of [0,.25,.5,.75,1]){candidate.blend=blend;for(const threshold of thresholds){const value=metrics(validation,candidate,threshold),entry={blend,threshold,...value};if(blend===0&&(!baseBest||entry.f1>baseBest.f1))baseBest=entry;if(!best||entry.f1>best.f1+1e-9||Math.abs(entry.f1-best.f1)<1e-9&&entry.precision>best.precision)best=entry;}}
  candidate.blend=best.blend;candidate.threshold=best.threshold;baseline.threshold=baseBest.threshold;
  const report={createdUtc:new Date().toISOString(),implementation:'Movira original residual logistic learner; public YOLOX-Tiny and RTMPose-s weights remain frozen',trainImages:training.length,validationImages:validation.length,testImages:testing.length,trainingExamples:data.length,positiveExamples:positives,negativeExamples:data.length-positives,epochs:epochCount,loss:lastLoss,selected:{blend:best.blend,threshold:best.threshold},baselineValidation:baseBest,validation:best,baselineTest:metrics(testing,baseline,baseBest.threshold),test:metrics(testing,candidate,best.threshold),trainingImageNames:training.map(r=>r.name),validationImageNames:validation.map(r=>r.name),testImageNames:testing.map(r=>r.name)};
  candidate.meta={createdUtc:report.createdUtc,trainingExamples:data.length,trainImages:training.length,validation:best,test:report.test,backboneFineTuned:false,training:'Residual person-score calibration only. Public detector/pose backbone weights are not retrained.'};return {calibration:candidate.export(),report};
}
scope.MoviraTrainer={random,sizeOf,samples,metrics,train};if(typeof module!=='undefined')module.exports=scope.MoviraTrainer;
})(globalThis);
