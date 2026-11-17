/* Offline development tool. Uses only Node built-ins and Movira's own engine. */
'use strict';
const fs=require('node:fs'),path=require('node:path'),M=require('../web/engine.js');
const dataRoot=path.resolve(process.argv[2]||path.join(process.env.USERPROFILE,'Desktop','Movira_test_Video'));
const epochs=Math.max(1,Math.min(500,Number(process.argv[3])||100));
const dataset=JSON.parse(fs.readFileSync(path.join(dataRoot,'movira-dataset.json'),'utf8').replace(/^\uFEFF/,''));
const ignored=new Map(),originalAnnotations=path.join(dataRoot,'person_keypoints_val2017.json');
if(fs.existsSync(originalAnnotations))for(const a of JSON.parse(fs.readFileSync(originalAnnotations,'utf8')).annotations){if(!ignored.has(a.image_id))ignored.set(a.image_id,[]);ignored.get(a.image_id).push({x:a.bbox[0],y:a.bbox[1],w:a.bbox[2],h:a.bbox[3]});}
const valEnd=Math.min(240,dataset.records.length-1),split=Math.floor(valEnd*.8),trainRecords=dataset.records.slice(0,split),valRecords=dataset.records.slice(split,valEnd),testRecords=dataset.records.slice(valEnd);
function prepare(records){
  const samples=[];
  for(const r of records){const data=fs.readFileSync(path.join(dataRoot,'pixels',r.name+'.bgra'));
    if(data.length!==r.width*r.height*4)throw Error('Pixel dimensions mismatch: '+r.name);
    for(const p of r.people){const box=M.crop(p.box);for(const mirrored of [false,true])samples.push({input:M.features(data,r.width,r.height,box,mirrored,true),positive:true,points:M.pointsTarget(p,box,mirrored),box:M.boxTarget(p,box,mirrored)});}
    const ignore=r.ignoreBoxes||ignored.get(Number(r.name.split('.')[0]))||r.people.map(p=>p.box);let count=0;for(const box of M.windows(r.width,r.height))if(count<Math.max(4,r.people.length*3)&&ignore.every(b=>M.overlap(box,b)<.01)){samples.push({input:M.features(data,r.width,r.height,box,false,true),positive:false,points:[]});count++;}
  }return samples;
}
const samples=prepare(trainRecords),validation=prepare(valRecords),testSamples=prepare(testRecords),model=new M.ConvModel(64),rng=M.random(3261);
if(!samples.some(s=>s.positive)||!samples.some(s=>!s.positive)||!validation.length)throw Error('Need labeled train/validation images and background examples.');
for(const r of trainRecords)for(const p of r.people)p.points.forEach((point,k)=>{if(point&&p.known[k])model.coverage[k]++;});
const before=M.evaluate(model,validation),start=Date.now();let best=null,bestError=Infinity,bestEpoch=0;
for(let epoch=0;epoch<epochs;epoch++){
  for(let i=samples.length-1;i>0;i--){const j=Math.floor(rng()*(i+1));[samples[i],samples[j]]=[samples[j],samples[i]];}
  let loss=0;for(const sample of samples)loss+=model.train(sample.positive&&rng()<.4?M.occlude(sample,rng):sample,.0015/(1+epoch*.025));
  if(epoch===9){let mined=0;for(const r of trainRecords){const data=fs.readFileSync(path.join(dataRoot,'pixels',r.name+'.bgra')),ignore=r.ignoreBoxes||ignored.get(Number(r.name.split('.')[0]))||r.people.map(p=>p.box),candidates=[];for(const box of M.windows(r.width,r.height))if(ignore.every(b=>M.overlap(box,b)<.02)){const input=M.features(data,r.width,r.height,box,false,true),score=model.forward(input).out[0];if(score>.4)candidates.push({input,score,positive:false,points:[]});}candidates.sort((a,b)=>b.score-a.score);samples.push(...candidates.slice(0,3));mined+=Math.min(3,candidates.length);}console.log(JSON.stringify({hardBackgroundExamples:mined}));}
  const metrics=M.evaluate(model,validation),error=metrics.normalizedJointError;
  const selection=error===null?Infinity:error+(1-metrics.classificationAccuracy)*.3;
  if(selection<bestError){bestError=selection;best=model.export();bestEpoch=epoch+1;}
  if((epoch+1)%10===0||epoch===0)console.log(JSON.stringify({epoch:epoch+1,loss:loss/samples.length,validation:metrics}));
}
const trained=M.Model.import(best),after=M.evaluate(trained,validation);
const testMetrics=M.evaluate(trained,testSamples);
const erasedRandom=M.random(18007),occludedTest=M.evaluate(trained,testSamples.map(s=>s.positive?M.occlude(s,erasedRandom):s));
let tp=0,fp=0,fn=0;
for(const r of testRecords){const bgra=fs.readFileSync(path.join(dataRoot,'pixels',r.name+'.bgra')),rgba=Uint8Array.from(bgra);for(let i=0;i<rgba.length;i+=4){rgba[i]=bgra[i+2];rgba[i+2]=bgra[i];}const predicted=M.detect(trained,rgba,r.width,r.height,.75),used=new Set();for(const p of predicted){let best=-1,score=0;for(let i=0;i<r.people.length;i++){const iou=M.overlap(p.box,r.people[i].box);if(!used.has(i)&&iou>score){score=iou;best=i;}}if(score>=.5){tp++;used.add(best);}else fp++;}fn+=r.people.length-used.size;}
const report={implementation:'Movira original two-layer CNN, learned boxes and occlusion augmentation; random initialization; no upstream code/weights',createdUtc:new Date().toISOString(),trainImages:trainRecords.map(r=>r.name),validationImages:valRecords.map(r=>r.name),testImages:testRecords.map(r=>r.name),trainSamples:samples.length,validationSamples:validation.length,testSamples:testSamples.length,epochs,bestEpoch,selectionCriterion:'validation normalized joint error + 0.3 * (1 - classification accuracy)',seconds:(Date.now()-start)/1000,before,after,test:testMetrics,occludedTest,fullImageTest:{threshold:.75,matchingIoU:.5,tp,fp,fn,precision:tp+fp?tp/(tp+fp):0,recall:tp+fn?tp/(tp+fn):0},coverage:trained.coverage,scope:'Joint error uses ground-truth person crops, including labeled occluded points, not full-image joint accuracy. Validation selects epoch; new images after index 240 reserved as final test. Erased test is synthetic input masking, not proof of real-world occlusion robustness. No feet labels in COCO. Experimental, not production-ready.'};
trained.meta={created:report.createdUtc,trainImages:trainRecords.length,validationImages:valRecords.length,validation:after,experimental:true,report:'TRAINING-REPORT.json',selectedEpoch:bestEpoch};
fs.writeFileSync(path.join(__dirname,'../web/own-model.json'),JSON.stringify(trained.export()));
fs.writeFileSync(path.join(__dirname,'../TRAINING-REPORT.json'),JSON.stringify(report,null,2));
fs.writeFileSync(path.join(dataRoot,'training-report.json'),JSON.stringify(report,null,2));
console.log(JSON.stringify({saved:'web/own-model.json',report:'TRAINING-REPORT.json',...after}));
