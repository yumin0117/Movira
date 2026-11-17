/* Synthetic occlusion evaluation, no external image or inference libraries. */
const fs=require('node:fs'),path=require('node:path'),M=require('../research/prototype/engine.js');
const root=path.resolve(process.argv[2]||path.join(process.env.USERPROFILE,'Desktop','Movira_test_Video'));
const dataset=JSON.parse(fs.readFileSync(path.join(root,'movira-dataset.json'),'utf8').replace(/^\uFEFF/,'')),reportPath=path.join(__dirname,'../research/prototype/training-report.json'),report=JSON.parse(fs.readFileSync(reportPath,'utf8')),model=M.Model.import(JSON.parse(fs.readFileSync(path.join(__dirname,'../research/prototype/model.json'),'utf8')));
const threshold=model.meta.recommendedThreshold||.75;
function rgba(r){const bgra=fs.readFileSync(path.join(root,'pixels',r.name+'.bgra')),out=Uint8Array.from(bgra);for(let i=0;i<out.length;i+=4){out[i]=bgra[i+2];out[i+2]=bgra[i];}return out;}
function matches(predictions,people){const used=new Set();let fp=0;for(const p of predictions){let best=-1,score=0;for(let i=0;i<people.length;i++){const iou=M.overlap(p.box,people[i].box);if(!used.has(i)&&iou>score){score=iou;best=i;}}if(score>=.5)used.add(best);else fp++;}return {used,fp};}
const cases={clean:{tp:0,fp:0,fn:0,retainedFromClean:0},objectPatch:{tp:0,fp:0,fn:0,retainedFromClean:0},personTexturePatch:{tp:0,fp:0,fn:0,retainedFromClean:0}};
let totalClean=0;
for(const name of report.testImages){
  const r=dataset.records.find(r=>r.name===name),original=rgba(r),textureRecord=dataset.records.find(s=>s.name!==name&&s.people.length),texture=rgba(textureRecord),source=textureRecord.people[0].box;
  const clean=matches(M.detect(model,original,r.width,r.height,threshold),r.people);totalClean+=clean.used.size;
  for(const [condition,result] of Object.entries(cases)){
    let found=clean;
    if(condition!=='clean'){
      const masked=Uint8Array.from(original);
      for(const person of r.people){const b=person.box,left=Math.max(0,Math.floor(b.x+b.w*.35)),top=Math.max(0,Math.floor(b.y+b.h*.33)),w=Math.max(1,Math.floor(b.w*.3)),h=Math.max(1,Math.floor(b.h*.3));
        for(let y=0;y<h;y++)for(let x=0;x<w;x++){
          if(left+x>=r.width||top+y>=r.height)continue;const at=((top+y)*r.width+left+x)*4;
          if(condition==='objectPatch'){masked[at]=105;masked[at+1]=112;masked[at+2]=116;}
          else{const sx=Math.min(textureRecord.width-1,Math.max(0,Math.floor(source.x+x/w*source.w))),sy=Math.min(textureRecord.height-1,Math.max(0,Math.floor(source.y+y/h*source.h))),from=(sy*textureRecord.width+sx)*4;for(let c=0;c<3;c++)masked[at+c]=texture[from+c];}
        }
      }
      found=matches(M.detect(model,masked,r.width,r.height,threshold),r.people);
    }
    result.tp+=found.used.size;result.fp+=found.fp;result.fn+=r.people.length-found.used.size;result.retainedFromClean+=[...clean.used].filter(i=>found.used.has(i)).length;
  }
}
for(const value of Object.values(cases)){value.precision=value.tp+value.fp?value.tp/(value.tp+value.fp):0;value.recall=value.tp+value.fn?value.tp/(value.tp+value.fn):0;value.retentionOfCleanTruePeople=totalClean?value.retainedFromClean/totalClean:null;}
report.syntheticFullImageOcclusion={images:report.testImages.length,threshold,matchingIoU:.5,mask:'30% width x 30% height of each annotated person bbox, torso region (9% area). Object color or texture sampled from a different person. Synthetic stress test, not real-world robustness proof.',cases};
fs.writeFileSync(reportPath,JSON.stringify(report,null,2));fs.writeFileSync(path.join(root,'training-report.json'),JSON.stringify(report,null,2));
const exported=model.export();exported.meta.fullImageTest=report.fullImageTest;exported.meta.syntheticOcclusion=report.syntheticFullImageOcclusion;exported.meta.test=report.test;exported.meta.dataSources=['https://cocodataset.org/#termsofuse','COCO Consortium annotations CC BY 4.0; selected image metadata CC BY 2.0'];
fs.writeFileSync(path.join(__dirname,'../research/prototype/model.json'),JSON.stringify(exported));console.log(JSON.stringify(report.syntheticFullImageOcclusion,null,2));
