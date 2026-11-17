const fs=require('node:fs'),path=require('node:path'),M=require('../research/prototype/engine.js');
const folder=path.resolve(__dirname,'../web/.dev');fs.mkdirSync(folder,{recursive:true});
for(const name of ['runtime-check','preprocess-check'])fs.copyFileSync(path.resolve(__dirname,'../tests/browser/'+name+'.html'),path.resolve(folder,'../.'+name+'.html'));
for(const [name,size,scale,offset] of [['person',3*416*416,255,0],['pose',3*256*192,4,-2]]){const rng=M.random(17),data=Float32Array.from({length:size},()=>rng()*scale+offset);fs.writeFileSync(path.join(folder,name+'.input.bin'),Buffer.from(data.buffer));}
