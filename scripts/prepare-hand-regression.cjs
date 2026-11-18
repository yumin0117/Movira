/* Preserve the prior Movira implementation for reproducible local A/B checks. */
const fs=require('node:fs'),path=require('node:path'),cp=require('node:child_process');
const root=path.resolve(__dirname,'..'),folder=path.join(root,'web/.dev');fs.mkdirSync(folder,{recursive:true});
for(const [file,name] of [['hands.js','Hands'],['vision-runtime.js','Runtime']]){let source=cp.execFileSync('git',['show','e28dfaf:web/'+file],{cwd:root,encoding:'utf8'});source=source.replaceAll('MoviraHands','MoviraHandsBefore').replaceAll('scope.MoviraRuntime=','scope.MoviraRuntimeBefore=');fs.writeFileSync(path.join(folder,'before-'+file),source);}
fs.copyFileSync(path.join(root,'tests/browser/hands-regression.html'),path.join(root,'web/.hands-regression-check.html'));
console.log('Prepared e28dfaf baseline and current regression page. Local user images remain in ignored web/.dev/regression.');
