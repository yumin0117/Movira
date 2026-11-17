const fs=require('node:fs'),path=require('node:path'),C=require('../web/calibration.js');
const source=path.resolve(process.argv[2]),root=path.resolve(__dirname,'..'),dataRoot=path.join(process.env.USERPROFILE,'Desktop','Movira_test_Video');
const data=JSON.parse(fs.readFileSync(source,'utf8')),calibration=C.Calibration.import(data.calibration).export();
fs.writeFileSync(path.join(root,'web/calibration.json'),JSON.stringify(calibration));fs.writeFileSync(path.join(root,'SCALE-TRAINING-REPORT.json'),JSON.stringify(data.report,null,2));
fs.mkdirSync(path.join(dataRoot,'scale-training'),{recursive:true});fs.copyFileSync(source,path.join(dataRoot,'scale-training','training-package.json'));fs.writeFileSync(path.join(dataRoot,'scale-training','report.json'),JSON.stringify(data.report,null,2));
console.log(JSON.stringify({trainingExamples:data.report.trainingExamples,test:data.report.test}));
