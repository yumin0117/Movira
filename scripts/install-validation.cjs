const fs=require('node:fs'),path=require('node:path'),source=path.resolve(process.argv[2]),root=path.resolve(__dirname,'..'),data=JSON.parse(fs.readFileSync(source,'utf8'));
data.dataset='COCO val2017, CC BY 2.0 image metadata; excludes the earlier 257 pose-example images; not used to fit or select the new calibration';
fs.writeFileSync(path.join(root,'SCALE-GPU-VALIDATION.json'),JSON.stringify(data,null,2));
const file=path.join(root,'web/calibration.json'),model=JSON.parse(fs.readFileSync(file,'utf8'));model.meta.independentTest=data.full;fs.writeFileSync(file,JSON.stringify(model));
fs.copyFileSync(source,path.join(process.env.USERPROFILE,'Desktop','Movira_test_Video','scale-training','independent-gpu-validation.json'));
console.log(JSON.stringify(data.full));
