const test=require('node:test'),assert=require('node:assert/strict'),http=require('node:http'),net=require('node:net'),{spawn}=require('node:child_process'),path=require('node:path');
const script=path.resolve(__dirname,'../scripts/serve.cjs');
const listen=server=>new Promise(resolve=>server.listen(0,'127.0.0.1',()=>resolve(server.address().port)));
const launch=port=>spawn(process.execPath,[script],{env:{...process.env,MOVIRA_PORT:String(port)},windowsHide:true,stdio:['ignore','pipe','pipe']});
test('Movira uses the requested port and serves its own page',async()=>{
  const reservation=net.createServer(),port=await listen(reservation);await new Promise(resolve=>reservation.close(resolve));
  const child=launch(port);let output='';child.stdout.on('data',chunk=>output+=chunk);child.stderr.on('data',chunk=>output+=chunk);
  try{await new Promise((resolve,reject)=>{const timer=setTimeout(()=>reject(Error(output||'Server did not start')),5000);child.once('exit',code=>{clearTimeout(timer);reject(Error('Server exited '+code+': '+output));});child.stdout.on('data',()=>{if(output.includes('Movira: http:')){clearTimeout(timer);resolve();}});});
    const response=await fetch(`http://127.0.0.1:${port}`);assert.equal(response.status,200);assert.match(await response.text(),/<title>Movira/);
  }finally{child.kill();}
});
test('a busy port reports an error without replacing or stopping the other app',async()=>{
  const other=http.createServer((request,response)=>response.end('other project')),port=await listen(other),child=launch(port);let error='';child.stderr.on('data',chunk=>error+=chunk);
  try{const code=await new Promise((resolve,reject)=>{const timer=setTimeout(()=>{child.kill();reject(Error('Busy-port check timed out'));},5000);child.once('exit',code=>{clearTimeout(timer);resolve(code);});});assert.equal(code,1);assert.match(error,/is busy/);assert.equal(await (await fetch(`http://127.0.0.1:${port}`)).text(),'other project');}
  finally{child.kill();other.closeAllConnections();await new Promise(resolve=>other.close(resolve));}
});
