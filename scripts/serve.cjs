/* Local HTTP preview for the browser Worker and WebGPU model loader. */
const http=require('node:http'),fs=require('node:fs'),path=require('node:path');
const port=Number(process.env.MOVIRA_PORT||5178);
if(!Number.isInteger(port)||port<1||port>65535)throw Error('MOVIRA_PORT must be between 1 and 65535');
const root=path.resolve(__dirname,'../web'),types={'.html':'text/html; charset=utf-8','.css':'text/css; charset=utf-8','.js':'text/javascript; charset=utf-8','.json':'application/json'};
http.createServer((request,response)=>{
  const url=new URL(request.url,'http://127.0.0.1');let name;
  try{name=path.resolve(root,'.'+decodeURIComponent(url.pathname==='/'?'/index.html':url.pathname));}catch{response.writeHead(400).end();return;}
  if(!name.startsWith(root+path.sep)){response.writeHead(403).end();return;}
  fs.readFile(name,(error,data)=>{if(error){response.writeHead(404).end();return;}response.writeHead(200,{'Content-Type':types[path.extname(name)]||'application/octet-stream','Cache-Control':'no-cache'});response.end(data);});
}).on('error',error=>{console.error(error.code==='EADDRINUSE'?`Movira port ${port} is busy. Choose another MOVIRA_PORT; no other server was stopped.`:error.message);process.exitCode=1;}).listen(port,'127.0.0.1',()=>console.log(`Movira: http://127.0.0.1:${port}`));
