import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
const root=path.resolve(fileURLToPath(new URL('../web/',import.meta.url)));
const types={'.html':'text/html; charset=utf-8','.js':'text/javascript; charset=utf-8','.mjs':'text/javascript; charset=utf-8','.css':'text/css; charset=utf-8','.svg':'image/svg+xml','.wasm':'application/wasm','.onnx':'application/octet-stream','.md':'text/plain; charset=utf-8'};
const server=http.createServer((request,response)=>{
  if(request.method!=='GET'&&request.method!=='HEAD'){response.writeHead(405);response.end();return;}
  let filename;
  try {const url=new URL(request.url,'http://localhost');filename=path.resolve(root,'.'+decodeURIComponent(url.pathname==='/'?'/index.html':url.pathname));}
  catch {response.writeHead(400);response.end();return;}
  if(!filename.startsWith(root+path.sep)&&filename!==path.join(root,'index.html')){response.writeHead(403);response.end();return;}
  fs.stat(filename,(error,stat)=>{
    if(error||!stat.isFile()){response.writeHead(404);response.end('Not found');return;}
    response.writeHead(200,{'Content-Type':types[path.extname(filename)]||'application/octet-stream','Content-Length':stat.size,'Cache-Control':'no-cache','X-Content-Type-Options':'nosniff'});
    if(request.method==='HEAD')response.end();else fs.createReadStream(filename).pipe(response);
  });
});
server.on('error',error=>{console.error(error.message);process.exitCode=1;});
server.listen(3000,'127.0.0.1',()=>console.log('Movira web: http://127.0.0.1:3000 (Ctrl+C to stop)'));
