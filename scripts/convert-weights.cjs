/* Original, limited ONNX data reader. Reads protobuf fields; imports no code. */
const fs=require('node:fs'),path=require('node:path');
function varint(bytes,state){let n=0n,shift=0n;while(state.at<bytes.length){const b=bytes[state.at++];n|=BigInt(b&127)<<shift;if(!(b&128))return Number(BigInt.asIntN(64,n));shift+=7n;if(shift>70n)throw Error('Invalid varint');}throw Error('Truncated varint');}
function message(bytes){const fields=new Map(),state={at:0};while(state.at<bytes.length){const tag=varint(bytes,state),key=tag>>>3,wire=tag&7;let value;if(wire===0)value=varint(bytes,state);else if(wire===2){const length=varint(bytes,state);if(length<0||state.at+length>bytes.length)throw Error('Invalid protobuf length');value=bytes.subarray(state.at,state.at+length);state.at+=length;}else if(wire===5){value=bytes.subarray(state.at,state.at+4);state.at+=4;}else if(wire===1){value=bytes.subarray(state.at,state.at+8);state.at+=8;}else throw Error('Unsupported protobuf wire '+wire);if(!fields.has(key))fields.set(key,[]);fields.get(key).push({wire,value});}return fields;}
const all=(m,k)=>m.get(k)||[],one=(m,k,fallback)=>all(m,k)[0]?.value??fallback,string=(m,k)=>Buffer.from(one(m,k,[])).toString('utf8');
function integers(m,k){const result=[];for(const f of all(m,k)){if(f.wire===0)result.push(f.value);else{const s={at:0};while(s.at<f.value.length)result.push(varint(f.value,s));}}return result;}
function floats(m,k){const result=[];for(const f of all(m,k))for(let i=0;i<f.value.length;i+=4)result.push(f.value.readFloatLE(i));return result;}
function tensor(bytes){const m=message(bytes),dims=integers(m,1),type=one(m,2,1),raw=one(m,9),length=dims.reduce((a,b)=>a*b,1),data=new Float32Array(length);
  if(raw){for(let i=0;i<length;i++){if(type===1)data[i]=raw.readFloatLE(i*4);else if(type===6)data[i]=raw.readInt32LE(i*4);else if(type===7)data[i]=Number(raw.readBigInt64LE(i*8));else if(type===11)data[i]=raw.readDoubleLE(i*8);else if(type===9||type===2)data[i]=raw[i];else throw Error('Unsupported tensor type '+type);}}
  else{const values=type===1?floats(m,4):integers(m,type===7?7:5);data.set(values);}
  return {name:string(m,8),dims,data};
}
function convert(filename,prefix){
  const root=message(fs.readFileSync(filename)),graph=message(one(root,7)),constants=[],nodes=[],chunks=[];let offset=0;
  function add(t,name){constants.push({name,shape:t.dims,offset,length:t.data.length});const bytes=Buffer.from(t.data.buffer,t.data.byteOffset,t.data.byteLength);chunks.push(bytes);offset+=bytes.length;return name;}
  for(const field of all(graph,5)){const t=tensor(field.value);add(t,t.name);}
  for(const [index,field] of all(graph,1).entries()){
    const m=message(field.value),attrs={};
    for(const field of all(m,5)){const a=message(field.value),name=string(a,1),type=one(a,20,0);if(type===1)attrs[name]=one(a,2).readFloatLE(0);else if(type===2)attrs[name]=one(a,3);else if(type===3)attrs[name]=string(a,4);else if(type===6)attrs[name]=floats(a,7);else if(type===7)attrs[name]=integers(a,8);else if(type===4)attrs[name]=add(tensor(one(a,5)),`__constant_${index}_${name}`);else throw Error('Unsupported attribute type '+type);}
    const inputs=all(m,1).map(f=>f.value.toString('utf8')),outputs=all(m,2).map(f=>f.value.toString('utf8')),op=string(m,4);
    nodes.push({name:string(m,3)||`node_${index}`,op,inputs,outputs,attrs});
  }
  function value(field){const m=message(field.value),type=message(one(m,2)),tensorType=message(one(type,1)),shape=message(one(tensorType,2));return {name:string(m,1),shape:all(shape,1).map(f=>one(message(f.value),1,1))};}
  const model={format:'movira-graph-v1',sourceFile:path.basename(filename),inputs:all(graph,11).map(value).filter(v=>!constants.some(c=>c.name===v.name)),outputs:all(graph,12).map(value),constants,nodes};
  fs.mkdirSync(path.dirname(prefix),{recursive:true});fs.writeFileSync(prefix+'.json',JSON.stringify(model));fs.writeFileSync(prefix+'.bin',Buffer.concat(chunks));
  const counts={};for(const n of nodes)counts[n.op]=(counts[n.op]||0)+1;console.log(JSON.stringify({file:filename,inputs:model.inputs,outputs:model.outputs,nodes:nodes.length,ops:counts,weightBytes:offset}));
  return model;
}
if(require.main===module)convert(process.argv[2],process.argv[3]);module.exports={message,varint,tensor,convert};
