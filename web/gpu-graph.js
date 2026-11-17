/* Movira tensor execution, authored here using only the browser WebGPU API.
   Model graph/parameters are data; no third-party inference runtime is loaded. */
(function(scope){
'use strict';
const product=a=>a.reduce((p,v)=>p*v,1),strides=s=>s.map((_,i)=>product(s.slice(i+1))),float=n=>Number.isFinite(n)?(Number.isInteger(n)?`${n}.0`:String(n)):n<0?'-3e38':'3e38';
function broadcast(a,b){const r=[],n=Math.max(a.length,b.length);for(let i=0;i<n;i++){const x=a[a.length-n+i]||1,y=b[b.length-n+i]||1;if(x!==y&&x!==1&&y!==1)throw Error(`Broadcast mismatch ${a} / ${b}`);r.push(Math.max(x,y));}return r;}
function coordinateIndex(index,outShape,inShape){const os=strides(outShape),is=strides(inShape);let result=0;for(let j=0;j<inShape.length;j++){const k=j+outShape.length-inShape.length;if(inShape[j]>1)result+=(Math.floor(index/os[k])%outShape[k])*is[j];}return result;}
function indexCode(index,outShape,inShape){const os=strides(outShape),is=strides(inShape),terms=[];for(let j=0;j<inShape.length;j++){const k=j+outShape.length-inShape.length;if(inShape[j]>1)terms.push(`((${index}/${os[k]}u)%${outShape[k]}u)*${is[j]}u`);}return terms.join('+')||'0u';}
function sliceSpec(shape,starts,ends,axes,steps){const offset=shape.map(()=>0),step=shape.map(()=>1),out=[...shape];axes=axes||starts.map((_,i)=>i);steps=steps||starts.map(()=>1);for(let j=0;j<starts.length;j++){const a=(axes[j]+shape.length)%shape.length,d=shape[a],s=steps[j];if(s<=0)throw Error('Negative slice stride is not used by these models');let l=starts[j]<0?starts[j]+d:starts[j],r=ends[j]<0?ends[j]+d:ends[j];l=Math.max(0,Math.min(d,l));r=Math.max(0,Math.min(d,r));offset[a]=l;step[a]=s;out[a]=Math.max(0,Math.ceil((r-l)/s));}return {offset,step,out};}
function sliceCode(spec,inputShape){const os=strides(spec.out),is=strides(inputShape);return inputShape.map((_,j)=>`(${spec.offset[j]}u+((i/${os[j]}u)%${spec.out[j]}u)*${spec.step[j]}u)*${is[j]}u`).join('+');}
function reshape(input,requested){const shape=requested.map((n,i)=>n===0?input[i]:n),unknown=shape.indexOf(-1);if(unknown>=0)shape[unknown]=product(input)/-product(shape);if(shape.some(n=>!Number.isInteger(n)||n<0)||product(shape)!==product(input))throw Error('Invalid reshape');return shape;}
function unary(op,x,attrs={}){if(op==='Sigmoid')return 1/(1+Math.exp(-x));if(op==='Relu')return Math.max(0,x);if(op==='Sqrt')return Math.sqrt(x);if(op==='HardSigmoid')return Math.max(0,Math.min(1,(attrs.alpha??.2)*x+(attrs.beta??.5)));throw Error('Unsupported CPU unary '+op);}
class Tensor{
  constructor(shape,cpu=null){this.shape=shape;this.size=product(shape);this.cpu=cpu;this.buffer=null;}
  view(shape){const result=new Tensor(shape,this.cpu);result.buffer=this.buffer;result.base=this.base||this;return result;}
}
class GPU{
  static async create(){if(!navigator.gpu)throw Error('GPU 분석에는 최신 Chrome 또는 Edge가 필요합니다.');const adapter=await navigator.gpu.requestAdapter();if(!adapter)throw Error('이 브라우저에서 GPU 분석 장치를 찾지 못했습니다.');const device=await adapter.requestDevice({requiredLimits:{maxStorageBufferBindingSize:Math.min(adapter.limits.maxStorageBufferBindingSize,536870912),maxBufferSize:Math.min(adapter.limits.maxBufferSize,536870912),maxStorageBuffersPerShaderStage:Math.min(16,adapter.limits.maxStorageBuffersPerShaderStage)}});return new GPU(device);}
  constructor(device){this.device=device;this.pipelines=new Map();this.errors=[];device.addEventListener('uncapturederror',e=>this.errors.push(e.error.message));}
  buffer(t){if(t.base){t.buffer=this.buffer(t.base);return t.buffer;}if(!t.buffer){t.buffer=this.device.createBuffer({size:Math.max(4,t.size*4),usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC});if(t.cpu&&t.size)this.device.queue.writeBuffer(t.buffer,0,t.cpu);}return t.buffer;}
  async pipeline(source){let p=this.pipelines.get(source);if(!p){p=(async()=>{const module=this.device.createShaderModule({code:source}),info=await module.getCompilationInfo();const errors=info.messages.filter(m=>m.type==='error');if(errors.length)throw Error(errors.map(m=>m.message).join('\n'));return this.device.createComputePipelineAsync({layout:'auto',compute:{module,entryPoint:'main'}});})();this.pipelines.set(source,p);}return p;}
  async kernel(inputs,output,body,dispatch=null,header='',workgroup='128'){
    const declarations=inputs.map((t,i)=>`@group(0) @binding(${i}) var<storage,read> x${i}:array<${t.kind||'f32'}>;`).join('\n')+`\n@group(0) @binding(${inputs.length}) var<storage,read_write> y:array<f32>;`;
    const source=declarations+'\n'+header+`\n@compute @workgroup_size(${workgroup}) fn main(@builtin(global_invocation_id) gid:vec3<u32>,@builtin(local_invocation_id) lid:vec3<u32>,@builtin(workgroup_id) group:vec3<u32>){${body}}`;
    const pipeline=await this.pipeline(source);inputs.forEach(t=>this.buffer(t));this.buffer(output);
    const bind=this.device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[...inputs.map((t,i)=>({binding:i,resource:{buffer:t.buffer}})),{binding:inputs.length,resource:{buffer:output.buffer}}]});
    return {pipeline,bind,dispatch:dispatch||[Math.ceil(output.size/128),1,1]};
  }
  async read(t,encoder){const read=this.device.createBuffer({size:Math.max(4,t.size*4),usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST});encoder.copyBufferToBuffer(this.buffer(t),0,read,0,Math.max(4,t.size*4));return read;}
}
function fuse(nodes){const uses=new Map();for(const n of nodes)for(const i of n.inputs){if(!uses.has(i))uses.set(i,[]);uses.get(i).push(n);}const skip=new Set(),result=[];
  for(const original of nodes){if(skip.has(original))continue;const n={...original,outputs:[...original.outputs],attrs:{...original.attrs}};
    if(n.op==='Conv'){
      const name=n.outputs[0],consumers=uses.get(name)||[],sig=consumers.find(x=>x.op==='Sigmoid'),mul=sig&&consumers.find(x=>x.op==='Mul'&&x.inputs.includes(sig.outputs[0])&&x.inputs.includes(name));
      if(mul&&consumers.length===2&&(uses.get(sig.outputs[0])||[]).length===1){n.activation='silu';n.outputs=[mul.outputs[0]];skip.add(sig);skip.add(mul);}
      else if(consumers.length===1&&['Relu','Sigmoid','HardSigmoid'].includes(consumers[0].op)){const a=consumers[0];n.activation=a.op;n.activationAttrs=a.attrs;n.outputs=[a.outputs[0]];skip.add(a);}
    }result.push(n);
  }return result;
}
class Graph{
  static async load(gpu,prefix,progress=()=>{}){const [description,response]=await Promise.all([fetch(prefix+'.json').then(r=>{if(!r.ok)throw Error('모델 설명을 불러오지 못했습니다.');return r.json();}),fetch(prefix+'.bin')]);if(!response.ok)throw Error('모델 가중치를 불러오지 못했습니다.');const binary=await response.arrayBuffer(),graph=new Graph(gpu,description,binary);await graph.prepare(progress);return graph;}
  constructor(gpu,description,binary){this.gpu=gpu;this.description=description;this.tensors=new Map();this.steps=[];this.owned=new Set();for(const c of description.constants)this.tensors.set(c.name,new Tensor(c.shape,new Float32Array(binary,c.offset,c.length)));for(const input of description.inputs)this.tensors.set(input.name,new Tensor(input.shape));}
  async prepare(progress){const nodes=fuse(this.description.nodes);for(let i=0;i<nodes.length;i++){try{await this.node(nodes[i]);}catch(e){throw Error(nodes[i].name+' ('+nodes[i].op+'): '+e.message);}if(i%10===0)progress(i/nodes.length);}progress(1);}
  async node(node){const {op,attrs:a}=node,ins=node.inputs.map(name=>name?this.tensors.get(name):null),x=ins[0],values=t=>t?Array.from(t.cpu||[]):[],axis=(a.axis??0),set=(t,j=0)=>{this.tensors.set(node.outputs[j],t);return t;},out=s=>set(new Tensor(s));
    for(let i=0;i<ins.length;i++)if(node.inputs[i]&&!ins[i])throw Error('Missing tensor '+node.inputs[i]);
    const add=async(inputs,output,body,dispatch,header,wg)=>this.steps.push(await this.gpu.kernel(inputs,output,body,dispatch,header,wg));
    if(op==='Constant'){set(this.tensors.get(a.value));return;}
    if(op==='Shape'){set(new Tensor([x.shape.length],Float32Array.from(x.shape)));return;}
    if(op==='Reshape'||op==='Flatten'){set(x.view(op==='Reshape'?reshape(x.shape,values(ins[1])):[product(x.shape.slice(0,axis||1)),product(x.shape.slice(axis||1))]));return;}
    if(op==='Unsqueeze'||op==='Squeeze'){const axes=a.axes||values(ins[1]);let shape=[...x.shape];if(op==='Unsqueeze'){const rank=shape.length+axes.length;for(const i of axes.map(n=>(n+rank)%rank).sort((a,b)=>a-b))shape.splice(i,0,1);}else{const selected=axes.length?axes.map(n=>(n+shape.length)%shape.length):shape.map((d,i)=>d===1?i:-1);shape=shape.filter((_,i)=>!selected.includes(i));}set(x.view(shape));return;}
    if(op==='Slice'){const spec=sliceSpec(x.shape,values(ins[1]),values(ins[2]),ins[3]?values(ins[3]):null,ins[4]?values(ins[4]):null),t=out(spec.out),code=sliceCode(spec,x.shape);if(x.cpu){t.cpu=Float32Array.from({length:t.size},(_,i)=>{const os=strides(spec.out),is=strides(x.shape);let at=0;for(let j=0;j<x.shape.length;j++)at+=(spec.offset[j]+(Math.floor(i/os[j])%spec.out[j])*spec.step[j])*is[j];return x.cpu[at];});}else await add([x],t,`let i=gid.x;if(i>=${t.size}u){return;}y[i]=x0[${code}];`);return;}
    if(op==='Transpose'){const perm=a.perm||x.shape.map((_,i)=>x.shape.length-1-i),shape=perm.map(i=>x.shape[i]),t=out(shape),ss=strides(shape),is=strides(x.shape),code=perm.map((p,j)=>`((i/${ss[j]}u)%${shape[j]}u)*${is[p]}u`).join('+');if(x.cpu)t.cpu=Float32Array.from({length:t.size},(_,i)=>x.cpu[perm.reduce((n,p,j)=>n+(Math.floor(i/ss[j])%shape[j])*is[p],0)]);else await add([x],t,`let i=gid.x;if(i>=${t.size}u){return;}y[i]=x0[${code}];`);return;}
    if(op==='Concat'){
      const ax=(axis+x.shape.length)%x.shape.length,shape=[...x.shape];shape[ax]=ins.reduce((n,t)=>n+t.shape[ax],0);const t=out(shape),inner=product(shape.slice(ax+1)),axisBlock=shape[ax]*inner;let offset=0,code='';
      for(let j=0;j<ins.length;j++){const block=ins[j].shape[ax]*inner;code+=`${j?'else ':''}if(k<${offset+block}u){y[i]=x${j}[outer*${block}u+k-${offset}u];}`;offset+=block;}
      if(ins.every(t=>t.cpu)){t.cpu=new Float32Array(t.size);const outer=product(shape.slice(0,ax));for(let n=0;n<outer;n++){let offset=n*axisBlock;for(const input of ins){const count=input.shape[ax]*inner;t.cpu.set(input.cpu.subarray(n*count,(n+1)*count),offset);offset+=count;}}}
      else await add(ins,t,`let i=gid.x;if(i>=${t.size}u){return;}let outer=i/${axisBlock}u;let k=i%${axisBlock}u;${code}`);return;
    }
    if(op==='Split'){const ax=(axis+x.shape.length)%x.shape.length,parts=a.split||values(ins[1])||[],sizes=parts.length?parts:node.outputs.map(()=>x.shape[ax]/node.outputs.length);let offset=0;for(let j=0;j<sizes.length;j++){const shape=[...x.shape];shape[ax]=sizes[j];const t=set(new Tensor(shape),j),spec={out:shape,offset:shape.map((_,i)=>i===ax?offset:0),step:shape.map(()=>1)};if(x.cpu){const inner=product(shape.slice(ax+1)),outer=product(shape.slice(0,ax));t.cpu=new Float32Array(t.size);for(let n=0;n<outer;n++)t.cpu.set(x.cpu.subarray((n*x.shape[ax]+offset)*inner,(n*x.shape[ax]+offset+sizes[j])*inner),n*sizes[j]*inner);}else await add([x],t,`let i=gid.x;if(i>=${t.size}u){return;}y[i]=x0[${sliceCode(spec,x.shape)}];`);offset+=sizes[j];}return;}
    if(['Add','Sub','Mul','Div','Pow'].includes(op)){
      const b=ins[1],shape=broadcast(x.shape,b.shape),t=out(shape),calculation=(u,v)=>op==='Add'?u+v:op==='Sub'?u-v:op==='Mul'?u*v:op==='Div'?u/v:Math.pow(u,v);
      if(x.cpu&&b.cpu)t.cpu=Float32Array.from({length:t.size},(_,i)=>calculation(x.cpu[coordinateIndex(i,shape,x.shape)],b.cpu[coordinateIndex(i,shape,b.shape)]));
      else{const u=`x0[${indexCode('i',shape,x.shape)}]`,v=`x1[${indexCode('i',shape,b.shape)}]`,expression=op==='Pow'?`pow(${u},${v})`:`${u}${{Add:'+',Sub:'-',Mul:'*',Div:'/'}[op]}${v}`;await add([x,b],t,`let i=gid.x;if(i>=${t.size}u){return;}y[i]=${expression};`);}return;
    }
    if(['Sigmoid','Relu','HardSigmoid','Sqrt','Clip'].includes(op)){
      const t=out([...x.shape]),low=ins[1]?.cpu?.[0]??a.min??-Infinity,high=ins[2]?.cpu?.[0]??a.max??Infinity;
      if(x.cpu)t.cpu=Float32Array.from(x.cpu,v=>op==='Clip'?Math.max(low,Math.min(high,v)):unary(op,v,a));
      else{const expression=op==='Sigmoid'?'1.0/(1.0+exp(-v))':op==='Relu'?'max(v,0.0)':op==='Sqrt'?'sqrt(v)':op==='Clip'?`clamp(v,${float(low)},${float(high)})`:`clamp(v*${float(a.alpha??.2)}+${float(a.beta??.5)},0.0,1.0)`;await add([x],t,`let i=gid.x;if(i>=${t.size}u){return;}let v=x0[i];y[i]=${expression};`);}return;
    }
    if(op==='ReduceSum'||op==='GlobalAveragePool'){
      const axes=op==='GlobalAveragePool'?[2,3]:(a.axes||values(ins[1])||x.shape.map((_,i)=>i)),normalized=axes.map(i=>(i+x.shape.length)%x.shape.length),keep=op==='GlobalAveragePool'||(a.keepdims??1),outAxes=x.shape.map((_,i)=>i).filter(i=>!normalized.includes(i)),reduceShape=normalized.map(i=>x.shape[i]),shape=keep?x.shape.map((d,i)=>normalized.includes(i)?1:d):outAxes.map(i=>x.shape[i]),t=out(shape),ss=strides(x.shape),rs=strides(reduceShape),os=strides(outAxes.map(i=>x.shape[i])),count=product(reduceShape);
      const base=outAxes.map((p,j)=>`((i/${os[j]}u)%${x.shape[p]}u)*${ss[p]}u`).join('+')||'0u',reduced=normalized.map((p,j)=>`((k/${rs[j]}u)%${x.shape[p]}u)*${ss[p]}u`).join('+')||'0u';
      if(x.cpu){t.cpu=new Float32Array(t.size);for(let i=0;i<t.size;i++){let sum=0,base=0;for(let j=0;j<outAxes.length;j++)base+=(Math.floor(i/os[j])%x.shape[outAxes[j]])*ss[outAxes[j]];for(let k=0;k<count;k++){let at=base;for(let j=0;j<normalized.length;j++)at+=(Math.floor(k/rs[j])%reduceShape[j])*ss[normalized[j]];sum+=x.cpu[at];}t.cpu[i]=op==='GlobalAveragePool'?sum/count:sum;}}
      else await add([x],t,`let i=gid.x;if(i>=${t.size}u){return;}let base=${base};var sum=0.0;for(var k=0u;k<${count}u;k++){sum+=x0[base+${reduced}];}y[i]=sum${op==='GlobalAveragePool'?'/'+float(count):''};`);return;
    }
    if(op==='Resize'){
      const sizes=values(ins[3]),scales=values(ins[2]),shape=sizes.length?sizes:x.shape.map((d,i)=>Math.floor(d*scales[i])),t=out(shape),ih=x.shape[2],iw=x.shape[3],oh=shape[2],ow=shape[3];if(a.mode!=='nearest')throw Error('Only nearest resize occurs in the supported detector');
      await add([x],t,`let i=gid.x;if(i>=${t.size}u){return;}let px=i%${ow}u;let py=(i/${ow}u)%${oh}u;let plane=i/${ow*oh}u;let sx=min(${iw-1}u,u32(floor(f32(px)*${float(iw/ow)})));let sy=min(${ih-1}u,u32(floor(f32(py)*${float(ih/oh)})));y[i]=x0[plane*${iw*ih}u+sy*${iw}u+sx];`);return;
    }
    if(op==='MaxPool'){
      const [kh,kw]=a.kernel_shape,[sh,sw]=a.strides||[1,1],[pt,pl,pb,pr]=a.pads||[0,0,0,0],ih=x.shape[2],iw=x.shape[3],oh=Math.floor((ih+pt+pb-kh)/sh)+1,ow=Math.floor((iw+pl+pr-kw)/sw)+1,t=out([x.shape[0],x.shape[1],oh,ow]);
      await add([x],t,`let i=gid.x;if(i>=${t.size}u){return;}let px=i%${ow}u;let py=(i/${ow}u)%${oh}u;let plane=i/${ow*oh}u;var best=-3e38;for(var ky=0;ky<${kh};ky++){for(var kx=0;kx<${kw};kx++){let ix=i32(px)*${sw}+kx-${pl};let iy=i32(py)*${sh}+ky-${pt};if(ix>=0&&ix<${iw}&&iy>=0&&iy<${ih}){best=max(best,x0[plane*${iw*ih}u+u32(iy)*${iw}u+u32(ix)]);}}}y[i]=best;`);return;
    }
    if(op==='Conv'){
      const w=ins[1],bias=ins[2],[oc,ic,kh,kw]=w.shape,group=a.group||1,[sh,sw]=a.strides||[1,1],[dh,dw]=a.dilations||[1,1],[pt,pl,pb,pr]=a.pads||[0,0,0,0],ih=x.shape[2],iw=x.shape[3],oh=Math.floor((ih+pt+pb-dh*(kh-1)-1)/sh)+1,ow=Math.floor((iw+pl+pr-dw*(kw-1)-1)/sw)+1,t=out([x.shape[0],oc,oh,ow]);
      const activation=v=>node.activation==='silu'?`${v}/(1.0+exp(-${v}))`:node.activation==='Relu'?`max(${v},0.0)`:node.activation==='Sigmoid'?`1.0/(1.0+exp(-${v}))`:node.activation==='HardSigmoid'?`clamp(${v}*${float(node.activationAttrs.alpha??.2)}+${float(node.activationAttrs.beta??.5)},0.0,1.0)`:v;
      if(group===1){
        let columns;
        if(kh===1&&kw===1&&sh===1&&sw===1&&pt===0&&pl===0&&oh===ih&&ow===iw)columns=x.view([ic,oh*ow]);
        else{const count=ic*kh*kw*oh*ow;if(!this.columnBuffer||this.columnBuffer.size<count){const previous=this.columnBuffer;this.columnBuffer=new Tensor([count]);this.gpu.buffer(this.columnBuffer);if(previous)this.owned.add(previous.buffer);}columns=new Tensor([ic*kh*kw,oh*ow]);columns.buffer=this.columnBuffer.buffer;
          await add([x],columns,`let i=gid.x;if(i>=${count}u){return;}let at=i%${oh*ow}u;let k=i/${oh*ow}u;let kx=k%${kw}u;let ky=(k/${kw}u)%${kh}u;let c=k/${kh*kw}u;let ix=i32(at%${ow}u)*${sw}+i32(kx)*${dw}-${pl};let iy=i32(at/${ow}u)*${sh}+i32(ky)*${dh}-${pt};var v=0.0;if(ix>=0&&ix<${iw}&&iy>=0&&iy<${ih}){v=x0[c*${iw*ih}u+u32(iy)*${iw}u+u32(ix)];}y[i]=v;`);
        }
        await this.matmul(w.view([oc,ic*kh*kw]),columns,t,bias,activation);return;
      }
      const inputs=[x,w,...(bias?[bias]:[])];
      await add(inputs,t,`let i=gid.x;if(i>=${t.size}u){return;}let px=i%${ow}u;let py=(i/${ow}u)%${oh}u;let c=(i/${ow*oh}u)%${oc}u;let batch=i/${oc*ow*oh}u;let start=(c/${oc/group}u)*${ic}u;var v=${bias?'x2[c]':'0.0'};for(var ch=0u;ch<${ic}u;ch++){for(var ky=0u;ky<${kh}u;ky++){for(var kx=0u;kx<${kw}u;kx++){let ix=i32(px)*${sw}+i32(kx)*${dw}-${pl};let iy=i32(py)*${sh}+i32(ky)*${dh}-${pt};if(ix>=0&&ix<${iw}&&iy>=0&&iy<${ih}){v+=x0[((batch*${x.shape[1]}u+start+ch)*${ih}u+u32(iy))*${iw}u+u32(ix)]*x1[((c*${ic}u+ch)*${kh}u+ky)*${kw}u+kx];}}}}y[i]=${activation('v')};`);return;
    }
    if(op==='MatMul'){const b=ins[1],shape=[...broadcast(x.shape.slice(0,-2),b.shape.slice(0,-2)),x.shape.at(-2),b.shape.at(-1)],t=out(shape);await this.matmul(x,b,t);return;}
    throw Error('Unsupported model operation '+op);
  }
  async matmul(a,b,t,bias=null,activation=v=>v){
    const m=a.shape.at(-2),k=a.shape.at(-1),n=b.shape.at(-1);if(b.shape.at(-2)!==k)throw Error('MatMul inner dimensions differ');const batchShape=broadcast(a.shape.slice(0,-2),b.shape.slice(0,-2)),batches=product(batchShape),aOffset=indexCode('batch',batchShape,a.shape.slice(0,-2)),bOffset=indexCode('batch',batchShape,b.shape.slice(0,-2));
    const inputs=[a,b,...(bias?[bias]:[])],body=`let row=group.y*16u+lid.y;let col=group.x*16u+lid.x;let batch=group.z;let ao=(${aOffset})*${m*k}u;let bo=(${bOffset})*${k*n}u;var v=0.0;for(var tile=0u;tile<${Math.ceil(k/16)}u;tile++){let ak=tile*16u+lid.x;let bk=tile*16u+lid.y;var av=0.0;var bv=0.0;if(row<${m}u&&ak<${k}u){av=x0[ao+row*${k}u+ak];}if(col<${n}u&&bk<${k}u){bv=x1[bo+bk*${n}u+col];}aa[lid.y*16u+lid.x]=av;bb[lid.y*16u+lid.x]=bv;workgroupBarrier();for(var j=0u;j<16u;j++){v+=aa[lid.y*16u+j]*bb[j*16u+lid.x];}workgroupBarrier();}if(row<${m}u&&col<${n}u){${bias?'v+=x2[row];':''}y[(batch*${m}u+row)*${n}u+col]=${activation('v')};}`;
    this.steps.push(await this.gpu.kernel(inputs,t,body,[Math.ceil(n/16),Math.ceil(m/16),batches],'var<workgroup> aa:array<f32,256>;var<workgroup> bb:array<f32,256>;','16,16'));
  }
  async run(input,debugNames=null,preparation=[]){
    const description=this.description,inputTensor=this.tensors.get(description.inputs[0].name);if(input){if(input.length!==inputTensor.size)throw Error('Model input size differs');this.gpu.device.queue.writeBuffer(this.gpu.buffer(inputTensor),0,input);}
    const encoder=this.gpu.device.createCommandEncoder(),pass=encoder.beginComputePass();for(const list of [preparation,this.steps])for(const step of list){pass.setPipeline(step.pipeline);pass.setBindGroup(0,step.bind);pass.dispatchWorkgroups(...step.dispatch);}pass.end();
    const names=debugNames||description.outputs.map(o=>o.name),reads=[];for(const name of names){const t=this.tensors.get(name);if(!t)throw Error('Unknown output '+name);reads.push({name,t,buffer:await this.gpu.read(t,encoder)});}this.gpu.device.queue.submit([encoder.finish()]);
    const result={};for(const r of reads){await r.buffer.mapAsync(GPUMapMode.READ);result[r.name]={shape:r.t.shape,data:new Float32Array(r.buffer.getMappedRange()).slice(0,r.t.size)};r.buffer.unmap();r.buffer.destroy();}if(this.gpu.errors.length)throw Error(this.gpu.errors.splice(0).join('\n'));return result;
  }
  async pick(name,channels,target){const input=this.tensors.get(name),cols=input.shape.at(-1),rows=input.size/cols,tensor=new Tensor([rows,channels.length]);this.tensors.set(target,tensor);const selection=channels.map((c,i)=>`${i?'else ':''}if(c==${i}u){value=x0[row*${cols}u+${c}u];}`).join('');this.steps.push(await this.gpu.kernel([input],tensor,`let i=gid.x;if(i>=${tensor.size}u){return;}let row=i/${channels.length}u;let c=i%${channels.length}u;var value=0.0;${selection}y[i]=value;`));return target;}
}
scope.MoviraGPU={GPU,Graph,Tensor,broadcast,reshape,sliceSpec};if(typeof module!=='undefined')module.exports=scope.MoviraGPU;
})(globalThis);
