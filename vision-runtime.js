/* Coordinates the original WebGPU executor and original image/pose processing. */
(function(scope){
'use strict';
class Runtime{
  static async create(progress=()=>{},pose=true){
    const gpu=await MoviraGPU.GPU.create(),detector=await MoviraGPU.Graph.load(gpu,'models/person',v=>progress('사람 검출 가중치 준비',v*.5));await detector.pick('output',[0,1,2,3,4,5],'_people');
    const poseGraph=pose?await MoviraGPU.Graph.load(gpu,'models/pose',v=>progress('관절 가중치 준비',.5+v*.5)):null,result=new Runtime(detector,poseGraph);if(pose&&typeof MoviraHands!=='undefined'){progress('자체 손 인식 준비',.98);result.hands=await MoviraHands.Hands.create(gpu);}progress('준비 완료',1);return result;
  }
  constructor(detector,pose){this.detector=detector;this.pose=pose;this.calibration=new MoviraCalibration.Calibration();this.parameters=new MoviraGPU.Tensor([16]);this.detector.gpu.buffer(this.parameters);}
  upload(frame){const gpu=this.detector.gpu,count=frame.width*frame.height;if(!this.pixels||this.pixels.size!==count){this.pixels?.buffer?.destroy();this.pixels=new MoviraGPU.Tensor([count]);this.pixels.kind='u32';gpu.buffer(this.pixels);}gpu.device.queue.writeBuffer(this.pixels.buffer,0,frame.pixels);}
  async prepareInput(graph,frame,view,pose=false){
    const gpu=this.detector.gpu,width=pose?192:416,height=pose?256:416,p=new Float32Array(16);p[0]=frame.width;p[1]=frame.height;p[2]=frame.bgra?1:0;
    if(pose){p[3]=view.cx;p[4]=view.cy;p[5]=view.w;p[6]=view.h;}else{const ratio=Math.min(416/view.w,416/view.h);p[3]=view.x;p[4]=view.y;p[5]=view.w;p[6]=view.h;p[7]=Math.max(1,Math.floor(view.w*ratio));p[8]=Math.max(1,Math.floor(view.h*ratio));}
    gpu.device.queue.writeBuffer(this.parameters.buffer,0,p);
    const target=graph.tensors.get(graph.description.inputs[0].name),header=`fn pixel(xx:i32,yy:i32,c:u32)->f32{if(xx<0||yy<0||xx>=i32(x1[0])||yy>=i32(x1[1])){return ${pose?'0.0':'114.0'};}let bits=x0[u32(yy)*u32(x1[0])+u32(xx)];return f32((bits>>(c*8u))&255u);}fn bilinear(px:f32,py:f32,c:u32)->f32{let ix=i32(floor(px));let iy=i32(floor(py));let dx=px-floor(px);let dy=py-floor(py);return mix(mix(pixel(ix,iy,c),pixel(ix+1,iy,c),dx),mix(pixel(ix,iy+1,c),pixel(ix+1,iy+1,c),dx),dy);}`;
    const common=`let i=gid.x;if(i>=${width*height*3}u){return;}let c=i/${width*height}u;let px=i%${width}u;let py=(i/${width}u)%${height}u;`;
    const coordinates=pose?`let sx=x1[3]+(f32(px)-96.0)*x1[5]/192.0;let sy=x1[4]+(f32(py)-128.0)*x1[6]/256.0;let channel=select(c,2u-c,x1[2]>0.5);var v=bilinear(sx,sy,channel);var mean=123.675;var sigma=58.395;if(c==1u){mean=116.28;sigma=57.12;}if(c==2u){mean=103.53;sigma=57.375;}y[i]=(v-mean)/sigma;`:`if(f32(px)>=x1[7]||f32(py)>=x1[8]){y[i]=114.0;return;}let sx=x1[3]+(f32(px)+0.5)*x1[5]/x1[7]-0.5;let sy=x1[4]+(f32(py)+0.5)*x1[6]/x1[8]-0.5;let channel=select(2u-c,c,x1[2]>0.5);y[i]=bilinear(sx,sy,channel);`;
    const step=await gpu.kernel([this.pixels,this.parameters],target,common+coordinates,null,header);return step;
  }
  async proposals(frame,detail=true,minimum=.04){
    this.upload(frame);const people=[];for(const view of MoviraVision.views(frame.width,frame.height,detail)){const ratio=Math.min(416/view.w,416/view.h),transform={...view,size:416,sx:Math.max(1,Math.floor(view.w*ratio))/view.w,sy:Math.max(1,Math.floor(view.h*ratio))/view.h},preparation=await this.prepareInput(this.detector,frame,view),output=await this.detector.run(null,['_people'],[preparation]);people.push(...MoviraVision.decodePeople(output._people.data,transform,frame.width,frame.height,minimum));}return MoviraVision.nms(people,.45);
  }
  async infer(frame,options={}){
    const threshold=options.threshold??this.calibration.threshold,video=options.video===true;
    if(video&&(!this.motion||this.stream!==options.stream||this.frameSize!==`${frame.width}x${frame.height}`)){this.motion=new MoviraMotion.Tracker();this.stream=options.stream;this.frameSize=`${frame.width}x${frame.height}`;}
    const scan=!video||this.motion.begin(options.time),detail=options.detail!==false;
    let people=[];
    if(scan){people=await this.proposals(frame,detail,Math.min(.04,threshold));for(const person of people){person.confidence=this.calibration.score(person,frame.width,frame.height);person.detected=true;}people=people.filter(p=>p.confidence>=threshold);if(video)this.motion.searched(options.time,detail);}
    else this.upload(frame);
    if(video)for(const p of this.motion.candidates(options.time,frame.width,frame.height))if(!people.some(q=>MoviraVision.iou(p.box,q.box)>.25))people.push(p);
    const kept=MoviraVision.nms(people,.45);
    if(!this.pose)return kept;
    let unique=await this.posePeople(frame,kept);
    if(video&&!scan&&this.motion.unreliable(unique,options.time)){
      // Recover on this frame when a crop loses a dancer or jumps to a neighbor.
      const fresh=await this.proposals(frame,detail,Math.min(.04,threshold));for(const p of fresh){p.confidence=this.calibration.score(p,frame.width,frame.height);p.detected=true;}
      unique=await this.posePeople(frame,MoviraVision.nms(fresh.filter(p=>p.confidence>=threshold),.45));this.motion.searched(options.time,detail);
    }
    if(options.refine)unique=await this.refinePeople(frame,unique);
    const result=video?this.motion.update(unique,options.time,frame.width,frame.height):unique;
    if(this.hands&&options.hands!==false)await this.hands.infer(frame,result,options);
    return result;
  }
  async posePeople(frame,people){const poses=[];for(const person of people){const crop=MoviraVision.poseCrop(person.box),preparation=await this.prepareInput(this.pose,frame,crop,true),outputs=await this.pose.run(null,null,[preparation]);poses.push({...person,points:MoviraVision.decodePose(outputs.simcc_x.data,outputs.simcc_y.data,crop,frame.width,frame.height)});}return MoviraVision.mergeBodyFragments(MoviraVision.deduplicatePoses(poses.filter(MoviraVision.bodyEvidence)));}
  async refinePeople(frame,people){for(const person of people){const original=person.points.map(p=>({...p}));for(const region of MoviraVision.refinementRegions(person,frame.width,frame.height)){const crop=MoviraVision.poseCrop(region.box),preparation=await this.prepareInput(this.pose,frame,crop,true),outputs=await this.pose.run(null,null,[preparation]),fine=MoviraVision.decodePose(outputs.simcc_x.data,outputs.simcc_y.data,crop,frame.width,frame.height);let distance=0,count=0;for(const k of region.anchors)if(original[k].confidence>.35&&fine[k].confidence>.35){distance+=Math.hypot(original[k].x-fine[k].x,original[k].y-fine[k].y);count++;}if(count<2||distance/count>person.box.h*.09)continue;for(const k of region.indices){const a=original[k],b=fine[k];if(b.confidence<.3||b.confidence<a.confidence*.85||Math.hypot(a.x-b.x,a.y-b.y)>person.box.h*.18)continue;const weight=2*b.confidence/Math.max(.001,a.confidence+2*b.confidence);person.points[k]={x:a.x+(b.x-a.x)*weight,y:a.y+(b.y-a.y)*weight,confidence:Math.max(a.confidence*.95,b.confidence)};}}}return people;}
}
scope.MoviraRuntime=Runtime;
})(globalThis);
