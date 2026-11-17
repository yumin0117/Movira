/* Movira temporal pose tracking. Written here; no tracking/filter library. */
(function(scope){
'use strict';
const clamp=(x,a,b)=>Math.max(a,Math.min(b,x));
function sensitivityThreshold(value){return (850-7*clamp(Number.isFinite(+value)?+value:50,0,100))/1000;}
function thresholdSensitivity(value){return clamp((.85-value)/.007,0,100);}
const anchors=[1,2,3,8,9,10],limbs=[[2,4],[4,6],[3,5],[5,7],[9,11],[11,13],[10,12],[12,14]];
const distance=(a,b)=>Math.hypot(a.x-b.x,a.y-b.y);
function center(p){const a=anchors.map(k=>p.points[k]).filter(q=>q&&q.confidence>.3);return a.length>=3?{x:a.reduce((s,q)=>s+q.x,0)/a.length,y:a.reduce((s,q)=>s+q.y,0)/a.length}:{x:p.box.x+p.box.w/2,y:p.box.y+p.box.h/2};}
function overlap(a,b){const area=Math.max(0,Math.min(a.x+a.w,b.x+b.w)-Math.max(a.x,b.x))*Math.max(0,Math.min(a.y+a.h,b.y+b.h)-Math.max(a.y,b.y));return area/Math.max(1,a.w*a.h+b.w*b.h-area);}
function alpha(cutoff,dt){return 1-Math.exp(-2*Math.PI*cutoff*dt);}
class Tracker{
  constructor(){this.reset();}
  reset(){this.tracks=[];this.nextId=1;this.time=null;this.lastSearch=-Infinity;this.lastDetailed=-Infinity;this.sinceSearch=0;}
  begin(time){if(!Number.isFinite(time))throw Error('영상 시각이 올바르지 않습니다.');if(this.time!==null&&(time<=this.time||time-this.time>.5))this.reset();this.sinceSearch++;return !this.tracks.length||time-this.lastSearch>=.24||this.sinceSearch>=6;}
  searched(time,detailed){this.lastSearch=time;this.sinceSearch=0;if(detailed)this.lastDetailed=time;}
  unreliable(poses,time){if(poses.length<this.tracks.length)return true;return poses.some(p=>{const t=this.tracks.find(t=>t.id===p.trackId);if(!t)return false;const c=center(p),dt=clamp(time-t.time,0,.15),d=Math.hypot(c.x-t.center.x-t.vx*dt,c.y-t.center.y-t.vy*dt);return d>t.box.h*.16||p.points.filter(q=>q.confidence>.25).length<7;});}
  candidates(time,width,height){return this.tracks.filter(t=>time-t.lastDetection<.65).map(t=>{const dt=clamp(time-t.time,0,.15),dx=clamp(t.vx*dt,-t.box.h*.18,t.box.h*.18),dy=clamp(t.vy*dt,-t.box.h*.18,t.box.h*.18),x=clamp(t.box.x+dx,0,width-8),y=clamp(t.box.y+dy,0,height-14);return {box:{x,y,w:Math.min(t.box.w,width-x),h:Math.min(t.box.h,height-y)},raw:t.raw,confidence:t.confidence,trackId:t.id,detected:false};});}
  update(poses,time,width,height){
    const pairs=[];for(let i=0;i<this.tracks.length;i++)for(let j=0;j<poses.length;j++){
      const t=this.tracks[i],p=poses[j],dt=clamp(time-t.time,0,.2),c=center(p),scale=Math.max(30,t.box.h,p.box.h),d=Math.hypot(c.x-t.center.x-t.vx*dt,c.y-t.center.y-t.vy*dt)/scale;
      const ratio=Math.abs(Math.log(Math.max(.01,p.box.h/t.box.h)));if((d>.65&&overlap(t.box,p.box)<.1)||ratio>.9)continue;
      let shape=0,n=0;for(const k of anchors){const a=t.points[k],b=p.points[k];if(a?.confidence>.3&&b?.confidence>.3){shape+=Math.hypot((a.x-t.center.x)-(b.x-c.x),(a.y-t.center.y)-(b.y-c.y))/scale;n++;}}
      pairs.push({i,j,cost:d+.25*(1-overlap(t.box,p.box))+.15*ratio+(n?shape/n:0)-(p.trackId===t.id?.25:0)});
    }
    pairs.sort((a,b)=>a.cost-b.cost);const usedTracks=new Set(),matched=new Map();for(const m of pairs)if(!usedTracks.has(m.i)&&!matched.has(m.j)){usedTracks.add(m.i);matched.set(m.j,this.tracks[m.i]);}
    const next=[],output=[];for(let j=0;j<poses.length;j++){
      const p=poses[j],old=matched.get(j),dt=old?clamp(time-old.time,1/240,.5):1/30,c=center(p),scale=Math.max(30,p.box.h),dx=old?c.x-old.center.x:0,dy=old?c.y-old.center.y:0;
      if(p.detected===false&&(!old||p.points.filter(q=>q.confidence>.25).length<7||time-old.lastDetection>=.65))continue;
      const velocities=[],points=p.points.map((q,k)=>{
        if(!q||!Number.isFinite(q.x+q.y))return {x:0,y:0,confidence:0};const prev=old?.points[k];let measured={...q};
        if(prev?.confidence>.18&&q.confidence<.55){
          // Reject only isolated low-confidence leaps, not coherent whole-body motion.
          const residual=Math.hypot(q.x-prev.x-dx,q.y-prev.y-dy),neighbors=limbs.filter(b=>b.includes(k)).map(b=>b[0]===k?b[1]:b[0]);
          const stable=neighbors.some(n=>p.points[n]?.confidence>.35&&old.points[n]?.confidence>.35&&Math.hypot(p.points[n].x-old.points[n].x-dx,p.points[n].y-old.points[n].y-dy)<scale*.09);
          const stretched=neighbors.some(n=>{const a=p.points[n],b=old.points[n];return a?.confidence>.3&&b?.confidence>.3&&distance(q,a)>Math.max(scale*.42,distance(prev,b)*1.9);});
          if(stable&&residual>scale*.18&&stretched)measured.confidence=0;
        }
        if(!prev||prev.confidence<.18||measured.confidence<.18){velocities[k]={x:0,y:0};return measured;}
        const v={x:(measured.x-prev.x)/dt,y:(measured.y-prev.y)/dt},before=old.velocities[k]||v,va=alpha(4,dt),filtered={x:before.x+va*(v.x-before.x),y:before.y+va*(v.y-before.y)};velocities[k]=filtered;
        // Speed-adaptive cutoff: stable when still, responsive during kicks/turns.
        const speed=Math.hypot(filtered.x,filtered.y)/scale,a=alpha(3+18*speed,dt);
        return {...measured,x:prev.x+a*(measured.x-prev.x),y:prev.y+a*(measured.y-prev.y)};
      });
      const velocity=old?{x:clamp(dx/dt,-scale*5,scale*5),y:clamp(dy/dt,-scale*5,scale*5)}:{x:0,y:0};
      const t={...p,id:old?.id??this.nextId++,center:c,points,velocities,time,vx:velocity.x,vy:velocity.y,lastDetection:p.detected===false?old.lastDetection:time};
      // Fresh detections bound drift; tracked crops move with the torso between searches.
      if(old&&p.detected===false){const x=clamp(old.box.x+dx,0,width-8),y=clamp(old.box.y+dy,0,height-14);t.box={x,y,w:Math.min(old.box.w,width-x),h:Math.min(old.box.h,height-y)};}
      next.push(t);output.push({...p,id:t.id,points});
    }
    this.tracks=next;this.time=time;return output;
  }
}
const api={Tracker,sensitivityThreshold,thresholdSensitivity};scope.MoviraMotion=api;if(typeof module!=='undefined')module.exports=api;
})(globalThis);
