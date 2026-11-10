'use client';
import {useEffect,useRef,useState} from 'react';
import type {KeyboardEvent,PointerEvent} from 'react';
import {compact,linearExtent,logExtent,logTicks,project} from './lib/scaling';
import type {PlannedPoint,ScalingMetric,ScalingPoint,View} from './lib/scaling';
import {LADDER_STATUS} from './lib/scaling';

export type FamilyTone={name:string;ink:string;mid:string;soft:string};
const clamp=(v:number,lo:number,hi:number)=>Math.max(lo,Math.min(hi,v));
const scientific=(v?:number)=>{if(!v)return '—';const e=Math.floor(Math.log10(v));return `${(v/10**e).toFixed(2)} × 10^${e}`;};

export default function ScalingPlot({points,planned,metric,view,onView,onReset,showPlanned,tones}:{points:ScalingPoint[];planned:PlannedPoint[];metric:ScalingMetric;view:View;onView:(view:View)=>void;onReset:()=>void;showPlanned:boolean;tones:(family:string)=>FamilyTone}){
 const wrap=useRef<HTMLDivElement>(null),drag=useRef<{x:number;y:number;view:View;moved:boolean}|null>(null);
 const [width,setWidth]=useState(900),[hover,setHover]=useState('');
 useEffect(()=>{if(!wrap.current)return;const observer=new ResizeObserver(([e])=>setWidth(Math.max(300,e.contentRect.width)));observer.observe(wrap.current);return()=>observer.disconnect();},[]);
 const W=width,H=Math.round(clamp(W*.56,320,620)),M={l:W<560?46:78,r:W<560?16:44,t:30,b:W<560?40:58};
 const value=(p:ScalingPoint)=>metric==='core'?p.core*100:p.bpb;
 const floor=showPlanned?planned:[];
 const xd=logExtent([...points,...floor].map(p=>p.params)),yd=logExtent([...points,...floor].map(p=>p.tokens)),zd=linearExtent(points.map(value));
 const nx=(v:number)=>(Math.log10(v)-xd[0])/(xd[1]-xd[0])*2-1,ny=(v:number)=>1-(Math.log10(v)-yd[0])/(yd[1]-yd[0])*2,nz=(v:number)=>(v-zd[0])/(zd[1]-zd[0])*2-1;
 const corners=[-1,1].flatMap(x=>[-1,1].flatMap(y=>[-1,1].map(z=>project(x,y,z,view))));
 const [x0,x1]=[Math.min(...corners.map(c=>c.x)),Math.max(...corners.map(c=>c.x))],[y0,y1]=[Math.min(...corners.map(c=>c.y)),Math.max(...corners.map(c=>c.y))];
 const k=Math.min((W-M.l-M.r)/(x1-x0||1),(H-M.t-M.b)/(y1-y0||1));
 const ox=M.l+(W-M.l-M.r-(x1-x0)*k)/2-x0*k,oy=M.t+(H-M.t-M.b-(y1-y0)*k)/2-y0*k;
 const at=(x:number,y:number,z:number)=>{const p=project(x,y,z,view);return {x:ox+p.x*k,y:oy+p.y*k,depth:p.depth};};
 const line=(a:number[],b:number[])=>{const p=at(a[0],a[1],a[2]),q=at(b[0],b[1],b[2]);return {x1:p.x,y1:p.y,x2:q.x,y2:q.y};};
 const length=(a:number[],b:number[])=>{const l=line(a,b);return Math.hypot(l.x2-l.x1,l.y2-l.y1);};
 // The walls farthest from the viewer carry the grid; tick labels sit on the floor edges nearest to it.
 const xs=at(1,0,0).depth<at(-1,0,0).depth?1:-1,ys=at(0,1,0).depth<at(0,-1,0).depth?1:-1;
 const xTicks=logTicks(xd),yTicks=logTicks(yd),zTicks=Array.from({length:5},(_,i)=>zd[0]+(zd[1]-zd[0])*i/4);
 const showX=length([-1,-ys,-1],[1,-ys,-1])>60,showY=length([-xs,-1,-1],[-xs,1,-1])>60,showZ=length([xs,ys,-1],[xs,ys,1])>60;
 const edges=[[xs,-ys],[-xs,ys],[xs,ys]].map(([x,y])=>({x,y,screen:at(x,y,0).x}));
 const zEdge=edges.reduce((a,b)=>b.screen<a.screen?b:a);
 const measured=points.map(p=>({p,top:at(nx(p.params),ny(p.tokens),nz(value(p))),base:at(nx(p.params),ny(p.tokens),-1)})).sort((a,b)=>a.top.depth-b.top.depth);
 const rings=floor.map(p=>({p,base:at(nx(p.params),ny(p.tokens),-1)}));
 const tilted=view.pitch>.06;
 const active=measured.find(m=>m.p.id===hover),ring=rings.find(r=>r.p.id===hover);
 function locate(e:PointerEvent<SVGSVGElement>){const r=e.currentTarget.getBoundingClientRect();return {x:(e.clientX-r.left)/r.width*W,y:(e.clientY-r.top)/r.height*H};}
 function move(e:PointerEvent<SVGSVGElement>){
  const d=drag.current;
  if(d){const dx=e.clientX-d.x,dy=e.clientY-d.y;if(Math.hypot(dx,dy)>3)d.moved=true;onView({yaw:d.view.yaw-dx*.008,pitch:clamp(d.view.pitch+dy*.008,0,Math.PI/2)});return;}
  const m=locate(e),targets=[...measured.map(x=>({id:x.p.id,...x.top})),...rings.map(x=>({id:x.p.id,...x.base}))];
  const near=targets.reduce<{id:string;d:number}|null>((best,t)=>{const dist=Math.hypot(t.x-m.x,t.y-m.y);return dist<14&&(!best||dist<best.d)?{id:t.id,d:dist}:best;},null);
  setHover(near?.id??'');
 }
 function key(e:KeyboardEvent<SVGSVGElement>){
  const step={ArrowLeft:[.08,0],ArrowRight:[-.08,0],ArrowUp:[0,-.08],ArrowDown:[0,.08]}[e.key];
  if(step){e.preventDefault();onView({yaw:view.yaw+step[0],pitch:clamp(view.pitch+step[1],0,Math.PI/2)});}
  if(e.key==='0'||e.key==='Home')onReset();
 }
 const tip=active?{x:active.top.x,y:active.top.y}:ring?{x:ring.base.x,y:ring.base.y}:null;
 return <div className="scaling-plot" ref={wrap}>
  <svg viewBox={`0 0 ${W} ${H}`} role="img" tabIndex={0} aria-label={`Finished runs by parameters, training tokens and ${metric==='core'?'CORE':'validation BPB'}; drag or use the arrow keys to rotate`}
   onPointerDown={e=>{e.currentTarget.setPointerCapture(e.pointerId);drag.current={x:e.clientX,y:e.clientY,view,moved:false};}}
   onPointerMove={move} onPointerUp={e=>{e.currentTarget.releasePointerCapture(e.pointerId);drag.current=null;}} onPointerLeave={()=>{if(!drag.current)setHover('');}}
   onDoubleClick={onReset} onKeyDown={key}>
   <polygon className="scaling-floor" points={[[-1,-1],[1,-1],[1,1],[-1,1]].map(([x,y])=>{const p=at(x,y,-1);return `${p.x},${p.y}`;}).join(' ')}/>
   {xTicks.map(t=><line key={'fx'+t} className="scaling-grid" {...line([nx(t),-1,-1],[nx(t),1,-1])}/>)}
   {yTicks.map(t=><line key={'fy'+t} className="scaling-grid" {...line([-1,ny(t),-1],[1,ny(t),-1])}/>)}
   {view.pitch<1.5?zTicks.map(t=><g key={'z'+t}><line className="scaling-grid" {...line([xs,-1,nz(t)],[xs,1,nz(t)])}/><line className="scaling-grid" {...line([-1,ys,nz(t)],[1,ys,nz(t)])}/></g>):null}
   {[[xs,ys],[xs,-ys],[-xs,ys]].map(([x,y])=><line key={x+','+y} className="scaling-edge" {...line([x,y,-1],[x,y,1])}/>)}
   {showX&&xTicks.map(t=>{const p=at(nx(t),-ys*1.13,-1);return <text key={'lx'+t} x={p.x} y={p.y+4} textAnchor="middle" className="axis-text">{compact(t)}</text>;})}
   {showY&&yTicks.map(t=>{const p=at(-xs*1.15,ny(t),-1);return <text key={'ly'+t} x={p.x} y={p.y+4} textAnchor="middle" className="axis-text">{compact(t)}</text>;})}
   {showZ&&zTicks.map(t=>{const p=at(zEdge.x,zEdge.y,nz(t));return <text key={'lz'+t} x={p.x-9} y={p.y+4} textAnchor="end" className="axis-text">{metric==='core'?t.toFixed(1):t.toFixed(3)}</text>;})}
   {showX&&(()=>{const p=at(0,-ys*1.34,-1);return <text x={p.x} y={p.y+4} textAnchor="middle" className="scaling-axis-title">Parameters</text>;})()}
   {showY&&(()=>{const p=at(-xs*1.4,0,-1);return <text x={p.x} y={p.y+4} textAnchor="middle" className="scaling-axis-title">Training tokens</text>;})()}
   {showZ&&(()=>{const p=at(zEdge.x,zEdge.y,1.16);return <text x={p.x} y={p.y} textAnchor="middle" className="scaling-axis-title">{metric==='core'?'CORE ×100 ↑':'Validation BPB ↓'}</text>;})()}
   {rings.map(({p,base})=>{const t=tones(p.family);return <circle key={p.id} cx={base.x} cy={base.y} r={hover===p.id?6:4.2} className="scaling-planned" stroke={t.ink} fill={hover===p.id?t.soft:'#FFFEFD'}/>;})}
   {measured.map(({p,top,base})=>{const t=tones(p.family),on=hover===p.id,dim=hover&&!on;return <g key={p.id} opacity={dim?.35:1} style={{transition:'opacity 140ms'}}>
    {tilted&&<><line x1={base.x} y1={base.y} x2={top.x} y2={top.y} stroke={t.mid} strokeDasharray="2 4"/><circle cx={base.x} cy={base.y} r={2.2} fill={t.mid} opacity={.55}/></>}
    {p.stage==='scaling'?<rect x={top.x-(on?7:5.6)} y={top.y-(on?7:5.6)} width={on?14:11.2} height={on?14:11.2} transform={`rotate(45 ${top.x} ${top.y})`} fill={t.ink} stroke="#FFFEFD" strokeWidth={1.6}/>:<circle cx={top.x} cy={top.y} r={on?7.5:5.4} fill={t.ink} stroke="#FFFEFD" strokeWidth={1.6}/>}
   </g>;})}
  </svg>
  {tip&&<div className="chart-tooltip scaling-tooltip" style={{left:`${tip.x/W*100}%`,top:`${tip.y/H*100}%`}}>
   {active?<><small>{active.p.stage==='scaling'?`Scaling stage · ${active.p.context.toLocaleString('en-US')}-token context`:`49B protocol · ${active.p.context.toLocaleString('en-US')}-token context`}</small>
    <div><span style={{color:tones(active.p.family).ink}}>● {active.p.name}</span><b>{tones(active.p.family).name}</b></div>
    <div><span>Parameters</span><b>{compact(active.p.params)}{active.p.active?` · ${compact(active.p.active)} in blocks`:''}</b></div>
    <div><span>Training tokens</span><b>{compact(active.p.tokens)}</b></div>
    <div><span>Training FLOPs</span><b>{scientific(active.p.flops)}</b></div>
    <div><span>CORE ×100</span><b>{(active.p.core*100).toFixed(2)}</b></div>
    <div><span>Validation BPB</span><b>{active.p.bpb.toFixed(4)}</b></div></>
   :ring&&<><small>Planned · {ring.p.tier} · {LADDER_STATUS[ring.p.status]??ring.p.status}</small>
    <div><span style={{color:tones(ring.p.family).ink}}>○ {tones(ring.p.family).name} · {ring.p.id.split('-').at(-1)}</span></div>
    <div><span>Parameters</span><b>{compact(ring.p.params)} · {compact(ring.p.active)} active in blocks</b></div>
    <div><span>Training tokens</span><b>{compact(ring.p.tokens)}</b></div></>}
  </div>}
 </div>;
}
