'use client';
import {useEffect,useId,useMemo,useRef,useState} from 'react';
import type {KeyboardEvent,PointerEvent,ReactNode} from 'react';
import {LADDER_STATUS,compact,denseFlops,isoFlopCurve,linearExtent,logExtent,logTicks,project} from './lib/scaling';
import type {PlannedPoint,ScalingMetric,ScalingPoint,View} from './lib/scaling';

export type FamilyTone={name:string;ink:string;mid:string;soft:string};
export type Sheet={id:string;flops:number;color:string;label:string};
const clamp=(v:number,lo:number,hi:number)=>Math.max(lo,Math.min(hi,v));
const scientific=(v?:number)=>{if(!v)return '—';const e=Math.floor(Math.log10(v));return `${(v/10**e).toFixed(2)} × 10^${e}`;};
// Camera distance in box units: enough perspective to read depth, little enough to keep the axes straight.
const DISTANCE=6.5,FIELD_CELLS=28;
const FIELD_RAMP=['#E4EDF5','#ECE7F4','#F8E8E1'];
const hex=(c:string)=>[1,3,5].map(i=>parseInt(c.slice(i,i+2),16));
function ramp(t:number){
 const [a,b,u]=t<.5?[FIELD_RAMP[0],FIELD_RAMP[1],t*2]:[FIELD_RAMP[1],FIELD_RAMP[2],t*2-1];
 const p=hex(a),q=hex(b);return `rgb(${p.map((v,i)=>Math.round(v+(q[i]-v)*u)).join(',')})`;
}

export default function ScalingPlot({points,planned,metric,view,onView,onReset,onInteract,showPlanned,showSheets,sheets,tones}:{points:ScalingPoint[];planned:PlannedPoint[];metric:ScalingMetric;view:View;onView:(view:View)=>void;onReset:()=>void;onInteract:()=>void;showPlanned:boolean;showSheets:boolean;sheets:Sheet[];tones:(family:string)=>FamilyTone}){
 const uid=useId().replace(/[^a-zA-Z0-9_-]/g,'');
 const wrap=useRef<HTMLDivElement>(null),drag=useRef<{x:number;y:number;view:View}|null>(null);
 const [width,setWidth]=useState(900),[hover,setHover]=useState('');
 useEffect(()=>{if(!wrap.current)return;const observer=new ResizeObserver(([e])=>setWidth(Math.max(300,e.contentRect.width)));observer.observe(wrap.current);return()=>observer.disconnect();},[]);
 const W=width,H=Math.round(clamp(W*.58,330,640)),M={l:W<560?40:70,r:W<560?14:36,t:36,b:W<560?36:54};
 const value=(p:ScalingPoint)=>metric==='core'?p.core*100:p.bpb;
 const floor=showPlanned?planned:[];
 const xd=logExtent([...points,...floor].map(p=>p.params),.1),yd=logExtent([...points,...floor].map(p=>p.tokens),.1),zd=linearExtent(points.map(value));
 const nx=(v:number)=>(Math.log10(v)-xd[0])/(xd[1]-xd[0])*2-1,ny=(v:number)=>1-(Math.log10(v)-yd[0])/(yd[1]-yd[0])*2,nz=(v:number)=>(v-zd[0])/(zd[1]-zd[0])*2-1;
 const ux=(x:number)=>10**(xd[0]+(x+1)/2*(xd[1]-xd[0])),uy=(y:number)=>10**(yd[0]+(1-y)/2*(yd[1]-yd[0]));
 const camera=(x:number,y:number,z:number)=>{const p=project(x,y,z,view),s=DISTANCE/(DISTANCE-p.depth);return {x:p.x*s,y:p.y*s,depth:p.depth,s};};
 const corners=[-1,1].flatMap(x=>[-1,1].flatMap(y=>[-1,1].map(z=>camera(x,y,z))));
 const [x0,x1]=[Math.min(...corners.map(c=>c.x)),Math.max(...corners.map(c=>c.x))],[y0,y1]=[Math.min(...corners.map(c=>c.y)),Math.max(...corners.map(c=>c.y))];
 const k=Math.min((W-M.l-M.r)/(x1-x0||1),(H-M.t-M.b)/(y1-y0||1));
 const ox=M.l+(W-M.l-M.r-(x1-x0)*k)/2-x0*k,oy=M.t+(H-M.t-M.b-(y1-y0)*k)/2-y0*k;
 const at=(x:number,y:number,z:number)=>{const p=camera(x,y,z);return {x:ox+p.x*k,y:oy+p.y*k,depth:p.depth,s:p.s};};
 const poly=(list:number[][])=>list.map(([x,y,z])=>{const p=at(x,y,z);return `${p.x.toFixed(1)},${p.y.toFixed(1)}`;}).join(' ');
 const line=(a:number[],b:number[])=>{const p=at(a[0],a[1],a[2]),q=at(b[0],b[1],b[2]);return {x1:p.x,y1:p.y,x2:q.x,y2:q.y};};
 const length=(a:number[],b:number[])=>{const l=line(a,b);return Math.hypot(l.x2-l.x1,l.y2-l.y1);};
 // The training compute of every floor position under the Dense size rule, shaded as a field.
 const field=useMemo(()=>Array.from({length:FIELD_CELLS},(_,i)=>Array.from({length:FIELD_CELLS},(_,j)=>Math.log10(denseFlops(ux(-1+(i+.5)*2/FIELD_CELLS),uy(-1+(j+.5)*2/FIELD_CELLS))))),[xd[0],xd[1],yd[0],yd[1]]);
 const [f0,f1]=[Math.min(...field.flat()),Math.max(...field.flat())];
 const inside=(p:{x:number;y:number})=>Math.abs(p.x)<=1.0001&&Math.abs(p.y)<=1.0001;
 const curve=(flops:number)=>isoFlopCurve(flops).map(p=>({x:nx(p.parameters),y:ny(p.tokens),layers:p.layers})).filter(inside);
 const contours=[];for(let e=Math.floor(f0);e<=Math.ceil(f1);e++)for(const m of [1,3]){const c=m*10**e;if(Math.log10(c)>f0&&Math.log10(c)<f1&&!(showSheets&&sheets.some(s=>Math.abs(Math.log10(s.flops/c))<.05)))contours.push(c);}
 // The walls farthest from the viewer carry the grid; tick labels sit on the floor edges nearest to it.
 const xs=at(1,0,0).depth<at(-1,0,0).depth?1:-1,ys=at(0,1,0).depth<at(0,-1,0).depth?1:-1;
 const xTicks=logTicks(xd),yTicks=logTicks(yd),zTicks=Array.from({length:5},(_,i)=>zd[0]+(zd[1]-zd[0])*i/4);
 const showX=length([-1,-ys,-1],[1,-ys,-1])>60,showY=length([-xs,-1,-1],[-xs,1,-1])>60,showZ=length([xs,ys,-1],[xs,ys,1])>60;
 const zEdge=[[xs,-ys],[-xs,ys],[xs,ys]].map(([x,y])=>({x,y,screen:at(x,y,0).x})).reduce((a,b)=>b.screen<a.screen?b:a);
 const tilted=view.pitch>.06;
 const measured=points.map(p=>({p,top:at(nx(p.params),ny(p.tokens),nz(value(p))),base:at(nx(p.params),ny(p.tokens),-1)}));
 const rings=floor.map(p=>({p,base:at(nx(p.params),ny(p.tokens),-1)}));
 const depths=measured.map(m=>m.top.depth),[d0,d1]=[Math.min(...depths,0),Math.max(...depths,1)];
 const fade=(depth:number)=>.55+.45*(depth-d0)/((d1-d0)||1);
 const active=measured.find(m=>m.p.id===hover),ring=rings.find(r=>r.p.id===hover);

 const layers:{depth:number;node:ReactNode}[]=[];
 if(showSheets)for(const s of sheets){
  const c=curve(s.flops);if(c.length<2)continue;
  const mesh=c.filter(p=>p.layers%4===0);
  const end=[c[0],c[c.length-1]].map(p=>({p,depth:at(p.x,p.y,-1).depth})).reduce((a,b)=>b.depth>a.depth?b:a).p,label=at(end.x,end.y,-.82);
  const mid=c[Math.floor(c.length/2)],low=at(mid.x,mid.y,-1),high=at(mid.x,mid.y,1),gid=`${uid}-sheet-${s.id}`;
  layers.push({depth:c.reduce((a,p)=>a+at(p.x,p.y,0).depth,0)/c.length,node:<g key={'sheet'+s.id} className="scaling-sheet">
   <defs><linearGradient id={gid} gradientUnits="userSpaceOnUse" x1={low.x} y1={low.y} x2={high.x} y2={high.y}><stop offset="0" stopColor={s.color} stopOpacity=".2"/><stop offset=".55" stopColor={s.color} stopOpacity=".06"/><stop offset="1" stopColor={s.color} stopOpacity="0"/></linearGradient></defs>
   <polygon points={poly([...c.map(p=>[p.x,p.y,-1]),...[...c].reverse().map(p=>[p.x,p.y,1])])} fill={`url(#${gid})`} stroke="none"/>
   {zTicks.slice(1,-1).map((t,i)=><polyline key={t} points={poly(c.map(p=>[p.x,p.y,nz(t)]))} fill="none" stroke={s.color} strokeOpacity={.16-i*.04}/>)}
   {mesh.map(p=><line key={p.layers} {...line([p.x,p.y,-1],[p.x,p.y,.2])} stroke={s.color} strokeOpacity={.13}/>)}
   <polyline points={poly(c.map(p=>[p.x,p.y,-1]))} fill="none" stroke={s.color} strokeOpacity={.8} strokeWidth={1.8}/>
   <text x={label.x} y={label.y} textAnchor="middle" className="scaling-sheet-label" fill={s.color}>{s.label}</text>
  </g>});
 }
 for(const {p,base} of rings){const t=tones(p.family),on=hover===p.id;layers.push({depth:base.depth,node:<circle key={p.id} cx={base.x} cy={base.y} r={(on?6:4.2)*base.s} className="scaling-planned" stroke={t.ink} fill={on?t.soft:'#FFFEFD'}/>});}
 for(const {p,top,base} of measured){
  const t=tones(p.family),on=hover===p.id,r=(on?8:6.2)*top.s,named=p.stage==='scaling'||p.id==='dense_full',o=hover&&!on?.28:named?Math.max(.85,fade(top.depth)):fade(top.depth),fill=`url(#${uid}-orb-${p.family})`;
  layers.push({depth:top.depth,node:<g key={p.id}><g opacity={o} style={{transition:'opacity 140ms'}}>
   {tilted&&<><line x1={base.x} y1={base.y} x2={top.x} y2={top.y} stroke={t.mid} strokeWidth={1} strokeDasharray="1.5 3.5"/><ellipse cx={base.x} cy={base.y} rx={3.4*base.s} ry={1.7*base.s} fill={t.ink} opacity={.2}/></>}
   <circle cx={top.x} cy={top.y} r={r*2.2} fill={t.ink} opacity={on?.28:.14} filter={`url(#${uid}-glow)`}/>
   {p.stage==='scaling'?<rect x={top.x-r} y={top.y-r} width={2*r} height={2*r} rx={1.5} transform={`rotate(45 ${top.x} ${top.y})`} fill={fill} stroke="#FFFFFF" strokeWidth={1.3}/>:<circle cx={top.x} cy={top.y} r={r} fill={fill} stroke="#FFFFFF" strokeWidth={1.2}/>}
  </g>{named&&!(hover&&!on)&&<text x={top.x+r+7} y={top.y+4} className="scaling-point-label" fill={t.ink}>{p.stage==='scaling'?p.name:'Dense · 49B tokens'}</text>}</g>});
 }
 layers.sort((a,b)=>a.depth-b.depth);

 function locate(e:PointerEvent<SVGSVGElement>){const r=e.currentTarget.getBoundingClientRect();return {x:(e.clientX-r.left)/r.width*W,y:(e.clientY-r.top)/r.height*H};}
 function move(e:PointerEvent<SVGSVGElement>){
  const d=drag.current;
  if(d){onView({yaw:d.view.yaw-(e.clientX-d.x)*.008,pitch:clamp(d.view.pitch+(e.clientY-d.y)*.008,0,Math.PI/2)});return;}
  const m=locate(e),targets=[...measured.map(x=>({id:x.p.id,...x.top})),...rings.map(x=>({id:x.p.id,...x.base}))];
  const near=targets.reduce<{id:string;d:number}|null>((best,t)=>{const dist=Math.hypot(t.x-m.x,t.y-m.y);return dist<14&&(!best||dist<best.d)?{id:t.id,d:dist}:best;},null);
  setHover(near?.id??'');
 }
 function key(e:KeyboardEvent<SVGSVGElement>){
  const step={ArrowLeft:[.08,0],ArrowRight:[-.08,0],ArrowUp:[0,-.08],ArrowDown:[0,.08]}[e.key];
  if(step){e.preventDefault();onInteract();onView({yaw:view.yaw+step[0],pitch:clamp(view.pitch+step[1],0,Math.PI/2)});}
  if(e.key==='0'||e.key==='Home')onReset();
 }
 const tip=active?{x:active.top.x,y:active.top.y}:ring?{x:ring.base.x,y:ring.base.y}:null;
 const families=[...new Set([...points,...floor].map(p=>p.family))];
 return <div className="scaling-plot" ref={wrap}>
  <svg viewBox={`0 0 ${W} ${H}`} role="img" tabIndex={0} aria-label={`Finished runs by parameters, training tokens and ${metric==='core'?'CORE':'validation BPB'}; drag or use the arrow keys to rotate`}
   onPointerDown={e=>{e.currentTarget.setPointerCapture(e.pointerId);onInteract();drag.current={x:e.clientX,y:e.clientY,view};}}
   onPointerMove={move} onPointerUp={e=>{e.currentTarget.releasePointerCapture(e.pointerId);drag.current=null;}} onPointerLeave={()=>{if(!drag.current)setHover('');}}
   onDoubleClick={onReset} onKeyDown={key}>
   <defs>
    <filter id={uid+'-glow'} x="-60%" y="-60%" width="220%" height="220%"><feGaussianBlur stdDeviation="3.4"/></filter>
    <linearGradient id={uid+'-wall'} x1="0" y1="0" x2="0" y2="1"><stop offset="0" stopColor="#FFFFFF" stopOpacity="0"/><stop offset="1" stopColor="#F3F5F9"/></linearGradient>
    {families.map(f=>{const t=tones(f);return <radialGradient key={f} id={`${uid}-orb-${f}`} cx="34%" cy="30%" r="78%"><stop offset="0" stopColor="#FFFFFF" stopOpacity=".95"/><stop offset=".4" stopColor={t.mid}/><stop offset="1" stopColor={t.ink}/></radialGradient>;})}
   </defs>
   {view.pitch<1.5&&<><polygon points={poly([[xs,-1,-1],[xs,1,-1],[xs,1,1],[xs,-1,1]])} fill={`url(#${uid}-wall)`} className="scaling-wall"/><polygon points={poly([[-1,ys,-1],[1,ys,-1],[1,ys,1],[-1,ys,1]])} fill={`url(#${uid}-wall)`} className="scaling-wall"/>
    {zTicks.map(t=><g key={'z'+t}><line className="scaling-grid" {...line([xs,-1,nz(t)],[xs,1,nz(t)])}/><line className="scaling-grid" {...line([-1,ys,nz(t)],[1,ys,nz(t)])}/></g>)}</>}
   {field.flatMap((row,i)=>row.map((v,j)=>{const a=-1+i*2/FIELD_CELLS,b=-1+j*2/FIELD_CELLS,s=2/FIELD_CELLS+.004;return <polygon key={i+'-'+j} points={poly([[a,b,-1],[a+s,b,-1],[a+s,b+s,-1],[a,b+s,-1]])} fill={ramp((v-f0)/((f1-f0)||1))}/>;}))}
   {contours.map(c=>{const pts=curve(c);return pts.length>1?<polyline key={c} points={poly(pts.map(p=>[p.x,p.y,-1]))} className="scaling-contour"/>:null;})}
   {xTicks.map(t=><line key={'fx'+t} className="scaling-floor-grid" {...line([nx(t),-1,-1],[nx(t),1,-1])}/>)}
   {yTicks.map(t=><line key={'fy'+t} className="scaling-floor-grid" {...line([-1,ny(t),-1],[1,ny(t),-1])}/>)}
   <polygon points={poly([[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1]])} className="scaling-floor-edge"/>
   {showX&&xTicks.map(t=>{const p=at(nx(t),-ys*1.13,-1);return <text key={'lx'+t} x={p.x} y={p.y+4} textAnchor="middle" className="axis-text">{compact(t)}</text>;})}
   {showY&&yTicks.map(t=>{const p=at(-xs*1.16,ny(t),-1);return <text key={'ly'+t} x={p.x} y={p.y+4} textAnchor="middle" className="axis-text">{compact(t)}</text>;})}
   {showZ&&zTicks.map(t=>{const p=at(zEdge.x,zEdge.y,nz(t));return <text key={'lz'+t} x={p.x-9} y={p.y+4} textAnchor="end" className="axis-text">{metric==='core'?t.toFixed(1):t.toFixed(3)}</text>;})}
   {showX&&(()=>{const p=at(0,-ys*1.34,-1);return <text x={p.x} y={p.y+4} textAnchor="middle" className="scaling-axis-title">Parameters</text>;})()}
   {showY&&(()=>{const p=at(-xs*1.42,0,-1);return <text x={p.x} y={p.y+4} textAnchor="middle" className="scaling-axis-title">Training tokens</text>;})()}
   {showZ&&(()=>{const p=at(zEdge.x,zEdge.y,1.16);return <text x={p.x} y={p.y} textAnchor="middle" className="scaling-axis-title">{metric==='core'?'CORE ×100 ↑':'Validation BPB ↓'}</text>;})()}
   {layers.map(l=>l.node)}
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
