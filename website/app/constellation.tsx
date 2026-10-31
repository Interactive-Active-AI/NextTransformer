'use client';
import {useEffect,useId,useRef,useState} from 'react';
import type {CSSProperties} from 'react';
import {chartName} from './learning-chart';
import {FINAL_STEP,constellation,placeLabels,spread} from './lib/constellation';
import type {Star} from './lib/constellation';
import {tone,FAMILY_ORDER} from './lib/palette';
import type {Method,Run} from './lib/types';
import './constellation.css';

const signed=(v:number,n:number)=>(Math.abs(v)<.5*10**-n?'±':v>0?'+':'−')+Math.abs(v).toFixed(n);
const sparkle=(x:number,y:number,r:number)=>`M${x} ${y-r}Q${x+r*.18} ${y-r*.18} ${x+r} ${y}Q${x+r*.18} ${y+r*.18} ${x} ${y+r}Q${x-r*.18} ${y+r*.18} ${x-r} ${y}Q${x-r*.18} ${y-r*.18} ${x} ${y-r}Z`;

export default function Constellation({runs,methods,baseline,onModel}:{runs:Run[];methods:Method[];baseline?:Run;onModel:(method:Method)=>void}){
 const uid=useId().replace(/[^a-zA-Z0-9_-]/g,'');
 const wrap=useRef<HTMLDivElement>(null),[width,setWidth]=useState(540),[hover,setHover]=useState<Star|null>(null);
 useEffect(()=>{if(!wrap.current)return;const observer=new ResizeObserver(([e])=>setWidth(Math.max(280,e.contentRect.width)));observer.observe(wrap.current);return()=>observer.disconnect();},[]);
 const stars=constellation(runs.filter(r=>r.id!==baseline?.id),methods,baseline,chartName);
 const W=width,H=W<460?290:318,L=16,R=22,T=18,B=18;
 const xs=stars.map(s=>spread(-s.dBpb)),ys=stars.map(s=>s.dCore);
 const x0=Math.min(0,...xs),x1=Math.max(0,...xs),y0=Math.min(0,...ys),y1=Math.max(0,...ys);
 const lo=x0-(x1-x0)*.06,hi=x1+(x1-x0)*.2,bottom=y0-(y1-y0)*.1,top=y1+(y1-y0)*.22;
 const x=(v:number)=>L+(v-lo)/(hi-lo)*(W-L-R),y=(v:number)=>T+(top-v)/(top-bottom)*(H-T-B);
 const radius=(s:Star)=>3.6+2.6*Math.sqrt(Math.max(s.params,3e8)/1.4e9);
 const placed=stars.map(s=>({s,cx:x(spread(-s.dBpb)),cy:y(s.dCore),r:radius(s)}));
 const origin={x:x(0),y:y(0)};
 const order=[...placed].sort((a,b)=>b.s.dCore-a.s.dCore);
 const labels=placeLabels([{id:'__dense',x:origin.x,y:origin.y,r:9,text:'Dense Baseline'},...order.map(p=>({id:p.s.id,x:p.cx,y:p.cy,r:p.r,text:p.s.name}))],{x:4,y:4,w:W-8,h:H-8},W<460?6.1:6.5);
 const families=[...new Set(placed.map(p=>p.s.family))].sort((a,b)=>FAMILY_ORDER.indexOf(a)-FAMILY_ORDER.indexOf(b));
 const lines=families.map(f=>({f,pts:placed.filter(p=>p.s.family===f).sort((a,b)=>a.cx-b.cx)})).filter(l=>l.pts.length>1);
 if(!stars.length)return <div className="constellation-card empty"><p>The constellation appears once the Dense Baseline and other runs reach the final checkpoint.</p></div>;
 return <div className="constellation-card">
  <div className="constellation-heading"><span>Where every model lands</span><div>{families.map(f=><i key={f} style={{'--tone':tone(f).ink} as CSSProperties}>{tone(f).name}</i>)}</div></div>
  <div className="constellation-plot" ref={wrap} onMouseLeave={()=>setHover(null)}>
   <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`Final validation BPB and CORE of ${stars.length} models relative to the Dense Baseline`}>
    <defs>{[...new Set([...families,'baseline'])].map(f=><radialGradient key={f} id={`${uid}-${f}`}><stop offset="0" stopColor={tone(f).mid} stopOpacity=".75"/><stop offset="1" stopColor={tone(f).mid} stopOpacity="0"/></radialGradient>)}<linearGradient id={uid+'-better'} x1="0" x2="1" y1="1" y2="0"><stop offset="0" stopColor="#F4F1F8" stopOpacity=".2"/><stop offset="1" stopColor="#FBF2F3" stopOpacity=".9"/></linearGradient></defs>
    <rect x={origin.x} y={T} width={Math.max(0,W-R-origin.x)} height={Math.max(0,origin.y-T)} rx="10" fill={`url(#${uid}-better)`}/>
    {W-R-origin.x>96&&<text x={W-R-8} y={T+18} textAnchor="end" className="constellation-better">better on both</text>}
    <line x1={L} x2={W-R} y1={origin.y} y2={origin.y} className="constellation-axis"/>
    <line x1={origin.x} x2={origin.x} y1={T} y2={H-B} className="constellation-axis"/>
    {lines.map(({f,pts})=><path key={f} d={pts.map((p,i)=>`${i?'L':'M'}${p.cx.toFixed(1)},${p.cy.toFixed(1)}`).join(' ')} className="constellation-line" stroke={tone(f).mid} pathLength={1}/>)}
    <g className="constellation-dense"><circle cx={origin.x} cy={origin.y} r="20" fill={`url(#${uid}-baseline)`}/><path d={sparkle(origin.x,origin.y,10)} fill="#526B85"/>{labels.get('__dense')&&<text x={labels.get('__dense')!.x} y={labels.get('__dense')!.y} textAnchor={labels.get('__dense')!.anchor} className="constellation-label dense">Dense Baseline</text>}</g>
    {placed.map(({s,cx,cy,r},i)=>{const t=tone(s.family),label=labels.get(s.id),dim=hover&&hover.id!==s.id;
     return <g key={s.id} className="constellation-star" style={{'--delay':`${i*55}ms`,'--twinkle':`${(i*.37)%3.2}s`,opacity:dim?.35:1} as CSSProperties} onMouseEnter={()=>setHover(s)} onFocus={()=>setHover(s)} onBlur={()=>setHover(null)} onClick={()=>s.method&&onModel(s.method)} tabIndex={0} role="button" aria-label={`${s.name}: val BPB ${signed(s.dBpb,4)}, CORE ${signed(s.dCore,2)} versus Dense`}>
      <circle cx={cx} cy={cy} r={r*2.6} fill={`url(#${uid}-${s.family})`} className="constellation-halo"/>
      <circle cx={cx} cy={cy} r={r} fill={t.ink} stroke="#FFFEFD" strokeWidth="1.4"/>
      {(label||hover?.id===s.id)&&<text x={label?.x??cx+r+5} y={label?.y??cy+4} textAnchor={label?.anchor??'start'} className="constellation-label" fill={t.ink}>{s.name}</text>}
     </g>;})}
   </svg>
   {hover&&<div className="constellation-tip" style={{left:Math.min(Math.max(x(spread(-hover.dBpb)),90),W-90),top:y(hover.dCore)}}><b>{hover.name}</b><span>{tone(hover.family).name}{hover.run.cohort==='native-v3'?' · historical cost estimate':''}</span><dl><div><dt>Val BPB</dt><dd>{signed(hover.dBpb,4)}</dd></div><div><dt>CORE</dt><dd>{signed(hover.dCore,2)}</dd></div></dl></div>}
  </div>
  <p className="constellation-caption"><span>Right: lower validation BPB · Up: higher CORE</span><span>vs Dense at step {FINAL_STEP.toLocaleString('en-US')} · √ scale on BPB</span></p>
 </div>;
}
