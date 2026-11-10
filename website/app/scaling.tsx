'use client';
import {useEffect,useMemo,useRef,useState} from 'react';
import {ArrowUpRight,BookOpen,RotateCcw,Users} from 'lucide-react';
import {REPO_URL,docs} from './lib/links';
import {TONES} from './lib/palette';
import {scalingLadder,scalingRecords} from './lib/results';
import {CHINCHILLA,LADDER_STATUS,VIEWS,cellStatus,compact,plannedPoints,scalingPoints,scalingSurfaces} from './lib/scaling';
import type {ScalingMetric,View} from './lib/scaling';
import type {LadderCell,Method,Run} from './lib/types';
import ScalingPlot from './scaling-plot';
import type {Budget,FamilyTone} from './scaling-plot';
import './scaling.css';

const TONE:Record<string,FamilyTone>={...TONES,baseline:{...TONES.baseline,name:'Dense'},moe:{name:'MoE',ink:'#2F7A72',mid:'#8FBDB6',soft:'#EEF6F5'}};
const tones=(family:string)=>TONE[family]??TONE.baseline;
const ORDER=['baseline','moe','looped','sparse','linear','residual','hybrid','others'];
const BUDGET_COLORS:Record<string,string>={T3:'#2F7A72',T4:'#4A6E9A',C200:'#965C66'};
const GROUP:Record<string,string>={open:'open',claimed:'active',running:'active',submitted:'active',recorded:'done',trained:'done',gated:'waiting',forecast:'waiting'};
const STEPS=[['Claim a cell','Pick an open cell on the ladder and open a claim issue.'],['Train and export','Use the frozen recipe at the cell\'s token count; nexttransformer.scaling.export writes the record.'],['Open a pull request','The record, the configuration and the checkpoint link. The plot updates on merge.']];
const MORE=[['Register a forecast','register-a-forecast'],['Evaluate checkpoints','evaluate-released-checkpoints'],['Improve the fits','improve-the-fits'],['Reproduce a cell','reproduce-a-cell'],['Add a family','add-an-architecture-family']];
const sci=(v:number)=>{const e=Math.floor(Math.log10(v));return <>{(v/10**e).toFixed(1)} × 10<sup>{e}</sup></>;};
const describe=(c:LadderCell,status:string)=>`${c.layers?`L${c.layers}`:'Predicted optimum'}${c.parameters?` · ${compact(c.parameters)} parameters · ${compact(c.tokens!)} tokens`:''} · ${LADDER_STATUS[status]??status}`;

export default function Scaling({runs,methods}:{runs:Run[];methods:Method[]}){
 const [metric,setMetric]=useState<ScalingMetric>('core'),[view,setView]=useState<View>(VIEWS.angled),[planned,setPlanned]=useState(true),[hidden,setHidden]=useState<string[]>([]);
 const interacted=useRef(false);
 // A slow turn into the default view on arrival; any interaction stops it.
 useEffect(()=>{
  if(window.matchMedia('(prefers-reduced-motion: reduce)').matches)return;
  const start=performance.now(),from={yaw:VIEWS.angled.yaw-.9,pitch:.16};let id=0;
  const tick=(now:number)=>{if(interacted.current)return;const t=Math.min(1,(now-start)/1700),e=1-(1-t)**3;
   if(t>=1){setView(VIEWS.angled);return;}setView({yaw:from.yaw+(VIEWS.angled.yaw-from.yaw)*e,pitch:from.pitch+(VIEWS.angled.pitch-from.pitch)*e});id=requestAnimationFrame(tick);};
  id=requestAnimationFrame(tick);
  const settle=setTimeout(()=>{if(!interacted.current)setView(VIEWS.angled);},1900);
  return()=>{cancelAnimationFrame(id);clearTimeout(settle);};
 },[]);
 const reset=()=>{interacted.current=true;setView(VIEWS.angled);};
 const all=useMemo(()=>scalingPoints(runs,methods,scalingRecords),[runs,methods]);
 const surfaces=useMemo(()=>scalingSurfaces(all,metric),[all,metric]);
 const rings=plannedPoints(scalingLadder,scalingRecords);
 const families=ORDER.filter(f=>all.some(p=>p.family===f)||rings.some(p=>p.family===f));
 const points=all.filter(p=>!hidden.includes(p.family)),floor=planned?rings.filter(p=>!hidden.includes(p.family)):[];
 const ladder=scalingLadder,cells=ladder?.cells??[],tiers=ladder?.tiers??[];
 const budgets:Budget[]=tiers.map(t=>({id:t.id,flops:t.flops,color:BUDGET_COLORS[t.id]??'#7F9DBA',label:t.id}));
 const sign=metric==='core'?'−':'+',family=surfaces[0]?.kind==='family';
 return <section className="scaling" id="scaling" aria-label="Scaling">
  <header className="scaling-head">
   <div><p className="eyebrow">04 / SCALING</p><h2>One budget, three choices.<span>Parameters, training tokens and architecture, chosen together for the best CORE and validation BPB.</span></h2></div>
   <div className="scaling-actions"><a className="primary" href={docs('scaling/overview/')}><BookOpen size={16}/> Protocol</a><a className="secondary" href={docs('scaling/contribute/')}><Users size={16}/> Take part</a></div>
  </header>
  <ol className="scaling-budgets" aria-label="Training budgets">{tiers.map(t=>{const own=cells.filter(c=>c.tier===t.id),done=own.filter(c=>cellStatus(c,scalingRecords)==='recorded').length;
   return <li key={t.id}><i style={{background:BUDGET_COLORS[t.id]}}/><b>{t.id}</b><span>{sci(t.flops)} FLOPs</span><small>{t.role==='fit'?'Fit':'Test'} · {done} of {own.length} recorded</small></li>;})}</ol>
  <div className="scaling-card">
   <div className="scaling-toolbar">
    <div className="metric-switch" role="group" aria-label="Height"><button aria-pressed={metric==='core'} onClick={()=>setMetric('core')}>CORE <span>↑</span></button><button aria-pressed={metric==='bpb'} onClick={()=>setMetric('bpb')}>Validation BPB <span>↓</span></button></div>
    <div className="scaling-legend" role="group" aria-label="Families">{families.map(f=><button key={f} aria-pressed={!hidden.includes(f)} onClick={()=>setHidden(h=>h.includes(f)?h.filter(x=>x!==f):[...h,f])}><i style={{background:tones(f).ink}}/>{tones(f).name}</button>)}</div>
    <div className="scaling-tools"><label className="checkbox"><input type="checkbox" checked={planned} onChange={e=>setPlanned(e.target.checked)}/>Planned</label><button className="icon" onClick={reset} aria-label="Reset view" title="Reset view"><RotateCcw size={14}/></button></div>
   </div>
   <ScalingPlot points={points} planned={floor} metric={metric} surfaces={surfaces} budgets={budgets} view={view} onView={setView} onReset={reset} onInteract={()=>{interacted.current=true;}} tones={tones}/>
   <p className="scaling-note"><span>{family?<>Surfaces fitted per family to the scaling ladder: value = a {sign} (b·N<sup>−α</sup> + c·D<sup>−β</sup>).</>:<><b>Provisional surface.</b> value = a {sign} (b·N<sup>−{CHINCHILLA.alpha}</sup> + c·D<sup>−{CHINCHILLA.beta}</sup>), fitted to every run shown with Chinchilla's exponents. Fits per family replace it once the ladder has runs at two budgets.</>}</span>
    <span className="scaling-key"><i className="dot"/>49B protocol<i className="diamond"/>Scaling stage<i className="ring"/>Planned · drag to rotate</span></p>
  </div>
  <div className="scaling-lower">
   <section className="scaling-panel" aria-label="The ladder"><header><h3>The ladder</h3><a href={REPO_URL+'/blob/main/results/scaling/ladder.json'} target="_blank" rel="noreferrer">All cells <ArrowUpRight size={14}/></a></header>
    <table className="ladder"><thead><tr><th/>{tiers.map(t=><th key={t.id} style={{color:BUDGET_COLORS[t.id]}}>{t.id}</th>)}</tr></thead>
     <tbody>{(ladder?.families??[]).map(f=><tr key={f.id}><th><i style={{background:tones(f.id==='dense'?'baseline':f.id).ink}}/>{f.name}{f.status==='gated'&&<small>Recipe gate</small>}</th>
      {tiers.map(t=>{const own=cells.filter(c=>c.family===f.id&&c.tier===t.id),done=own.filter(c=>cellStatus(c,scalingRecords)==='recorded').length;
       return <td key={t.id}><span className="ladder-dots">{own.map(c=>{const status=cellStatus(c,scalingRecords);return <i key={c.id} className={GROUP[status]} title={describe(c,status)}/>;})}</span><small>{done}/{own.length}</small></td>;})}</tr>)}</tbody></table>
    <p className="ladder-key"><span><i className="open"/>Open</span><span><i className="active"/>Claimed or training</span><span><i className="done"/>Trained</span><span><i className="waiting"/>Waiting</span></p>
   </section>
   <section className="scaling-panel" aria-label="Take part"><header><h3>Take part</h3><a href={docs('scaling/contribute/')}>Guide <ArrowUpRight size={14}/></a></header>
    <ol className="scaling-steps">{STEPS.map(([title,text],i)=><li key={title}><span>{i+1}</span><div><b>{i===0?<a href={REPO_URL+'/issues/new?template=scaling_cell.yml'} target="_blank" rel="noreferrer">{title}</a>:title}</b><p>{text}</p></div></li>)}</ol>
    <p className="scaling-more">{MORE.map(([label,anchor])=><a key={anchor} href={docs('scaling/contribute/')+'#'+anchor}>{label}</a>)}</p>
   </section>
  </div>
 </section>;
}
