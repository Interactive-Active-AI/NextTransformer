'use client';
import {useState} from 'react';
import {ArrowUpRight,BookOpen,Box,Eye,RotateCcw,Users} from 'lucide-react';
import {REPO_URL,docs} from './lib/links';
import {TONES} from './lib/palette';
import {scalingLadder,scalingRecords} from './lib/results';
import {LADDER_STATUS,VIEWS,cellStatus,compact,familyKey,plannedPoints,scalingPoints} from './lib/scaling';
import type {ScalingMetric,View} from './lib/scaling';
import type {LadderCell,Method,Run} from './lib/types';
import ScalingPlot from './scaling-plot';
import type {FamilyTone} from './scaling-plot';
import './scaling.css';

const TONE:Record<string,FamilyTone>={...TONES,baseline:{...TONES.baseline,name:'Dense'},moe:{name:'MoE',ink:'#2F7A72',mid:'#8FBDB6',soft:'#EEF6F5'}};
const tones=(family:string)=>TONE[family]??TONE.baseline;
const ORDER=['baseline','moe','looped','sparse','linear','residual','hybrid','others'];
const VIEW_LABELS:[string,string][]=[['angled','3D'],['parameters','Parameters'],['tokens','Tokens'],['floor','Floor plan']];
const TIER_NOTES:Record<string,string>={T3:'Five sizes per family map the trade-off between parameters and tokens.',T4:'The budget of the 49B comparison protocol, with five sizes per family.',C200:'Set by the Dense 200B-token reference run; predicted optima are trained here.'};
const TASKS=[['Train a ladder cell','Claim an open cell and train it under the frozen recipe.','About 10²⁰ training FLOPs at T3'],['Add an architecture family','A method with a size rule; its cells join the ladder.','Configurations and cost proofs'],['Register a forecast','Predict the C200 optimum before the test runs start.','No training compute'],['Evaluate released checkpoints','Long-context and downstream evaluations on public weights.','Inference only'],['Improve the fits','Estimators and uncertainty for the compute-optimal frontier.','A laptop is enough'],['Reproduce a cell','Retrain a recorded cell with another seed, or re-run its checks.','One cell, or none']];
const STEPS=[['01','Claim','Open the claim issue for one cell; a maintainer marks it claimed on this board.'],['02','Prepare','Pinned ClimbMix revision, tokenizer and CORE bundle, and the frozen scaling recipe.'],['03','Configure','Write the cell configuration and check its training FLOPs against the tier budget.'],['04','Train','Train with the scaling trainer, on any hardware; resume as often as needed.'],['05','Export','nexttransformer.scaling.export checks the run and writes its record.'],['06','Pull request','The record, the checkpoint link and the W&B run. The board and the plot update on merge.']];
const sci=(v:number)=>{const e=Math.floor(Math.log10(v));return <>{(v/10**e).toFixed(1)} × 10<sup>{e}</sup></>;};

function Cell({cell,status}:{cell:LadderCell;status:string}){
 const moe=cell.family==='moe';
 return <li className={'ladder-cell '+status} title={LADDER_STATUS[status]}>
  <b>{cell.layers?`L${cell.layers}`:'Optimum'}</b>
  <span>{cell.parameters?<>{compact(moe?cell.activeBlockParameters!:cell.blockParameters!)}{moe?' active':''} · {compact(cell.tokens!)} tokens</>:'Size from the registered forecast'}</span>
  <em>{cell.claim?<a href={cell.claim.issue} target="_blank" rel="noreferrer">{cell.claim.by}</a>:LADDER_STATUS[status]}</em>
 </li>;
}

export default function Scaling({runs,methods}:{runs:Run[];methods:Method[]}){
 const [metric,setMetric]=useState<ScalingMetric>('core'),[view,setView]=useState<View>(VIEWS.angled),[planned,setPlanned]=useState(true),[hidden,setHidden]=useState<string[]>([]);
 const all=scalingPoints(runs,methods,scalingRecords),rings=plannedPoints(scalingLadder,scalingRecords);
 const families=ORDER.filter(f=>all.some(p=>p.family===f)||rings.some(p=>p.family===f));
 const points=all.filter(p=>!hidden.includes(p.family)),floor=rings.filter(p=>!hidden.includes(p.family));
 const ladder=scalingLadder,cells=ladder?.cells??[];
 const toggle=(f:string)=>setHidden(h=>h.includes(f)?h.filter(x=>x!==f):[...h,f]);
 const preset=Object.entries(VIEWS).find(([,v])=>v.yaw===view.yaw&&v.pitch===view.pitch)?.[0];
 return <section className="scaling" id="scaling" aria-label="Scaling">
  <div className="scaling-hero"><div><p className="eyebrow">04 / SCALING</p><h2>One budget, spent three ways.<span>Parameters, training tokens and architecture, chosen together for the best CORE and validation BPB.</span></h2>
   <p className="scaling-lede">The leaderboard fixes the compute and the model size. The scaling stage fixes only the compute: at each budget, every architecture family may trade parameters against training tokens. Two small budgets locate each family's optimum; the largest tests a forecast registered before its runs start.</p>
   <div className="scaling-actions"><a className="primary" href={docs('scaling/overview/')}><BookOpen size={16}/> Scaling protocol</a><a className="secondary" href={docs('scaling/contribute/')}><Users size={16}/> Take part</a><a className="scaling-text-link" href={REPO_URL+'/tree/main/results/scaling'} target="_blank" rel="noreferrer">Records <ArrowUpRight size={14}/></a></div></div>
   <div className="scaling-tiers">{(ladder?.tiers??[]).map(t=>{const own=cells.filter(c=>c.tier===t.id),done=own.filter(c=>cellStatus(c,scalingRecords)==='recorded').length;return <div key={t.id} className={t.role}><span>{t.id} · {t.role==='fit'?'FIT':'TEST'}</span><b>{sci(t.flops)} FLOPs</b><p>{TIER_NOTES[t.id]}</p><small>{own.length} cells · {done} recorded</small></div>;})}</div>
  </div>
  <div className="scaling-surface">
   <div className="results-toolbar scaling-toolbar"><div className="metric-switch" role="group" aria-label="Height"><button aria-pressed={metric==='core'} onClick={()=>setMetric('core')}>CORE <span>↑</span></button><button aria-pressed={metric==='bpb'} onClick={()=>setMetric('bpb')}>Validation BPB <span>↓</span></button></div>
    <div className="chart-mode" role="group" aria-label="Viewpoint">{VIEW_LABELS.map(([id,label])=><button key={id} className={preset===id?'active':''} aria-pressed={preset===id} onClick={()=>setView(VIEWS[id])}>{id==='angled'?<Box size={13}/>:<Eye size={13}/>}{label}</button>)}<button onClick={()=>setView(VIEWS.angled)} aria-label="Reset view"><RotateCcw size={13}/></button></div></div>
   <div className="scaling-legend">{families.map(f=><button key={f} aria-pressed={!hidden.includes(f)} onClick={()=>toggle(f)}><i style={{background:tones(f).ink}}/>{tones(f).name}</button>)}<span className="scaling-shapes"><i className="dot"/>49B protocol<i className="diamond"/>Scaling stage<label className="checkbox"><input type="checkbox" checked={planned} onChange={e=>setPlanned(e.target.checked)}/>Planned runs</label></span></div>
   <ScalingPlot points={points} planned={floor} metric={metric} view={view} onView={setView} onReset={()=>setView(VIEWS.angled)} showPlanned={planned} tones={tones}/>
   <div className="scaling-caption"><span>Each point is a finished run: total parameters with embeddings, training tokens, and final {metric==='core'?'CORE':'validation BPB'} as height; colour is the architecture family. Circles follow the 49B protocol at a 2,048-token context and diamonds the scaling stage at 4,096, so compare BPB within one shape. Hollow rings on the floor are planned cells.</span><span>Drag or use the arrow keys to rotate · double-click to reset</span></div>
  </div>
  {ladder&&<div className="scaling-board"><div className="scaling-board-heading"><div><h3>The ladder</h3><span>{cells.length} cells · {ladder.protocol} · token counts from the theoretical training FLOPs of each size</span></div><a href={docs('scaling/contribute/')}>How to claim a cell <ArrowUpRight size={14}/></a></div>
   <div className="ladder-grid" style={{'--tiers':ladder.tiers.length} as any}><span/>{ladder.tiers.map(t=><span key={t.id} className="ladder-tier">{t.id}<small>{sci(t.flops)} FLOPs</small></span>)}
    {ladder.families.map(f=><div key={f.id} className="ladder-row" style={{'--family':tones(familyKey(f.id)).ink} as any}><div className="ladder-family"><b><i/>{f.name}</b><p>{f.rule}</p>{f.gate&&<small>{f.gate}</small>}</div>{ladder.tiers.map(t=><ul key={t.id} aria-label={`${f.name} at ${t.id}`}>{cells.filter(c=>c.family===f.id&&c.tier===t.id).map(c=><Cell key={c.id} cell={c} status={cellStatus(c,scalingRecords)}/>)}</ul>)}</div>)}
   </div></div>}
  <div className="scaling-lower">
   <div className="scaling-panel"><h3>What the community can build</h3><p>Every contribution is a file in the repository, reviewed and merged like any result.</p><ul className="scaling-tasks">{TASKS.map(([title,text,cost])=><li key={title}><b>{title}</b><span>{text}</span><small>{cost}</small></li>)}</ul></div>
   <div className="scaling-panel"><h3>How to take part</h3><p>Training a cell, step by step. The <a href={docs('scaling/contribute/')}>contribution guide</a> has the commands and the other paths.</p><ol className="scaling-steps">{STEPS.map(([n,title,text])=><li key={n}><span>{n}</span><div><b>{title}</b><p>{text}</p></div></li>)}</ol></div>
  </div>
 </section>;
}
