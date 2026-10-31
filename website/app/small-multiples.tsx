'use client';
import {useEffect,useState} from 'react';
import {ArrowUpRight} from 'lucide-react';
import Chart,{chartName,seriesColor} from './learning-chart';
import {sharedChartDomain} from './lib/chart-series';
import Pagination from './pagination';
import type {Method,Run} from './lib/types';
export default function SmallMultiples({runs,methods,baseline,metric,cursor,max,allRange,onFocus}:{runs:Run[];methods:Method[];baseline?:Run;metric:'val'|'core';cursor:number;max:number;allRange:boolean;onFocus:(run:Run)=>void}){
 const [page,setPage]=useState(0),choices=runs.filter(r=>r.methodId!=='dense_full'),size=6;
 const key=choices.map(r=>r.id).join('|');
 useEffect(()=>setPage(0),[key]);
 const activePage=Math.min(page,Math.max(0,Math.ceil(choices.length/size)-1));
 const domain=sharedChartDomain(runs,metric,allRange);
 return <div className="small-multiples"><div className="multiples-note"><span><i/>{baseline?'Each method against Dense Baseline · the shaded gap is the difference':'Independent trajectories · full Dense pending'}</span><span>Shared axes · measured checkpoints</span></div><div className="multiples-grid">{choices.slice(activePage*size,(activePage+1)*size).map(run=>{
  const method=methods.find(m=>m.id===run.methodId),name=chartName(method)||run.methodId,last=run[metric].filter(p=>p.step<=cursor).at(-1),end=run[metric].at(-1)?.step??max;
  const densePoint=baseline?.[metric].find(p=>p.step===last?.step),gap=last&&densePoint?(last.value-densePoint.value)*(metric==='core'?100:1):undefined;
  const digits=metric==='val'?4:2,even=gap!==undefined&&Math.abs(gap)<.5*10**-digits;
  const pair=[baseline&&{...baseline,[metric]:baseline[metric].filter(p=>p.step<=end)},run].filter((r):r is Run=>!!r);
  return <article className="multiples-panel" key={run.id}><button className="multiples-heading" onClick={()=>onFocus(run)} aria-label={'Focus on '+name}><span><i style={{background:seriesColor(run.methodId,methods)}}/>{name}<ArrowUpRight size={14}/></span><b>{last?((metric==='core'?100:1)*last.value).toFixed(digits):'—'}</b></button><Chart runs={pair} methods={methods} metric={metric} cursor={cursor} max={max} allRange={allRange} small gapFill domain={domain} ariaLabel={name+(baseline?' and Dense Baseline, shared ':' · shared ')+(metric==='val'?'validation BPB':'CORE')+' scale'}/><div className="multiples-caption">{gap===undefined?<span>{last?'Step '+last.step.toLocaleString('en-US'):'No evaluation yet'} · {metric==='val'?'BPB ↓':'CORE ×100 ↑'}</span>:<><span>Δ vs Dense</span><b>{(even?'±':gap>0?'+':'−')+Math.abs(gap).toFixed(digits)}</b></>}</div></article>;
 })}</div>{!choices.length&&<p className="multiples-empty">No measured peer experiments for this family yet. Dense Baseline remains available in the focused view.</p>}<Pagination page={activePage} total={choices.length} size={size} onChange={setPage} label="Experiments"/></div>;
}
