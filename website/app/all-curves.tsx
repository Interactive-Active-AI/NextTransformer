'use client';
import {useState} from 'react';
import type {Method,Run} from './lib/types';
import Chart,{chartName,seriesColor} from './learning-chart';

export default function AllCurves({runs,methods,metric,cursor,max,allRange}:{runs:Run[];methods:Method[];metric:'val'|'core';cursor:number;max:number;allRange:boolean}){
 const [pinned,setPinned]=useState(''),[preview,setPreview]=useState('');
 const requestedFocus=preview||pinned;
 const focus=runs.some(run=>run.methodId===requestedFocus)?requestedFocus:'';
 return <div className="all-curves">
  <div className="all-curves-heading"><span>{runs.length} curves on shared axes</span><span>{metric==='val'?'Bits / byte · lower is better':'CORE ×100 · higher is better'}</span></div>
  <div className="all-curves-legend" role="group" aria-label="Highlight a curve" onMouseLeave={()=>setPreview('')}>
   {runs.map(run=><button key={run.id} aria-pressed={pinned===run.methodId} onMouseEnter={()=>setPreview(run.methodId)} onClick={()=>setPinned(p=>p===run.methodId?'':run.methodId)}><i className={run.methodId==='dense_full'?'dense':''} style={{backgroundColor:seriesColor(run.methodId,methods)}}/>{chartName(methods.find(m=>m.id===run.methodId))}</button>)}
   {pinned&&<button className="clear-curve-focus" onMouseEnter={()=>setPreview('')} onClick={()=>setPinned('')}>Clear focus</button>}
  </div>
  <Chart runs={runs} methods={methods} metric={metric} cursor={cursor} max={max} allRange={allRange} wide compact overview focus={focus}/>
  <div className="plot-caption"><span>Training step</span><span>Hover a curve or select a model to focus</span></div>
 </div>;
}
