'use client';
import {useId,useState} from 'react';
import type {ReactNode} from 'react';
import {ChevronDown,ChevronRight,ArrowRight} from 'lucide-react';
import Chart,{chartName,seriesColor} from './learning-chart';
import {CORE_TASKS,sharedTaskSteps,taskEvaluation} from './lib/core-scores';
import type {ScoreKind} from './lib/core-scores';
import type {Method,Run} from './lib/types';
import './core-breakdown.css';

const format=(n:number)=>new Intl.NumberFormat('en-US').format(n);
const score=(v:number|undefined)=>v===undefined?'—':(v*100).toFixed(2);
const difference=(v:number|undefined)=>v===undefined?'—':(v>0?'+':'')+(v*100).toFixed(2);

function BenchmarkSelect({label,ariaLabel,value,onChange,children}:{label:string;ariaLabel:string;value:string;onChange:(value:string)=>void;children:ReactNode}){
 return <label className="benchmark-field"><span>{label}</span><span className="benchmark-select"><select aria-label={ariaLabel} value={value} onChange={e=>onChange(e.target.value)}>{children}</select><ChevronDown size={16} strokeWidth={1.7} aria-hidden="true"/></span></label>;
}

export default function CoreBreakdown({runs,methods,baseline,focusedId,cursor,onFocus,onShowAll,embedded=false,initialView='pair'}:{runs:Run[];methods:Method[];baseline?:Run;focusedId?:string;cursor:number;onFocus:(id:string)=>void;onShowAll:()=>void;embedded?:boolean;initialView?:'pair'|'matrix'}){
 const uid=useId();
 const [open,setOpen]=useState(embedded),[view,setView]=useState<'pair'|'matrix'>(initialView),[kind,setKind]=useState<ScoreKind>('centered'),[checkpoint,setCheckpoint]=useState('follow'),[expanded,setExpanded]=useState(''),[order,setOrder]=useState('benchmark'),[referenceId,setReferenceId]=useState('');
 const reference=baseline??runs.find(r=>r.id===referenceId);
 const choices=runs.filter(r=>r.id!==reference?.id);
 const focused=choices.find(r=>r.id===focusedId)??choices[0];
 const eligible=runs.filter(r=>r.id!==reference?.id&&r.taskEvaluations?.some(e=>Object.keys(e[kind]??{}).length));
 const matrixRuns=[reference,...eligible].filter((r):r is Run=>!!r);
 const comparison=(view==='pair'?[reference,focused]:matrixRuns).filter((r):r is Run=>!!r);
 const steps=sharedTaskSteps(comparison,kind);
 const activeCheckpoint=checkpoint==='follow'||steps.includes(Number(checkpoint))?checkpoint:'follow';
 const step=activeCheckpoint==='follow'?steps.filter(s=>s<=cursor).at(-1):steps.find(s=>String(s)===activeCheckpoint);
 const denseScores=taskEvaluation(reference,step,kind),ownScores=taskEvaluation(focused,step,kind);
 const getName=(r:Run|undefined)=>r?chartName(methods.find(m=>m.id===r.methodId))||r.methodId:'Method';
 const color=(r:Run|undefined)=>seriesColor(r?.methodId??'dense_full',methods);
 const delta=(id:string)=>ownScores?.[id]!==undefined&&denseScores?.[id]!==undefined?ownScores[id]-denseScores[id]:undefined;
 const tasks=[...CORE_TASKS].sort((a,b)=>order==='gap'?Math.abs(delta(b.id)??0)-Math.abs(delta(a.id)??0):0);
 // A fixed domain across this pair's history keeps replay changes comparable.
 const allValues=[reference,focused].flatMap(r=>(r?.taskEvaluations??[]).flatMap(e=>Object.values(e[kind]??{})));
 const min=kind==='raw'?0:Math.min(0,...allValues)-.035,max=kind==='raw'?1:Math.max(.1,...allValues)+.035;
 const x=(v:number)=>(v-min)/(max-min)*100;
 const matrixGaps=CORE_TASKS.flatMap(t=>eligible.map(r=>{const value=taskEvaluation(r,step,kind)?.[t.id],dense=denseScores?.[t.id];return value!==undefined&&dense!==undefined?value-dense:0;}));
 const magnitude=Math.max(.01,...(reference?matrixGaps:matrixRuns.flatMap(r=>Object.values(taskEvaluation(r,step,kind)??{}))).map(Math.abs));
 const missing=runs.filter(r=>r.id!==reference?.id&&!eligible.some(e=>e.id===r.id));
 function resetComparison(id:string){onFocus(id);setExpanded('');}
 function taskChart(taskId:string,selectedRun:Run|undefined=focused){
  if(!selectedRun)return null;
  const plotted=[reference,selectedRun].filter((r):r is Run=>!!r).map(r=>({...r,core:(r.taskEvaluations??[]).filter(e=>e[kind]?.[taskId]!==undefined).map(e=>({step:e.step,value:e[kind]![taskId]}))}));
  return <div className="benchmark-detail"><div className="benchmark-detail-heading"><div><span>BENCHMARK TRAJECTORY</span><h4>{CORE_TASKS.find(t=>t.id===taskId)?.label}</h4></div><span>{kind==='centered'?'Centered score':'Raw accuracy'} ×100</span></div><div className="benchmark-series-key">{reference&&<span><i className="dense"/>{getName(reference)}</span>}<span><i style={{background:color(selectedRun)}}/>{getName(selectedRun)}</span></div><Chart runs={plotted} methods={methods} metric="core" cursor={step??cursor} max={Math.max(1,...plotted.flatMap(r=>r.core.map(p=>p.step)))} wide allowNegative={kind==='centered'} ariaLabel={`${CORE_TASKS.find(t=>t.id===taskId)?.label} measured benchmark scores across training checkpoints`}/><p>Measured evaluation checkpoints only · lines are visual guides.</p></div>;
 }
 const noRecords=!choices.some(r=>r.taskEvaluations?.length);
 return <section className="core-breakdown" aria-label="CORE benchmark breakdown">
  {!embedded&&<button className="benchmark-toggle" aria-expanded={open} aria-controls={uid} onClick={()=>setOpen(!open)}><span><b>Benchmark breakdown</b><small>Explore individual scores and choose a comparison</small></span><span className="benchmark-toggle-action">{open?'Collapse':'Explore 22 benchmarks'}<ChevronDown size={19} style={{transform:open?'rotate(180deg)':undefined}}/></span></button>}
  {open&&<div id={uid} className="benchmark-content">
   <div className="benchmark-toolbar"><div className="benchmark-mode" role="group" aria-label="Benchmark comparison view">{[['pair','Paired dots'],['matrix','Model heatmap']].map(([id,label])=><button key={id} onClick={()=>{setView(id as 'pair'|'matrix');setExpanded('');}} aria-pressed={view===id}>{label}</button>)}</div><BenchmarkSelect label="Score" ariaLabel="Benchmark scoring" value={kind} onChange={value=>setKind(value as ScoreKind)}><option value="centered">Centered score</option><option value="raw">Raw accuracy</option></BenchmarkSelect><BenchmarkSelect label="Checkpoint" ariaLabel="Benchmark checkpoint" value={activeCheckpoint} onChange={setCheckpoint}><option value="follow">Follow replay</option>{steps.map(s=><option key={s} value={s}>Step {format(s)}</option>)}</BenchmarkSelect></div>
   {!baseline&&<div className="benchmark-pair-controls"><BenchmarkSelect label="Reference model" ariaLabel="Benchmark reference model" value={referenceId} onChange={value=>{setReferenceId(value);setCheckpoint('follow');setExpanded('');}}><option value="">None · full Dense pending</option>{runs.map(r=><option key={r.id} value={r.id}>{getName(r)}{r.methodId==='sssl'?' · historical':''}</option>)}</BenchmarkSelect><span className="baseline-note">Choose a recorded model for an explicit comparison.</span></div>}
   {view==='pair'&&<div className="benchmark-pair-controls"><BenchmarkSelect label="Compare a method" ariaLabel="Benchmark method" value={focused?.id??''} onChange={resetComparison}>{choices.map(r=><option key={r.id} value={r.id}>{getName(r)}{runs.filter(x=>x.methodId===r.methodId).length>1?' · '+r.name:''}</option>)}</BenchmarkSelect><BenchmarkSelect label="Order" ariaLabel="Benchmark order" value={order} onChange={setOrder}><option value="benchmark">Benchmark order</option><option value="gap">Absolute gap</option></BenchmarkSelect></div>}
   <div className="benchmark-context" role="status"><span>{step!==undefined?<>Shared evaluation · <b>step {format(step)}</b></>:steps.length?'Waiting for the first shared evaluation in this replay.':'No shared task-level checkpoint is available for this selection.'}</span><span>{kind==='centered'?'Random-baseline adjusted · may be negative':'Original accuracy'} · higher is better</span></div>
   {step===undefined?<div className="benchmark-empty"><p>{noRecords?'Task-level results have not been imported for this selection yet.':'Choose a recorded checkpoint or advance the training replay to view scores.'}</p>{noRecords&&<button onClick={onShowAll}>Explore all experiments <ArrowRight size={16}/></button>}</div>:view==='pair'?<>
    <div className="benchmark-column-head"><span>BENCHMARK</span><span className="benchmark-scale">{[0,1,2,3,4].map(i=><i key={i} style={{left:(i*25)+'%'}}>{((min+(max-min)*i/4)*100).toFixed(0)}</i>)}</span><span>{reference?'Δ / '+getName(reference):'SCORE'}</span></div>
    <div className="benchmark-pair-list">{tasks.map(task=>{const dense=denseScores?.[task.id],value=ownScores?.[task.id],gap=delta(task.id),isOpen=expanded===task.id;return <div key={task.id} className="benchmark-row-wrap"><button className="benchmark-pair-row" aria-expanded={isOpen} onClick={()=>setExpanded(isOpen?'':task.id)} aria-label={`${task.label}: ${getName(reference)} ${score(dense)}, ${getName(focused)} ${score(value)}, difference ${difference(gap)}. ${isOpen?'Collapse':'Expand'} training curve.`}><span className="benchmark-label"><ChevronRight size={14} style={{transform:isOpen?'rotate(90deg)':undefined}}/><span>{task.label}<small>{reference?score(dense)+' → ':''}{score(value)}</small></span></span><span className="benchmark-rail" aria-hidden="true">{dense!==undefined&&value!==undefined&&<i className="benchmark-connector" style={{left:Math.min(x(dense),x(value))+'%',width:Math.abs(x(value)-x(dense))+'%'}}/>}{dense!==undefined&&<i className="benchmark-point dense" style={{left:x(dense)+'%'}}/>}{value!==undefined&&<i className="benchmark-point method" style={{left:x(value)+'%',background:color(focused)}}/>}</span><b className="benchmark-delta">{reference?difference(gap):score(value)}</b></button>{isOpen&&taskChart(task.id)}</div>;})}</div>
    <div className="benchmark-series-key">{reference&&<span><i className="dense"/>{getName(reference)}</span>}<span><i style={{background:color(focused)}}/>{getName(focused)}</span><span>{kind==='centered'?'Centered score':'Raw accuracy'} ×100 · select a row to explore its training curve</span></div>
   </>:<><p className="benchmark-scroll-hint">Scroll horizontally to compare models →</p><div className="benchmark-matrix-scroll"><table className="benchmark-matrix"><caption>Individual benchmark scores at step {format(step)}. {reference?'Differences use '+getName(reference)+' at exactly the same step.':'Absolute scores; full Dense results are pending.'}</caption><thead><tr><th>BENCHMARK</th>{matrixRuns.map(r=><th key={r.id}>{getName(r)}<small>{!reference||r.id===reference.id?'Score ×100':'Δ / '+getName(reference)}</small></th>)}</tr></thead><tbody>{CORE_TASKS.map(task=><tr key={task.id}><th scope="row">{task.label}</th>{matrixRuns.map(r=>{const value=taskEvaluation(r,step,kind)?.[task.id],dense=denseScores?.[task.id],gap=value!==undefined&&dense!==undefined?value-dense:undefined,tint=reference?gap:value;return <td key={r.id}><button disabled={value===undefined||r.id===reference?.id} style={{background:r.id===reference?.id?'#F0F4F8':tint===undefined?'transparent':`rgba(${tint>=0?'138,108,178':'202,131,148'},${.06+Math.abs(tint)/magnitude*.32})`}} onClick={()=>{resetComparison(r.id);setView('pair');setExpanded(task.id);setCheckpoint(String(step));}} aria-label={`${task.label}, ${getName(r)}: ${score(value)}; ${reference?'difference from '+getName(reference)+' '+difference(gap):'absolute score'}. Explore this benchmark.`}><b>{!reference||r.id===reference.id?score(value):difference(gap)}</b>{reference&&r.id!==reference.id&&value!==undefined&&<small>score {score(value)}</small>}</button></td>;})}</tr>)}</tbody></table></div><div className="benchmark-heatmap-key"><span><i className="above"/>{reference?'Above '+getName(reference):'Positive score'}</span><span><i className="below"/>{reference?'Below '+getName(reference):'Negative score'}</span><span>{reference?'Colour shows the difference':'Colour shows the absolute score'} · click a result to explore</span></div>{missing.length>0&&<p className="benchmark-coverage">No task-level records yet: {missing.map(getName).join(', ')}.</p>}</>}
   <p className="benchmark-footnote">Centered score = (accuracy − random baseline) / (1 − random baseline). CORE is the mean of the 22 centered scores. Missing measurements stay empty; comparisons use exact shared checkpoints.</p>
  </div>}
 </section>;
}
