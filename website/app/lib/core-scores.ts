import specification from '../../data/core-tasks.json' with {type:'json'};
import type {CoreEvaluation, Run} from './types';

export const CORE_TASKS = specification.tasks;
export type ScoreKind = 'centered' | 'raw';

export function trackingTaskEvaluations(rows:Record<string,any>[],axis:string):CoreEvaluation[]{
 const records=new Map<number,CoreEvaluation>();
 for(const row of rows){
  const step=row[axis];
  if(!Number.isInteger(step)||step<0)continue;
  const raw:Record<string,number>={},centered:Record<string,number>={};
  for(const task of CORE_TASKS){
   const value=row['core/results/'+task.id];
   if(typeof value==='number'&&Number.isFinite(value)&&value>=0&&value<=1){raw[task.id]=value;centered[task.id]=(value-task.randomBaseline)/(1-task.randomBaseline);}
  }
  // A complete task row must reproduce the logged aggregate before publication.
  const aggregate=row['core/core_metric']??row['core/metric'];
  const mean=Object.values(centered).reduce((a,b)=>a+b,0)/CORE_TASKS.length;
  if(Object.keys(centered).length!==CORE_TASKS.length||!Number.isFinite(aggregate)||Math.abs(mean-aggregate)>1e-6)continue;
  records.set(step,{step,raw,centered});
 }
 return [...records.values()].sort((a,b)=>a.step-b.step);
}

export function mergeTaskEvaluations(previous:CoreEvaluation[]=[],incoming:CoreEvaluation[]=[]):CoreEvaluation[]{
 return [...new Map([...previous,...incoming].map(e=>[e.step,e])).values()].sort((a,b)=>a.step-b.step);
}

export function taskEvaluation(run:Run|undefined,step:number|undefined,kind:ScoreKind='centered'){
 if(step===undefined)return undefined;
 const evaluation=run?.taskEvaluations?.find(e=>e.step===step);
 return evaluation?.[kind];
}

export function sharedTaskSteps(runs:(Run|undefined)[],kind:ScoreKind='centered'):number[]{
 if(!runs.length||runs.some(r=>!r))return [];
 const steps=(runs[0]?.taskEvaluations??[]).filter(e=>Object.keys(e[kind]??{}).length>0).map(e=>e.step);
 return [...new Set(steps)].filter(step=>runs.every(r=>Object.keys(taskEvaluation(r,step,kind)??{}).length>0)).sort((a,b)=>a-b);
}

export function sharedTaskStep(runs:(Run|undefined)[],cursor:number,kind:ScoreKind='centered'){
 return sharedTaskSteps(runs,kind).filter(step=>step<=cursor).at(-1);
}
