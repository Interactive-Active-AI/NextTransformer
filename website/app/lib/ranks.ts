import type {Run} from './types';

export type RankPoint={step:number;rank:number;value:number};

// Steps at which ranks are drawn: every CORE event, or validation every 4,000 updates plus the final step.
export function rankSteps(runs:Run[],metric:'core'|'val',final=46875){
 const steps=[...new Set(runs.flatMap(r=>r[metric].map(p=>p.step)))].filter(s=>s>0&&s<=final&&(metric==='core'||s%4000===0||s===final)).sort((a,b)=>a-b);
 return steps.filter(s=>runs.filter(r=>r[metric].some(p=>p.step===s&&Number.isFinite(p.value))).length>=2);
}

// Each run's rank at every step where it has an exact measurement; ties share a rank, as on the leaderboard.
export function rankTrajectories(runs:Run[],metric:'core'|'val',steps:number[]):Map<string,RankPoint[]>{
 const out=new Map<string,RankPoint[]>(runs.map(r=>[r.id,[]]));
 for(const step of steps){
  const scored=runs.flatMap(r=>{const p=r[metric].find(p=>p.step===step&&Number.isFinite(p.value));return p?[{run:r,value:p.value}]:[];})
   .sort((a,b)=>metric==='core'?b.value-a.value:a.value-b.value);
  let rank=0,previous:number|undefined;
  scored.forEach((s,i)=>{if(previous===undefined||Math.abs(s.value-previous)>1e-12)rank=i+1;previous=s.value;out.get(s.run.id)!.push({step,rank,value:s.value});});
 }
 return out;
}
