import specification from '../../data/core-tasks.json' with {type:'json'};
import type {Run} from './types';

export const DENSE_BASELINE_ID = 'dense_full';
export const SSSL_ID = 'sssl';
const FINAL_STEP=46875,INPUT_TOKENS=49152000000,COMPUTE_TARGET=4.67239832544785;
const VAL_STEPS=[...Array.from({length:188},(_,i)=>i*250),FINAL_STEP];
const CORE_STEPS=[8000,16000,24000,32000,40000,FINAL_STEP];

/** Display migration only. Never rewrite stored run IDs, checkpoints or cost proofs. */
export function baselineIdentity(run:Run):Run {
 return run.methodId==='dense'?{...run,methodId:SSSL_ID,originalMethodId:run.originalMethodId??'dense'}:run;
}

/** A maintainer-reviewed acceptance record is necessary, never inferred from a score. */
export function acceptedFullDense(run:Run):boolean {
 const a=run.baselineAcceptance,c=run.trainingCompute,p=run.proof;
 if(run.methodId!==DENSE_BASELINE_ID||!a||a.status!=='accepted'||a.planVersion!=='v2')return false;
 if(!a.protocolHash||a.protocolHash!==p?.protocol_hash||!/^[a-f0-9]{40}$/.test(a.sourceCommit)||a.sourceCommit!==p?.src_head||!/^https:\/\//.test(a.reportUrl))return false;
 if(!run.complete||run.status!=='finished'||run.step!==FINAL_STEP||p?.total_steps!==FINAL_STEP||p?.total_input_tokens!==INPUT_TOKENS||p?.global_batch_tokens!==1048576)return false;
 if(!c?.final||c.step!==FINAL_STEP||c.inputTokens!==INPUT_TOKENS||c.accounting!=='reference-aten-math-v2'||!Number.isFinite(c.averageGflopsPerInputToken)||Math.abs(c.averageGflopsPerInputToken/COMPUTE_TARGET-1)>.02)return false;
 if(!VAL_STEPS.every(step=>run.val.some(point=>point.step===step&&Number.isFinite(point.value))))return false;
 return CORE_STEPS.every(step=>{
  const aggregate=run.core.find(point=>point.step===step),tasks=run.taskEvaluations?.find(e=>e.step===step);
  if(!aggregate||!Number.isFinite(aggregate.value)||!tasks)return false;
  if(!specification.tasks.every(t=>Number.isFinite(tasks.centered[t.id])))return false;
  const mean=specification.tasks.reduce((sum,t)=>sum+tasks.centered[t.id],0)/specification.tasks.length;
  return Math.abs(mean-aggregate.value)<=1e-6;
 });
}

/** Missing, pilot, partial or unaccepted full-Dense runs never fill the baseline slot. */
export function fullDenseBaseline(runs:Run[]):Run|undefined {
 return runs.filter(acceptedFullDense).sort((a,b)=>(b.lastRecordedAt??0)-(a.lastRecordedAt??0))[0];
}
