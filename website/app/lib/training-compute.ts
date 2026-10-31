import {z} from 'zod';
import type {Run,TrainingCompute} from './types';
export const LEGACY_ACCOUNTING='Historical matrix FLOPs · 3 × forward + recompute';
export function getTrainingCompute(run:Run|undefined):TrainingCompute|undefined {
 const c=run?.trainingCompute;
 return c&&Number.isFinite(c.averageGflopsPerInputToken)&&c.averageGflopsPerInputToken>0&&Number.isFinite(c.inputTokens)&&c.inputTokens>0&&Number.isInteger(c.step)&&c.step>=0&&c.accounting?c:undefined;
}
/** Cost deltas require the same completed budget AND accounting definition. */
export function trainingComputeDelta(run:Run|undefined,baseline:Run|undefined):number|undefined {
 const a=getTrainingCompute(run),b=getTrainingCompute(baseline);
 if(!a||!b||!a.final||!b.final||a.accounting!==b.accounting||a.step!==b.step||a.inputTokens!==b.inputTokens)return undefined;
 return (a.averageGflopsPerInputToken/b.averageGflopsPerInputToken-1)*100;
}
/** Only explicitly reported cumulative fields; configuration cost proofs are not averages. */
export function computeFromSummary(summary:Record<string,any>,step:number,final:boolean,accounting:string,source:string):TrainingCompute|undefined {
 // Native tracker prefixes event fields with train/; JSON exports use bare counters.
 // Read each pair together so counters from different event scopes cannot be mixed.
 const pair=[['train/training_flops_estimate','train/input_tokens'],['training_flops_estimate','input_tokens']].find(([f,t])=>Number.isFinite(summary[f])&&summary[f]>0&&Number.isFinite(summary[t])&&summary[t]>0);
 if(!pair)return undefined;
 const [flops,tokens]=pair.map(key=>summary[key]);
 if(!Number.isFinite(flops)||flops<=0||!Number.isFinite(tokens)||tokens<=0||!Number.isInteger(step)||step<0)return undefined;
 return {averageGflopsPerInputToken:flops/tokens/1e9,trainingFlopsEstimate:flops,inputTokens:tokens,step,final,accounting,source};
}

// Validates a reported training-compute record.
export const trainingComputeSchema=z.object({
 averageGflopsPerInputToken:z.number().finite().positive().optional(),
 trainingFlopsEstimate:z.number().finite().positive().optional(),inputTokens:z.number().finite().positive(),
 step:z.number().int().min(0).max(10000000),final:z.boolean().default(false),
 accounting:z.string().trim().min(3).max(200),source:z.string().trim().min(3).max(300).default('Contributor-reported training logs'),
}).superRefine((d,c)=>{
 if(d.averageGflopsPerInputToken===undefined&&d.trainingFlopsEstimate===undefined)c.addIssue({code:'custom',message:'Report an average or a cumulative FLOPs estimate'});
 if(d.averageGflopsPerInputToken!==undefined&&d.trainingFlopsEstimate!==undefined){const expected=d.trainingFlopsEstimate/d.inputTokens/1e9;if(Math.abs(expected-d.averageGflopsPerInputToken)>Math.max(5e-7,expected*1e-6))c.addIssue({code:'custom',message:'Average compute does not match cumulative FLOPs / input tokens / 1e9'});}
}).transform(d=>({...d,averageGflopsPerInputToken:d.trainingFlopsEstimate!==undefined?d.trainingFlopsEstimate/d.inputTokens/1e9:d.averageGflopsPerInputToken!}));
