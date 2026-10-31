import assert from 'node:assert/strict';
import {test} from 'node:test';
import {baselineIdentity,fullDenseBaseline} from '../app/lib/baseline.ts';
import {rankingCheckpoint,leaderboardEntries} from '../app/lib/leaderboard.ts';
import {runs as publishedRuns} from './results.mjs';

const legacy={id:'original-run',methodId:'dense',name:'original name',cohort:'native-v3',val:[{step:8000,value:.8}],core:[{step:8000,value:.2}],proof:{src_head:'frozen'},trainingCompute:{averageGflopsPerInputToken:4.775226}};
test('SSSL display migration preserves run identity, measured points and provenance',()=>{
 const migrated=baselineIdentity(legacy);
 assert.equal(migrated.methodId,'sssl');
 assert.equal(migrated.originalMethodId,'dense');
 for(const key of ['id','name','val','core','proof','trainingCompute'])assert.equal(migrated[key],legacy[key]);
 assert.equal(legacy.methodId,'dense');
 assert.equal(baselineIdentity(migrated),migrated);
});
test('full Dense must have its own explicit identity and measurements',()=>{
 const pending={...legacy,id:'new-run',methodId:'dense_full',val:[],core:[]};
 assert.equal(fullDenseBaseline([legacy,baselineIdentity(legacy),pending]),undefined);
 const measured={...pending,val:[{step:8000,value:.9}]};
 assert.equal(fullDenseBaseline([legacy,measured]),undefined);
 assert.equal(baselineIdentity(measured),measured);
});
test('replay remains usable without inventing a Dense run or future checkpoint',()=>{
 const runs=[baselineIdentity(legacy)];
 assert.equal(rankingCheckpoint(undefined,'core',7999,runs),undefined);
 assert.equal(rankingCheckpoint(undefined,'core',9000,runs),8000);
 assert.equal(leaderboardEntries(runs,'core','replay',8000)[0].run.methodId,'sssl');
 assert.equal(fullDenseBaseline(runs),undefined);
});

const tasks=(await import('../data/core-tasks.json',{with:{type:'json'}})).default.tasks;
const accepted={...legacy,id:'accepted-full',methodId:'dense_full',status:'finished',complete:true,step:46875,
 proof:{protocol_hash:'frozen-protocol',src_head:'a'.repeat(40),total_steps:46875,total_input_tokens:49152000000,global_batch_tokens:1048576},
 baselineAcceptance:{status:'accepted',planVersion:'v2',protocolHash:'frozen-protocol',sourceCommit:'a'.repeat(40),reportUrl:'https://example.org/audit'},
 trainingCompute:{averageGflopsPerInputToken:4.67239832544785,inputTokens:49152000000,step:46875,final:true,accounting:'reference-aten-math-v2'},
 val:[...Array.from({length:188},(_,i)=>({step:i*250,value:.8})),{step:46875,value:.7}],
 core:[8000,16000,24000,32000,40000,46875].map(step=>({step,value:.2})),
 taskEvaluations:[8000,16000,24000,32000,40000,46875].map(step=>({step,centered:Object.fromEntries(tasks.map(t=>[t.id,.2]))}))};
test('a full Dense baseline requires explicit acceptance, complete evaluations and matched compute',()=>{
 assert.equal(fullDenseBaseline([accepted]),accepted);
 for(const patch of [
  {baselineAcceptance:undefined},{complete:false},{status:'pilot'},{step:8000},
  {val:accepted.val.slice(1)},{core:accepted.core.slice(1)},
  {taskEvaluations:accepted.taskEvaluations.map(e=>({...e,centered:{[tasks[0].id]:.2}}))},
  {core:accepted.core.map(p=>({...p,value:.3}))},
  {trainingCompute:{...accepted.trainingCompute,averageGflopsPerInputToken:5}},
  {trainingCompute:{...accepted.trainingCompute,accounting:'historical'}},
  {trainingCompute:{...accepted.trainingCompute,inputTokens:49152000001}},
  {proof:{...accepted.proof,protocol_hash:'different-protocol'}}
 ])assert.equal(fullDenseBaseline([{...accepted,...patch}]),undefined);
 const pilot={...accepted,id:'newer-pilot',lastRecordedAt:999,baselineAcceptance:undefined};
 assert.equal(fullDenseBaseline([pilot,accepted]),accepted);
});
test('the published full Dense run is the accepted baseline and SSSL never fills the slot',()=>{
 const published=publishedRuns().map(baselineIdentity),baseline=fullDenseBaseline(published);
 assert.equal(baseline?.methodId,'dense_full');
 assert.equal(baseline.baselineAcceptance.sourceCommit,baseline.proof.src_head);
 assert.equal(fullDenseBaseline(published.filter(r=>r.methodId!=='dense_full')),undefined);
});
