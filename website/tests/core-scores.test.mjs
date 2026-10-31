import assert from 'node:assert/strict';
import {test} from 'node:test';
import {runs} from './results.mjs';
import {CORE_TASKS,trackingTaskEvaluations,sharedTaskSteps,sharedTaskStep,taskEvaluation,mergeTaskEvaluations} from '../app/lib/core-scores.ts';

const data=runs();

test('all imported task evaluations reproduce their logged CORE and raw normalization',()=>{
 let count=0;
 for(const run of data)for(const evaluation of run.taskEvaluations??[]){
  assert.equal(Object.keys(evaluation.centered).length,22);
  const aggregate=run.core.find(p=>p.step===evaluation.step);
  assert.ok(aggregate,run.id+' missing aggregate at '+evaluation.step);
  assert.ok(Math.abs(Object.values(evaluation.centered).reduce((a,b)=>a+b,0)/22-aggregate.value)<1e-7);
  for(const task of CORE_TASKS)assert.ok(Math.abs((evaluation.raw[task.id]-task.randomBaseline)/(1-task.randomBaseline)-evaluation.centered[task.id])<1e-7);
  count++;
 }
 assert.ok(count>=64);
});

test('tracking import preserves negative centered scores and rejects partial or inconsistent rows',()=>{
 const source=data[0].taskEvaluations[0];
 const row={'trainer/step':source.step,'core/core_metric':data[0].core[0].value,...Object.fromEntries(Object.entries(source.raw).map(([key,value])=>['core/results/'+key,value]))};
 assert.equal(trackingTaskEvaluations([row],'trainer/step').length,1);
 const partial={...row};delete partial['core/results/boolq'];
 assert.deepEqual(trackingTaskEvaluations([partial],'trainer/step'),[]);
 assert.deepEqual(trackingTaskEvaluations([{...row,'core/core_metric':.99}],'trainer/step'),[]);
 const circle=data.find(r=>r.methodId==='circle').taskEvaluations.at(-1);
 const negative={'step':circle.step,'core/core_metric':Object.values(circle.centered).reduce((a,b)=>a+b,0)/22,...Object.fromEntries(Object.entries(circle.raw).map(([key,value])=>['core/results/'+key,value]))};
 assert.ok(trackingTaskEvaluations([negative],'step')[0].centered.boolq<0);
});

test('comparison uses exact shared checkpoints and never future or undated task summaries',()=>{
 const a={taskEvaluations:[{step:8000,centered:{arc_easy:.4}},{step:16000,centered:{arc_easy:.5}}]};
 const b={taskEvaluations:[{step:8000,centered:{arc_easy:.3}},{step:24000,centered:{arc_easy:.6}}]};
 assert.deepEqual(sharedTaskSteps([a,b]),[8000]);
 assert.equal(sharedTaskStep([a,b],7999),undefined);
 assert.equal(sharedTaskStep([a,b],24000),8000);
 assert.equal(taskEvaluation(a,24000),undefined);
 assert.deepEqual(sharedTaskSteps([a,{tasks:{arc_easy:.6}}]),[]);
 assert.deepEqual(sharedTaskSteps([a,b],'raw'),[]);
 const merged=mergeTaskEvaluations(a.taskEvaluations,[{step:8000,centered:{arc_easy:.45}}]);
 assert.deepEqual(merged.map(e=>e.step),[8000,16000]);
 assert.equal(merged[0].centered.arc_easy,.45);
 assert.equal(a.taskEvaluations[0].centered.arc_easy,.4);
});
