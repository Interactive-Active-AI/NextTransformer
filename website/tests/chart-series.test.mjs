import assert from 'node:assert/strict';
import {test} from 'node:test';
import {bpbDifference} from '../app/lib/chart-series.ts';
const baseline={id:'dense',methodId:'dense',val:[{step:0,value:1.2},{step:250,value:.9},{step:500,value:.8}]};
const method={id:'method',methodId:'method',val:[{step:0,value:1.1},{step:125,value:1.0},{step:250,value:.95},{step:500,value:.78},{step:750,value:.7}]};
test('BPB differences use matching checkpoints without interpolation or extrapolation',()=>{
 const result=bpbDifference([baseline,method],baseline);
 assert.deepEqual(result[1].val.map(p=>p.step),[0,250,500]);
 assert.ok(Math.abs(result[1].val[0].value+.1)<1e-10);
 assert.ok(Math.abs(result[1].val[1].value-.05)<1e-10);
 assert.ok(Math.abs(result[1].val[2].value+.02)<1e-10);
 assert.deepEqual(result[0].val.map(p=>p.value),[0,0,0]);
 assert.equal(method.val.length,5);
 assert.equal(method.val[2].value,.95);
});
test('no Dense record produces no inferred differences',()=>assert.deepEqual(bpbDifference([method],undefined),[]));

test('small multiples share one stable domain and ignore invalid values',async()=>{
 const {sharedChartDomain}=await import('../app/lib/chart-series.ts');
 const records=[{val:[{step:0,value:9},{step:8000,value:.8},{step:16000,value:.7}],core:[{step:8000,value:.2}]},{val:[{step:8000,value:.9},{step:16000,value:NaN}],core:[{step:8000,value:.4}]}];
 const domain=sharedChartDomain(records,'val');
 assert.ok(Math.abs(domain[0]-.67)<1e-10);assert.ok(Math.abs(domain[1]-.93)<1e-10);
 assert.ok(sharedChartDomain(records,'val',true)[1]>9);
 assert.deepEqual(sharedChartDomain(records,'core'),[17,43]);
 assert.deepEqual(sharedChartDomain([],'val'),[0,1]);
});
