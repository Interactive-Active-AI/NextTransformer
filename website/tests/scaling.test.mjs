import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {test} from 'node:test';
import {CHINCHILLA,FINAL_STEP,LADDER_STATUS,VIEWS,cellStatus,compact,denseFlops,denseShape,fitSurface,isoFlopCurve,logTicks,plannedPoints,project,scalingPoints,scalingSurfaces,surfaceAt} from '../app/lib/scaling.ts';
import {methods,runs} from './results.mjs';

const ladder=JSON.parse(readFileSync(new URL('../../results/scaling/ladder.json',import.meta.url),'utf8'));
const V=32768,T=4096;
// Theoretical training FLOPs per input token, as the scaling trainer counts them; MoE counts active parameters.
const flops=(cell)=>{const d=cell.width,L=cell.layers;return 6*cell.activeBlockParameters+12*T*d*L+6*d*V;};

test('every ladder cell names a known family and tier, with a unique id and a known status',()=>{
 const families=new Set(ladder.families.map(f=>f.id)),tiers=new Set(ladder.tiers.map(t=>t.id)),ids=ladder.cells.map(c=>c.id);
 assert.equal(new Set(ids).size,ids.length);
 for(const c of ladder.cells){
  assert.ok(families.has(c.family)&&tiers.has(c.tier),c.id);
  assert.ok(c.status in LADDER_STATUS,c.id);
  assert.ok(c.claim===null||(c.claim.by&&c.claim.issue.startsWith('https://')),c.id);
 }
});

test('sized cells spend their tier budget within 1%, in whole updates, at width 64 × layers',()=>{
 const tiers=Object.fromEntries(ladder.tiers.map(t=>[t.id,t]));
 for(const c of ladder.cells.filter(c=>c.layers&&c.status!=='trained')){
  const tier=tiers[c.tier];
  assert.equal(c.width,64*c.layers,c.id);
  assert.equal(c.tokens,c.steps*tier.batchTokens,c.id);
  assert.ok(Math.abs(c.tokens*flops(c)/tier.flops-1)<.01,c.id);
  assert.equal(c.parameters,c.blockParameters+(2+c.layers/2)*V*c.width,c.id);
  if(c.family==='dense')assert.equal(c.blockParameters,12*c.width**2*c.layers,c.id);
 }
 const reference=ladder.cells.find(c=>c.id==='dense-C200-L40');
 assert.equal(reference.record,'dense_200b');
 assert.ok(Math.abs(reference.tokens*flops(reference)/tiers.C200.flops-1)<1e-4);
});

test('a published record marks its cell recorded and takes it off the floor of the plot',()=>{
 const reference=ladder.cells.find(c=>c.record==='dense_200b'),record={id:'dense_200b'};
 assert.equal(cellStatus(reference,[]),'trained');
 assert.equal(cellStatus(reference,[record]),'recorded');
 assert.ok(plannedPoints(ladder,[]).some(p=>p.id===reference.id));
 assert.ok(!plannedPoints(ladder,[record]).some(p=>p.id===reference.id));
 assert.ok(!plannedPoints(ladder,[]).some(p=>p.id.endsWith('forecast')));
});

test('the plot shows one point per run with a final CORE, a final BPB and a parameter count',()=>{
 const all=runs(),ms=methods(),points=scalingPoints(all,ms,[]);
 const expected=all.filter(r=>r.core.some(p=>p.step===FINAL_STEP)&&Number.isFinite(r.finalBpb)&&ms.find(m=>m.id===r.methodId)?.params);
 assert.equal(points.length,expected.length);
 assert.ok(points.every(p=>p.stage==='reference'&&p.context===2048&&p.tokens===49152000000&&p.core>0&&p.bpb>0));
});

test('log ticks, compact labels and the projection behave at the preset views',()=>{
 assert.deepEqual(logTicks([8,11]),[1e8,1e9,1e10,1e11]);
 assert.deepEqual(logTicks([9.1,9.6]),[2e9]);
 assert.equal(compact(4991221760),'4.99B');
 assert.equal(compact(49152000000),'49.2B');
 const side=VIEWS.parameters,front=project(.3,-1,.5,side),back=project(.3,1,.5,side);
 assert.ok(Math.abs(front.x-back.x)<1e-12&&Math.abs(front.y-back.y)<1e-12);
 const top=VIEWS.floor,low=project(.2,.4,-1,top),high=project(.2,.4,1,top);
 assert.ok(Math.abs(low.x-high.x)<1e-12&&Math.abs(low.y-high.y)<1e-12);
});

test('the Dense rule reproduces the reference run, and iso-FLOP curves spend their budget exactly',()=>{
 const reference=denseShape(40),c200=ladder.tiers.find(t=>t.id==='C200').flops;
 assert.equal(reference.parameters,4991221760);
 assert.ok(Math.abs(reference.flops*199999094784/c200-1)<1e-4);
 assert.ok(Math.abs(denseFlops(reference.parameters,199999094784)/(reference.flops*199999094784)-1)<1e-6);
 for(const p of isoFlopCurve(1e20,8,48,4))assert.ok(Math.abs(denseFlops(p.parameters,p.tokens)/1e20-1)<1e-6);
});

test('surface fits recover known coefficients and stay monotone',()=>{
 const truth={kind:'provisional',metric:'core',intercept:70,n:8000,d:30000,...CHINCHILLA,rmse:0,runs:0};
 const grid=[3e8,1e9,3e9].flatMap(params=>[2e10,6e10,2e11].map(tokens=>({id:params+'/'+tokens,params,tokens,core:surfaceAt(truth,params,tokens)/100,bpb:.7,stage:'reference'})));
 const fit=fitSurface(grid,'core');
 assert.ok(Math.abs(fit.intercept-70)<1e-6&&Math.abs(fit.n/8000-1)<1e-6&&Math.abs(fit.d/30000-1)<1e-6);
 assert.equal(fitSurface(grid.filter(p=>p.tokens===2e10),'core'),undefined);
 const real=scalingSurfaces(scalingPoints(runs(),methods(),[]).concat([{id:'s',name:'s',family:'baseline',stage:'scaling',params:5e9,tokens:2e11,core:.46,bpb:.575,context:4096}]),'core')[0];
 assert.equal(real.kind,'provisional');
 assert.ok(real.n>=0&&real.d>=0&&surfaceAt(real,5e9,2e11)>surfaceAt(real,5e8,2e10));
});

test('a family with ladder runs at two budgets gets its own fitted surface',()=>{
 const truth={kind:'family',metric:'bpb',intercept:.4,n:90,d:200,alpha:.3,beta:.36,rmse:0,runs:0};
 // Sampled like the ladder: five sizes on each of two iso-FLOP lines, tokens = C / 6N.
 const own=[1e19,1e20].flatMap(flops=>[1e8,2e8,4e8,8e8,1.6e9].map(params=>{const tokens=flops/(6*params);return {id:flops+'/'+params,name:'',family:'baseline',stage:'scaling',params,tokens,core:.3,bpb:surfaceAt(truth,params,tokens),context:4096};}));
 const [fit]=scalingSurfaces(own,'bpb');
 assert.equal(fit.kind,'family');
 assert.ok(Math.abs(fit.alpha-.3)<.011&&Math.abs(fit.beta-.36)<.011&&fit.rmse<1e-3);
});
