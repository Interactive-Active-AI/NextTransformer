import assert from 'node:assert/strict';
import {test} from 'node:test';
import {FINAL_STEP,constellation,placeLabels,spread} from '../app/lib/constellation.ts';
import {rankSteps,rankTrajectories} from '../app/lib/ranks.ts';
import {fullDenseBaseline} from '../app/lib/baseline.ts';
import {leaderboardEntries,preferredRuns} from '../app/lib/leaderboard.ts';
import {methods,runs} from './results.mjs';

const final=(run,kind)=>run[kind].find(p=>p.step===FINAL_STEP)?.value;

test('the constellation places every final run by its exact difference from Dense at the final step',()=>{
 const all=runs(),baseline=fullDenseBaseline(all),pool=preferredRuns(all).filter(r=>r.id!==baseline.id);
 const stars=constellation(pool,methods(),baseline,m=>m?.name??'');
 const measured=pool.filter(r=>final(r,'val')!==undefined&&final(r,'core')!==undefined);
 assert.equal(stars.length,measured.length);
 for(const s of stars){
  assert.equal(s.dBpb,final(s.run,'val')-final(baseline,'val'));
  assert.ok(Math.abs(s.dCore-(final(s.run,'core')-final(baseline,'core'))*100)<1e-9);
 }
 assert.deepEqual(constellation(pool,methods(),undefined,()=>''),[]);
 assert.ok(spread(-.04)<spread(-.01)&&spread(-.01)<0&&spread(0)===0&&spread(.01)>0);
});

test('constellation labels never overlap each other or a star, and stay inside the plot',()=>{
 const points=Array.from({length:24},(_,i)=>({id:'p'+i,x:40+(i%6)*18,y:40+Math.floor(i/6)*9,r:5,text:'Model '+i}));
 const bounds={x:0,y:0,w:320,h:200},labels=placeLabels(points,bounds);
 const boxes=[...labels].map(([id,l])=>{const p=points.find(q=>q.id===id),w=p.text.length*6.6;return {x:l.anchor==='start'?l.x:l.anchor==='end'?l.x-w:l.x-w/2,y:l.y-11,w,h:14};});
 const overlap=(a,b)=>a.x<b.x+b.w-.01&&b.x<a.x+a.w-.01&&a.y<b.y+b.h-.01&&b.y<a.y+a.h-.01;
 boxes.forEach((a,i)=>{
  assert.ok(a.x>=0&&a.x+a.w<=bounds.w);
  boxes.slice(i+1).forEach(b=>assert.ok(!overlap(a,b)));
  points.forEach(p=>assert.ok(!overlap(a,{x:p.x-p.r,y:p.y-p.r,w:2*p.r,h:2*p.r})));
 });
 assert.ok(labels.size>0&&labels.size<points.length);
});

test('the rank race uses exact evaluation steps and agrees with the final leaderboard',()=>{
 const pool=preferredRuns(runs());
 assert.deepEqual(rankSteps(pool,'core'),[8000,16000,24000,32000,40000,FINAL_STEP]);
 assert.ok(rankSteps(pool,'val').every(s=>s%4000===0||s===FINAL_STEP));
 for(const metric of ['core','val']){
  const tracks=rankTrajectories(pool,metric,[FINAL_STEP]);
  for(const entry of leaderboardEntries(pool,metric,'final',undefined))assert.equal(tracks.get(entry.run.id).at(-1)?.rank,entry.rank);
 }
 const tie=[{id:'a',core:[{step:8000,value:.3}]},{id:'b',core:[{step:8000,value:.3}]},{id:'c',core:[{step:8000,value:.2}]}];
 const ranks=rankTrajectories(tie,'core',[8000]);
 assert.deepEqual(['a','b','c'].map(id=>ranks.get(id)[0].rank),[1,1,3]);
});
