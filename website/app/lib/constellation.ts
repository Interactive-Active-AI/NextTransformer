import type {Method,Run} from './types';

export const FINAL_STEP=46875;
export type Star={id:string;methodId:string;family:string;name:string;params:number;dBpb:number;dCore:number;run:Run;method?:Method};
const final=(run:Run|undefined,kind:'val'|'core')=>run?.[kind].find(p=>p.step===FINAL_STEP&&Number.isFinite(p.value))?.value;

// Every run measured at the final step, placed by its difference from the accepted Dense baseline
// at that same step: BPB in bits per byte, CORE ×100. Nothing is interpolated or inferred.
export function constellation(runs:Run[],methods:Method[],baseline:Run|undefined,name:(m:Method|undefined)=>string):Star[]{
 const val=final(baseline,'val'),core=final(baseline,'core');
 if(!baseline||val===undefined||core===undefined)return [];
 return runs.flatMap(run=>{
  const v=final(run,'val'),c=final(run,'core'),method=methods.find(m=>m.id===run.methodId);
  if(v===undefined||c===undefined)return [];
  return [{id:run.id,methodId:run.methodId,family:method?.family??'baseline',name:name(method)||run.methodId,params:method?.params??0,dBpb:v-val,dCore:(c-core)*100,run,method}];
 });
}

// A signed square root spreads the many small BPB differences near the baseline while keeping order and sign.
export const spread=(v:number)=>Math.sign(v)*Math.sqrt(Math.abs(v));

type Box={x:number;y:number;w:number;h:number};
const overlaps=(a:Box,b:Box)=>a.x<b.x+b.w&&b.x<a.x+a.w&&a.y<b.y+b.h&&b.y<a.y+a.h;

// Greedy label placement: right, left, above, below; a label that fits nowhere is shown only on hover.
export function placeLabels(points:{id:string;x:number;y:number;r:number;text:string}[],bounds:Box,charWidth=6.6,height=14){
 const placed:Box[]=points.map(p=>({x:p.x-p.r,y:p.y-p.r,w:2*p.r,h:2*p.r})),labels=new Map<string,{x:number;y:number;anchor:'start'|'end'|'middle'}>();
 const inside=(b:Box)=>b.x>=bounds.x&&b.y>=bounds.y&&b.x+b.w<=bounds.x+bounds.w&&b.y+b.h<=bounds.y+bounds.h;
 for(const p of points){
  const w=p.text.length*charWidth,gap=p.r+5;
  const options=[{x:p.x+gap,y:p.y+4,anchor:'start' as const,box:{x:p.x+gap,y:p.y-height/2,w,h:height}},
   {x:p.x-gap,y:p.y+4,anchor:'end' as const,box:{x:p.x-gap-w,y:p.y-height/2,w,h:height}},
   {x:p.x,y:p.y-gap-2,anchor:'middle' as const,box:{x:p.x-w/2,y:p.y-gap-height,w,h:height}},
   {x:p.x,y:p.y+gap+11,anchor:'middle' as const,box:{x:p.x-w/2,y:p.y+gap,w,h:height}}];
  const choice=options.find(o=>inside(o.box)&&!placed.some(b=>overlaps(b,o.box)));
  if(choice){placed.push(choice.box);labels.set(p.id,{x:choice.x,y:choice.y,anchor:choice.anchor});}
 }
 return labels;
}
