import type {LadderCell,Method,Run,ScalingLadder,ScalingRecord} from './types';

export const FINAL_STEP=46875;
export const REFERENCE_CONTEXT=2048;
export type ScalingMetric='core'|'bpb';
export type ScalingPoint={id:string;name:string;family:string;stage:'reference'|'scaling';params:number;active?:number;tokens:number;flops?:number;core:number;bpb:number;context:number};
export type PlannedPoint={id:string;family:string;tier:string;params:number;active:number;tokens:number;status:string};
export type View={yaw:number;pitch:number};

// Scaling families share the homepage palette: the Dense ladder takes the baseline tone.
export const familyKey=(family:string)=>family==='dense'?'baseline':family;

/** One point per finished run: final CORE and BPB of the 49B protocol's runs, and of scaling-stage records. */
export function scalingPoints(runs:Run[],methods:Method[],records:ScalingRecord[]):ScalingPoint[]{
 const reference=runs.flatMap(run=>{
  const method=methods.find(m=>m.id===run.methodId),core=run.core.find(p=>p.step===FINAL_STEP)?.value,tokens=run.trainingCompute?.inputTokens;
  return method?.params&&core!==undefined&&Number.isFinite(run.finalBpb)&&tokens?[{id:run.id,name:method.name,family:method.family,stage:'reference' as const,params:method.params,tokens,flops:run.trainingCompute?.trainingFlopsEstimate,core,bpb:run.finalBpb!,context:REFERENCE_CONTEXT}]:[];
 });
 const scaling=records.map(r=>({id:'scaling/'+r.id,name:r.name,family:familyKey(r.family),stage:'scaling' as const,params:r.parameters.total,active:r.parameters.blocks,tokens:r.training.inputTokens,flops:r.training.trainingFlopsEstimate,core:r.finalCore.value,bpb:r.finalBpb.val,context:r.model.sequenceLength}));
 return [...reference,...scaling];
}

export const LADDER_STATUS:Record<string,string>={open:'Open',claimed:'Claimed',running:'Training',submitted:'In review',recorded:'Recorded',trained:'Trained · record in review',gated:'Waiting for the recipe gate',forecast:'After the forecast'};
/** A cell whose record is published counts as recorded, whatever the ladder file says. */
export const cellStatus=(cell:LadderCell,records:ScalingRecord[])=>cell.record&&records.some(r=>r.id===cell.record)?'recorded':cell.status;

/** Planned cells with a fixed size, drawn on the floor of the plot until their record arrives. */
export function plannedPoints(ladder:ScalingLadder|undefined,records:ScalingRecord[]):PlannedPoint[]{
 return (ladder?.cells??[]).flatMap(c=>{const status=cellStatus(c,records);return c.parameters&&c.tokens&&status!=='recorded'?[{id:c.id,family:familyKey(c.family),tier:c.tier,params:c.parameters,active:c.activeBlockParameters??c.parameters,tokens:c.tokens,status}]:[];});
}

export function logExtent(values:number[],pad=.08):[number,number]{
 const logs=values.filter(v=>v>0).map(Math.log10);
 if(!logs.length)return [0,1];
 const lo=Math.min(...logs),hi=Math.max(...logs),span=Math.max(hi-lo,.5);
 return [lo-span*pad,hi+span*pad];
}

export function linearExtent(values:number[],pad=.12):[number,number]{
 if(!values.length)return [0,1];
 const lo=Math.min(...values),hi=Math.max(...values),span=hi-lo||Math.abs(hi)*.1||1;
 return [lo-span*pad,hi+span*pad];
}

/** Tick values at 1, 2 and 5 per decade inside [lo, hi] (log10 units); only decades when that gives at least three. */
export function logTicks([lo,hi]:[number,number]):number[]{
 const all:number[]=[],decades:number[]=[];
 for(let e=Math.floor(lo);e<=Math.ceil(hi);e++)for(const m of [1,2,5]){const v=m*10**e,l=Math.log10(v);if(l>=lo&&l<=hi){all.push(v);if(m===1)decades.push(v);}}
 return decades.length>=3?decades:all;
}

export function compact(v:number){
 for(const [unit,scale] of [['T',1e12],['B',1e9],['M',1e6]] as const)if(v>=scale)return `${+(v/scale).toPrecision(3)}${unit}`;
 return String(Math.round(v));
}

export const VOCAB=32768,CONTEXT=4096;

/** The Dense size rule at a (possibly fractional) layer count: width 64 × layers, value embeddings on half the layers. */
export function denseShape(layers:number){
 const d=64*layers,blocks=12*d*d*layers;
 return {width:d,blocks,parameters:blocks+(2+layers/2)*VOCAB*d,flops:72*d*d*layers+12*CONTEXT*d*layers+6*d*VOCAB};
}

/** Training FLOPs of the Dense-rule model with this many parameters, trained on this many tokens. */
export function denseFlops(parameters:number,tokens:number){
 let lo=1,hi=512;
 for(let i=0;i<48;i++){const mid=(lo+hi)/2;if(denseShape(mid).parameters<parameters)lo=mid;else hi=mid;}
 return denseShape((lo+hi)/2).flops*tokens;
}

/** The floor curve of one training budget under the Dense size rule: parameters and tokens of every size that spends it. */
export function isoFlopCurve(flops:number,from=4,to=160,step=.25){
 const points:{parameters:number;tokens:number;layers:number}[]=[];
 for(let layers=from;layers<=to;layers+=step){const s=denseShape(layers);points.push({parameters:s.parameters,tokens:flops/s.flops,layers});}
 return points;
}

// The default view faces the budget sheets: iso-FLOP curves run from many tokens and few parameters to the reverse.
export const VIEWS:Record<string,View>={angled:{yaw:-.72,pitch:.42},parameters:{yaw:0,pitch:0},tokens:{yaw:Math.PI/2,pitch:0},floor:{yaw:0,pitch:Math.PI/2}};

// Half-extents of the drawn box: wider than tall, so the plot fills a landscape card.
export const BOX={x:1.35,y:1.35,z:.85};

/** Rotates a point of the [-1, 1]³ data cube, drawn as BOX, about the vertical axis, then tilts it; screen y grows downward and depth toward the viewer. */
export function project(x:number,y:number,z:number,{yaw,pitch}:View){
 x*=BOX.x;y*=BOX.y;z*=BOX.z;
 const c=Math.cos(yaw),s=Math.sin(yaw),rx=x*c-y*s,ry=x*s+y*c;
 return {x:rx,y:ry*Math.sin(pitch)-z*Math.cos(pitch),depth:ry*Math.cos(pitch)+z*Math.sin(pitch)};
}
