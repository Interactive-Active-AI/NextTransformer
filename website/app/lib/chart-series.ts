import type {Run} from './types';

/** Compare only evaluations recorded at exactly the same training step. */
export function bpbDifference(runs: Run[], baseline: Run | undefined): Run[] {
  if (!baseline) return [];
  const values = new Map(baseline.val.map(point => [point.step, point.value]));
  return runs.map(run => ({...run, val: run.val
    .filter(point => values.has(point.step))
    .map(point => ({step: point.step, value: point.value - values.get(point.step)!}))}));
}

/** One stable scale for all small multiples, including during replay. */
export function sharedChartDomain(runs: Run[], metric: 'val'|'core', allRange=false): [number,number] {
  const first=metric==='val'&&!allRange?8000:0;
  const values=runs.flatMap(run=>run[metric].filter(p=>p.step>=first&&Number.isFinite(p.value)).map(p=>p.value*(metric==='core'?100:1)));
  if(!values.length)return [0,1];
  const min=Math.min(...values),max=Math.max(...values),pad=(max-min)*.15||.02;
  return [metric==='core'?Math.max(0,min-pad):min-pad,max+pad];
}
