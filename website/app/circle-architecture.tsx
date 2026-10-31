'use client';
import {useId} from 'react';

const ink='#33465A',muted='#647B93',blue='#527FA8',purple='#8A6CB2',rose='#BA7185';

export default function CircleArchitecture({large=false}:{large?:boolean}) {
 const id=useId().replace(/[^a-zA-Z0-9_-]/g,'');
 const point=(angle:number)=>({x:160+102*Math.cos(angle*Math.PI/180),y:165+102*Math.sin(angle*Math.PI/180)});
 const ring='M160 63A102 102 0 1 1 159.99 63';
 const positions=Array.from({length:9},(_,i)=>point(-90+i*40));
 const arrow=(suffix:string)=><defs><marker id={id+suffix} markerWidth="7" markerHeight="7" refX="5.5" refY="3.5" orient="auto" markerUnits="userSpaceOnUse"><path d="M1 1L5.5 3.5L1 6" fill="none" stroke="context-stroke" strokeWidth="1.35" strokeLinecap="round" strokeLinejoin="round"/></marker></defs>;
 return <figure className={'architecture-figure circle-figure'+(large?' is-detailed':'')}>
  <figcaption><strong>One full circuit. Then choose when to leave.</strong><span>Nine distinct blocks are reused around a ring. After the first circuit, a shared halter decides whether each token continues.</span></figcaption>
  <div className="circle-flow">
   <section className="circle-stage circle-stage-ring">
    {large&&<div className="circle-stage-heading"><b>01</b><div><strong>Complete the first circuit</strong><p>All tokens execute B₁ through B₉.</p></div></div>}
    <svg className="circle-ring" viewBox="0 0 320 330" role="img" aria-label="Nine distinct core blocks connected clockwise, with shared block weights reused on subsequent circuits">
     {arrow('-ring')}
     <text x="160" y="18" textAnchor="middle" fill={blue} fontSize="12">Prelude · 3 blocks</text>
     <path d="M160 26V41" fill="none" stroke={blue} strokeWidth="1.6" markerEnd={`url(#${id}-ring)`}/>
     <circle cx="160" cy="165" r="102" fill="none" stroke="#E1D8ED" strokeWidth="1.2"/>
     {positions.map((_,i)=>{const start=point(-90+i*40+14),end=point(-90+(i+1)*40-14);return <path key={i} d={`M${start.x} ${start.y}A102 102 0 0 1 ${end.x} ${end.y}`} fill="none" stroke={purple} strokeWidth="1.7" markerEnd={`url(#${id}-ring)`}/>;})}
     <circle r="4.2" fill={purple} className="circle-motion"><animateMotion path={ring} dur="7s" repeatCount="indefinite"/></circle>
     {positions.map((p,i)=><g key={i}><rect x={p.x-22} y={p.y-16} width="44" height="32" rx="7" fill={i===0?'#EEF5FC':'#F3EEF9'} stroke={i===0?blue:purple} strokeWidth="1.25"/><text x={p.x} y={p.y+5} textAnchor="middle" fill={i===0?blue:purple} fontSize="14" fontWeight="600">B{i+1}</text></g>)}
     <circle cx="160" cy="165" r="64" fill="#FAF8FE"/>
     <text x="160" y="160" textAnchor="middle" fill={ink} fontSize="18" fontWeight="600">9 distinct blocks</text>
     <text x="160" y="182" textAnchor="middle" fill={muted} fontSize="12">reused each circuit</text>
     <text x="160" y="313" textAnchor="middle" fill={purple} fontSize="12">B₉ → B₁ on the next circuit</text>
    </svg>
   </section>
   {large&&<section className="circle-stage circle-stage-adaptive">
    <div className="circle-stage-heading"><b>02</b><div><strong>Decide after each block</strong><p>Per-token halting starts at step 9.</p></div></div>
    <svg className="circle-decision" viewBox="0 0 320 330" role="img" aria-label="After the first circuit, a shared halter uses state, state change and position. Continue to the next block or freeze the state and stop supplying in-loop keys and values. All states rejoin at the coda.">
     {arrow('-decision')}
     <rect x="88" y="10" width="144" height="40" rx="7" fill="#F3EEF9" stroke={purple} strokeWidth="1.2"/>
     <text x="160" y="35" textAnchor="middle" fill={purple} fontSize="14" fontWeight="550">Next core block</text>
     <path d="M160 52V76" fill="none" stroke={purple} strokeWidth="1.6" markerEnd={`url(#${id}-decision)`}/>
     <rect x="68" y="84" width="184" height="60" rx="8" fill="#EFF4FC" stroke={blue} strokeWidth="1.2"/>
     <text x="160" y="109" textAnchor="middle" fill={ink} fontSize="16" fontWeight="600">Shared halter</text>
     <text x="160" y="129" textAnchor="middle" fill={muted} fontSize="11">state · Δstate · position</text>
     <path d="M315 114H260" fill="none" stroke={blue} strokeWidth="1.6" markerEnd={`url(#${id}-decision)`}/>
     <text x="313" y="99" textAnchor="end" fill={blue} fontSize="11">after B₉</text>
     <path d="M160 146V153" fill="none" stroke={blue} strokeWidth="1.5"/>
     <path d="M160 157L179 176L160 195L141 176Z" fill="white" stroke={blue} strokeWidth="1.3"/>
     <text x="160" y="180" textAnchor="middle" fill={ink} fontSize="10">halt?</text>
     <path d="M139 176H28V30H80" fill="none" stroke={purple} strokeWidth="1.7" strokeDasharray="5 5" markerEnd={`url(#${id}-decision)`} className="circle-continue"/>
     <text x="35" y="165" fill={purple} fontSize="12" fontWeight="550">Continue</text>
     <path d="M160 197V214" fill="none" stroke={rose} strokeWidth="1.6" markerEnd={`url(#${id}-decision)`}/>
     <text x="177" y="211" fill={rose} fontSize="12" fontWeight="550">Exit</text>
     <rect x="88" y="221" width="144" height="45" rx="7" fill="#FCF0F3" stroke={rose} strokeWidth="1.2"/>
     <text x="160" y="241" textAnchor="middle" fill={ink} fontSize="14" fontWeight="600">Freeze state</text>
     <text x="160" y="257" textAnchor="middle" fill={rose} fontSize="11">no in-loop KV</text>
     <path d="M160 268V279" fill="none" stroke={rose} strokeWidth="1.6" markerEnd={`url(#${id}-decision)`}/>
     <rect x="71" y="286" width="178" height="40" rx="7" fill="#EFF5FB" stroke={blue} strokeWidth="1.2"/>
     <text x="160" y="303" textAnchor="middle" fill={ink} fontSize="14" fontWeight="550">Coda + readout</text>
     <text x="160" y="318" textAnchor="middle" fill={muted} fontSize="10">all tokens rejoin · 3 coda blocks</text>
     <circle r="3.5" fill={purple} className="circle-motion"><animateMotion path="M139 176H28V30H80" dur="4s" repeatCount="indefinite"/></circle>
    </svg>
   </section>}
  </div>
  <div className="circle-facts"><div><b>9</b><span>required core steps</span></div><div><b>27</b><span>maximum core steps</span></div><div><b>17.5</b><span>target mean depth</span></div></div>
  {large&&<p className="circle-limit">At step 27, all remaining tokens exit. Exited states stay frozen until the coda.</p>}
 </figure>;
}
