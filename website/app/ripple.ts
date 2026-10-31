// A soft ripple under the pointer on navigation, buttons, and model cards. Skipped when motion is reduced.
const TARGETS='.tabs button,.primary,.secondary,.directory-open,.arena-open,.leaderboard-podium button,.chart-mode button,.metric-switch button,.leaderboard-metric button';

export function attachRipples(){
 const press=(event:PointerEvent)=>{
  if(window.matchMedia('(prefers-reduced-motion: reduce)').matches)return;
  const host=(event.target as Element|null)?.closest<HTMLElement>(TARGETS);
  if(!host)return;
  const box=host.getBoundingClientRect(),size=Math.max(box.width,box.height)*2.2,ripple=document.createElement('span');
  host.classList.add('ripple-host');
  ripple.className='ripple';
  ripple.style.cssText=`width:${size}px;height:${size}px;left:${event.clientX-box.left-size/2}px;top:${event.clientY-box.top-size/2}px`;
  host.appendChild(ripple);
  ripple.addEventListener('animationend',()=>ripple.remove(),{once:true});
 };
 document.addEventListener('pointerdown',press);
 return()=>document.removeEventListener('pointerdown',press);
}
