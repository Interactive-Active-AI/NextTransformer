import {useEffect,useState} from 'react';
import {ArrowUpRight,MessageCircle,MessagesSquare,QrCode,X} from 'lucide-react';
import {DISCORD_URL,DISCUSSIONS_URL,WECHAT_QR,docs} from './lib/links';

export default function Community({onClose}:{onClose:()=>void}){
 const [qr,setQr]=useState(true);
 useEffect(()=>{const previous=document.body.style.overflow;document.body.style.overflow='hidden';const close=(e:KeyboardEvent)=>{if(e.key==='Escape')onClose();};window.addEventListener('keydown',close);return()=>{document.body.style.overflow=previous;window.removeEventListener('keydown',close);};},[onClose]);
 return <div className="modal-overlay" onClick={onClose}><section className="info-modal community-modal" role="dialog" aria-modal="true" aria-labelledby="community-title" onClick={e=>e.stopPropagation()}>
  <button className="close-button" onClick={onClose} aria-label="Close community"><X size={20}/></button>
  <p className="eyebrow">COMMUNITY</p>
  <h2 id="community-title">Research together.<em>Questions, ideas and new results.</em></h2>
  <div className={'community-grid'+(DISCORD_URL?'':' two')}>
   {DISCORD_URL&&<a className="community-card discord" href={DISCORD_URL} target="_blank" rel="noreferrer"><span className="community-icon"><MessageCircle size={20}/></span><b>Discord</b><p>Chat with the team and other researchers, worldwide.</p><span className="community-action">Join the server <ArrowUpRight size={14}/></span></a>}
   <div className="community-card wechat"><span className="community-icon"><QrCode size={20}/></span><b>WeChat group</b><p>For researchers in China. Scan to join.</p>{qr?<img src={WECHAT_QR} width="148" height="148" alt="WeChat group QR code" onError={()=>setQr(false)}/>:<span className="qr-placeholder" role="img" aria-label="QR code coming soon">QR code coming soon</span>}</div>
   <a className="community-card discussions" href={DISCUSSIONS_URL} target="_blank" rel="noreferrer"><span className="community-icon"><MessagesSquare size={20}/></span><b>GitHub Discussions</b><p>Questions, method proposals and announcements, searchable.</p><span className="community-action">Open discussions <ArrowUpRight size={14}/></span></a>
  </div>
  <p className="community-note">New to the project? Start with the <a href={docs('getting-started/quick-start/')}>quick start</a> or the <a href={docs('community/')}>community guide</a>.</p>
 </section></div>;
}
