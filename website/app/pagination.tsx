'use client';
import {ChevronLeft,ChevronRight} from 'lucide-react';
export default function Pagination({page,total,size,onChange,label}:{page:number;total:number;size:number;onChange:(page:number)=>void;label:string}){
 const pages=Math.ceil(total/size);
 if(pages<2)return null;
 return <nav className="pagination" aria-label={label+' pagination'}><span>{page*size+1}–{Math.min(total,(page+1)*size)} of {total}</span><div><button disabled={page===0} onClick={()=>onChange(page-1)} aria-label={'Previous '+label.toLowerCase()}><ChevronLeft size={17}/></button>{Array.from({length:pages},(_,i)=><button key={i} aria-label={label+' page '+(i+1)} aria-current={page===i?'page':undefined} onClick={()=>onChange(i)}>{i+1}</button>)}<button disabled={page===pages-1} onClick={()=>onChange(page+1)} aria-label={'Next '+label.toLowerCase()}><ChevronRight size={17}/></button></div></nav>;
}
