// The homepage families: ink for text and marks, mid for lines, soft for fills (see the website README).
export type Tone={name:string;ink:string;mid:string;soft:string};
export const TONES:Record<string,Tone>={
 baseline:{name:'Baseline',ink:'#6E7176',mid:'#ABADB0',soft:'#F3F4F4'},
 looped:{name:'Looped',ink:'#74638C',mid:'#ADA1C2',soft:'#F4F1F8'},
 sparse:{name:'Sparse',ink:'#965C66',mid:'#C89CA3',soft:'#FBF2F3'},
 linear:{name:'Linear',ink:'#4A6E9A',mid:'#96ABC4',soft:'#F1F3F7'},
 residual:{name:'Residual',ink:'#2F3C62',mid:'#868EA4',soft:'#EEEFF2'},
 hybrid:{name:'Hybrid',ink:'#B56754',mid:'#D4A79C',soft:'#F9F3F1'},
 others:{name:'Others',ink:'#926F23',mid:'#C0AB7F',soft:'#F6F3ED'}};
/** Display order everywhere: Others comes last. */
export const FAMILY_ORDER=Object.keys(TONES);
export const tone=(family:string|undefined)=>TONES[family??'baseline']??TONES.baseline;
