// Advance widths of Inter semibold in em, rounded up per class so estimates never undershoot.
const CLASSES: [RegExp, number][] = [
  [/[ijlI]/, .27], [/[ .,:;'·]/, .33], [/[ftr1()[\]/]/, .42], [/[-_]/, .5],
  [/[acekzsvxy]/, .6], [/[bdghnopqu257]/, .63], [/[034689+=×&EFLJPRSTZ]/, .68],
  [/[ABCDGHKNOQUVXY]/, .77], [/[mwM]/, .92], [/[W→@%]/, 1.02],
];

/** Box width of a custom-diagram row with `cols` nodes: 24-unit gaps for the arrows across a 552-unit band. */
export const nodeWidth = (cols: number) => Math.min(170, (552 - (cols - 1) * 24) / cols);

export function textWidth(text: string, size: number): number {
  let em = 0;
  for (const ch of text) em += CLASSES.find(([re]) => re.test(ch))?.[1] ?? 1.02;
  return em * size;
}

/** One line if it fits; otherwise the most balanced two-line split that does not begin a line with a lone symbol. */
export function wrapLabel(text: string, size: number, room: number): string[] {
  if (textWidth(text, size) <= room) return [text];
  const words = text.split(' ');
  let best = [text], widest = Infinity;
  for (let i = 1; i < words.length; i++) {
    if (/^[·+→×=]$/.test(words[i])) continue;
    const lines = [words.slice(0, i).join(' '), words.slice(i).join(' ')];
    const width = Math.max(...lines.map(line => textWidth(line, size)));
    if (width < widest) { best = lines; widest = width; }
  }
  return best;
}

/** The largest common font size at which every label fits its box in at most two lines; lines that still overflow
 * at the smallest size are shortened with an ellipsis. */
export function fitLabels(labels: string[], room: number, sizes = [14, 13, 12]): {size: number; lines: string[][]} {
  const fits = (size: number) => labels.every(label => wrapLabel(label, size, room).every(line => textWidth(line, size) <= room));
  const size = sizes.find(fits) ?? sizes.at(-1)!;
  const shorten = (line: string) => {
    if (textWidth(line, size) <= room) return line;
    let cut = line;
    while (cut.length > 1 && textWidth(cut + '…', size) > room) cut = cut.slice(0, -1);
    return cut.trimEnd() + '…';
  };
  return {size, lines: labels.map(label => wrapLabel(label, size, room).map(shorten))};
}
