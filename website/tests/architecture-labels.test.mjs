import assert from 'node:assert/strict';
import {test} from 'node:test';
import {methods} from './results.mjs';
import {fitLabels, nodeWidth, textWidth, wrapLabel} from '../app/lib/node-labels.ts';

test('every custom-diagram node label is shown in full, in at most two lines that fit its box', () => {
  for (const method of methods().filter(m => m.architecture?.kind === 'custom')) {
    const labels = method.architecture.nodes.slice(0, 8);
    const cols = labels.length <= 4 ? labels.length : Math.ceil(labels.length / 2);
    const room = nodeWidth(cols) - 10;
    const {size, lines} = fitLabels(labels, room);
    lines.forEach((label, i) => {
      assert.equal(label.join(' '), labels[i], `${method.id}: "${labels[i]}" was shortened`);
      assert.ok(label.length <= 2, `${method.id}: "${labels[i]}" needs ${label.length} lines`);
      for (const line of label) assert.ok(textWidth(line, size) <= room, `${method.id}: "${line}" overflows`);
    });
  }
});

test('labels break between words, never before a lone symbol', () => {
  assert.deepEqual(wrapLabel('HC read · write · mix', 14, 90), ['HC read ·', 'write · mix']);
  assert.deepEqual(wrapLabel('23 blocks', 14, 90), ['23 blocks']);
  const {lines} = fitLabels(['Supercalifragilisticexpialidocious'], 60);
  assert.match(lines[0][0], /…$/);
});
