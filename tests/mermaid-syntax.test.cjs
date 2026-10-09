const assert = require('node:assert/strict');
const { test } = require('node:test');
const { readFileSync, mkdtempSync, writeFileSync, rmSync } = require('node:fs');
const { tmpdir } = require('node:os');
const { join, resolve } = require('node:path');
const { spawnSync } = require('node:child_process');
const { extractDiagrams, validateMarkdown } = require('../scripts/check_mermaid.cjs');

test('extracts diagrams inside details, quotes and lists while ignoring code examples', () => {
  const source = [
    '# Example', '<details>', '<summary>Answer</summary>', '',
    '```mermaid', 'flowchart TD', 'A --> B', '```', '</details>', '',
    '````markdown', '```mermaid', 'this is an example, not a diagram', '```', '````', '',
    '~~~mermaid', 'sequenceDiagram', 'A->>B: hello', '~~~', '',
    '> ```mermaid', '> flowchart LR', '> X --> Y', '> ```', '',
    '- Item', '', '  ```mermaid', '  flowchart TD', '  C --> D', '  ```',
  ].join('\n');
  const diagrams = extractDiagrams(source);
  assert.equal(diagrams.length, 4);
  assert.equal(diagrams[0].line, 5);
  assert.ok(diagrams.every(diagram => diagram.closed));
  assert.ok(diagrams.every(diagram => !diagram.source.includes('not a diagram')));
});

test('valid flowcharts, quoted punctuation and runtime sequence diagrams parse', async () => {
  const source = '```mermaid\nflowchart TD\nA["PD (Kp/Kd)"] --> B["电机"]\n```\n' +
    '<details>\n\n```mermaid\nsequenceDiagram\nparticipant P as train.py\nP->>P: PPO update\n```\n</details>';
  assert.deepEqual(await validateMarkdown(source), { count: 2, errors: [] });
});

test('rejects the merged Graph-MambaNav reserved-word node regression', async () => {
  const invalid = await validateMarkdown('```mermaid\nflowchart TB\nA --> graph["Graph-Mamba"]\n```');
  assert.equal(invalid.errors.length, 1);
  assert.match(invalid.errors[0].message, /Parse error/);
  const fixed = readFileSync(resolve(__dirname, '../wiki/entities/paper-sa-2608-13723-graph-mambanav.md'), 'utf8');
  assert.equal((await validateMarkdown(fixed)).errors.length, 0);
});

test('rejects unquoted parentheses, bad sequence syntax and unknown diagram types', async () => {
  for (const diagram of ['flowchart TD\nA[Policy (actor)] --> B', 'sequenceDiagram\nA=>B: hello', 'flowchartTYPO TD\nA --> B']) {
    assert.equal((await validateMarkdown(`\`\`\`mermaid\n${diagram}\n\`\`\``)).errors.length, 1);
  }
});

test('rejects empty and unclosed fences instead of silently skipping them', async () => {
  assert.match((await validateMarkdown('```mermaid\n```')).errors[0].message, /Empty/);
  const unclosed = await validateMarkdown('# Page\n\n```mermaid\nflowchart TD\nA --> B');
  assert.equal(unclosed.errors[0].line, 3);
  assert.match(unclosed.errors[0].message, /Unclosed/);
});

test('rejects tilde fences and placeholders that the site cannot render', async () => {
  assert.match((await validateMarkdown('~~~mermaid\nflowchart TD\nA --> B\n~~~')).errors[0].message, /alternate/);
  assert.match((await validateMarkdown('§§§mermaid\nflowchart TD\nA --> B\n§§§')).errors[0].message, /placeholder/);
  assert.equal((await validateMarkdown('````markdown\n§§§mermaid\n```mermaid\nexample\n```\n````')).errors.length, 0);
});

test('site and parser use the same exact Mermaid version', () => {
  const version = require('../package.json').devDependencies.mermaid;
  assert.match(version, /^\d+\.\d+\.\d+$/);
  const site = readFileSync(resolve(__dirname, '../docs/main.js'), 'utf8');
  assert.ok(site.includes(`mermaid@${version}/dist/mermaid.min.js`));
});

test('CLI exits nonzero and annotates invalid Mermaid for GitHub Actions', () => {
  const directory = mkdtempSync(join(tmpdir(), 'mermaid-ci-'));
  try {
    const file = join(directory, 'invalid.md');
    writeFileSync(file, '# Invalid\n\n```mermaid\nflowchart TD\nA --> graph\n```');
    const result = spawnSync(process.execPath, [resolve(__dirname, '../scripts/check_mermaid.cjs'), file], {
      encoding: 'utf8', env: { ...process.env, GITHUB_ACTIONS: 'true' },
    });
    assert.equal(result.status, 1);
    assert.match(result.stderr, /::error file=.*invalid\.md,line=3,title=Mermaid syntax::/);
    writeFileSync(file, '```mermaid\nflowchart TD\nA --> B\n```');
    const valid = spawnSync(process.execPath, [resolve(__dirname, '../scripts/check_mermaid.cjs'), file], { encoding: 'utf8' });
    assert.equal(valid.status, 0, valid.stderr);
  } finally {
    rmSync(directory, { recursive: true, force: true });
  }
});
