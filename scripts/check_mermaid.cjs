#!/usr/bin/env node
// Validate published Markdown with the same pinned Mermaid parser as the site.
const { readFileSync, readdirSync } = require('node:fs');
const { resolve, relative, join } = require('node:path');
const MarkdownIt = require('markdown-it');
const { JSDOM } = require('jsdom');

const ROOT = resolve(__dirname, '..');
// html:false also discovers fences inside <details>, as docs/main.js does.
const markdown = new MarkdownIt({ html: false });
let parserPromise;

function extractDiagrams(source) {
  const lines = source.split('\n');
  return markdown.parse(source, {}).filter(token =>
    token.type === 'fence' && token.info.trim().split(/\s+/)[0].toLowerCase() === 'mermaid'
  ).map(token => {
    const lastLine = lines[token.map[1] - 1] || '';
    const marker = token.markup[0];
    const closed = new RegExp(`^[\\s>]*${marker}{${token.markup.length},}\\s*$`).test(lastLine);
    return { source: token.content, line: token.map[0] + 1, closed, markup: token.markup };
  });
}

async function getParser() {
  if (!parserPromise) {
    // Mermaid sanitizes labels via DOMPurify even during parse; no browser/network needed.
    const { window } = new JSDOM('');
    globalThis.window = window;
    globalThis.document = window.document;
    parserPromise = import('mermaid').then(({ default: mermaid }) => {
      mermaid.initialize({ startOnLoad: false, securityLevel: 'strict' });
      return mermaid;
    });
  }
  return parserPromise;
}

async function validateMarkdown(source) {
  const diagrams = extractDiagrams(source);
  const errors = [];
  const codeLines = new Set();
  for (const token of markdown.parse(source, {})) {
    if (token.type !== 'fence' && token.type !== 'code_block') continue;
    for (let line = token.map[0]; line < token.map[1]; line++) codeLines.add(line);
  }
  source.split('\n').forEach((line, index) => {
    if (!codeLines.has(index) && /^\s*§{3,}mermaid\s*$/i.test(line)) {
      errors.push({ line: index + 1, message: 'Replace Mermaid placeholder with ```mermaid' });
    }
  });
  const mermaid = await getParser();
  // Sequential: Mermaid maintains shared parser/configuration state.
  for (const diagram of diagrams) {
    if (diagram.markup !== '```') {
      errors.push({ line: diagram.line, message: 'Use ```mermaid fences; the site does not support alternate Mermaid fences' });
      continue;
    }
    if (!diagram.closed || !diagram.source.trim()) {
      errors.push({ line: diagram.line, message: diagram.closed ? 'Empty Mermaid diagram' : 'Unclosed Mermaid fence' });
      continue;
    }
    try {
      await mermaid.parse(diagram.source);
    } catch (error) {
      errors.push({ line: diagram.line, message: String(error.message || error) });
    }
  }
  return { count: diagrams.length, errors };
}

function markdownFiles(directory) {
  return readdirSync(directory, { withFileTypes: true }).flatMap(entry => {
    const path = join(directory, entry.name);
    return entry.isDirectory() ? markdownFiles(path) : entry.name.endsWith('.md') ? [path] : [];
  }).sort();
}

async function main(paths = process.argv.slice(2)) {
  const files = paths.length ? paths.map(path => resolve(path)) :
    ['wiki', 'roadmap', 'tech-map', 'references'].flatMap(dir => markdownFiles(join(ROOT, dir)));
  let count = 0;
  let failures = 0;
  for (const file of files) {
    const result = await validateMarkdown(readFileSync(file, 'utf8'));
    count += result.count;
    for (const error of result.errors) {
      failures++;
      const path = relative(ROOT, file);
      console.error(`${path}:${error.line}: ${error.message}`);
      if (process.env.GITHUB_ACTIONS === 'true') {
        const escape = value => String(value).replace(/%/g, '%25').replace(/\r/g, '%0D').replace(/\n/g, '%0A');
        const property = value => escape(value).replace(/:/g, '%3A').replace(/,/g, '%2C');
        console.error(`::error file=${property(path)},line=${error.line},title=Mermaid syntax::${escape(error.message)}`);
      }
    }
  }
  console.log(`Mermaid ${require('mermaid/package.json').version}: ${count} diagrams in ${files.length} Markdown files; ${failures} errors.`);
  return failures ? 1 : 0;
}

module.exports = { extractDiagrams, validateMarkdown, main };
if (require.main === module) main().then(code => { process.exitCode = code; }).catch(error => {
  console.error(error);
  process.exitCode = 1;
});
