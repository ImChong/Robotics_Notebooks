#!/usr/bin/env python3
"""统计 wiki/roadmap/tech-map 节点的 Mermaid 图与源码运行时序图覆盖率。

口径：
- 节点 = wiki/、roadmap/、tech-map/ 下的 *.md
- 有 Mermaid = 正文含 ```mermaid 围栏
- 有源码 = frontmatter 的 sources: 引用了 sources/repos/
- 有时序图 = 含以 sequenceDiagram 开头的 mermaid 块
- 「不适用」= 页内「源码运行时序图」标题后 300 字内出现「不适用」
用法：python3 scripts/audit_diagram_coverage.py [--list missing-mermaid|missing-seq]
"""
import argparse, glob, re
from collections import Counter

FENCE = re.compile(r'^```mermaid\s*\n(.*?)^```', re.S | re.M)
NA = re.compile(r'源码运行时序图[\s\S]{0,300}?不适用')

def scan():
    rows = []
    for root in ('wiki', 'roadmap', 'tech-map'):
        for f in sorted(glob.glob(f'{root}/**/*.md', recursive=True)):
            t = open(f, encoding='utf-8').read()
            m = re.match(r'---\n(.*?)\n---', t, re.S)
            fm = m.group(1) if m else ''
            blocks = FENCE.findall(t)
            rows.append(dict(
                f=f, mermaid=bool(blocks),
                seq=any(b.lstrip().startswith('sequenceDiagram') for b in blocks),
                na=bool(NA.search(t)), repo='sources/repos/' in fm,
                paper=f.startswith('wiki/entities/paper-')))
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--list', choices=['missing-mermaid', 'missing-seq'])
    a = ap.parse_args()
    rows = scan()
    if a.list:
        for r in rows:
            if (a.list == 'missing-mermaid' and not r['mermaid']) or \
               (a.list == 'missing-seq' and r['repo'] and not r['seq']):
                print(r['f'])
        return
    print(f"节点总数 {len(rows)}；有 Mermaid {sum(r['mermaid'] for r in rows)}；无 Mermaid {sum(not r['mermaid'] for r in rows)}")
    by, miss = Counter(), Counter()
    for r in rows:
        p = r['f'].split('/')
        k = '/'.join(p[:2]) if p[0] == 'wiki' and len(p) > 2 else p[0]
        by[k] += 1; miss[k] += not r['mermaid']
    for k in sorted(by): print(f'  {k}: {by[k]}，无 Mermaid {miss[k]}')
    src = [r for r in rows if r['repo']]
    print(f"有源码节点 {len(src)}；有时序图 {sum(r['seq'] for r in src)}；无时序图 {sum(not r['seq'] for r in src)}"
          f"（其中已标「不适用」{sum(not r['seq'] and r['na'] for r in src)}）")
    ps = [r for r in src if r['paper']]
    print(f"其中论文实体页 {len(ps)}；无时序图 {sum(not r['seq'] for r in ps)}"
          f"（已标不适用 {sum(not r['seq'] and r['na'] for r in ps)}）")

if __name__ == '__main__':
    main()
