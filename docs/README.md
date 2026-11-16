# Docs 目录说明

`docs/` 是本知识库的 GitHub Pages 静态站点根目录，另包含维护团队的内部看板与质量审计记录（部署打包时由 [pages.yml](../.github/workflows/pages.yml) 剔除，不发布到线上站点）。

## 站点页面入口

| 页面 | 用途 |
|---|---|
| `index.html` | 首页：最新节点动态、研究机构与赛道入口、全站搜索 |
| `graph.html` | 2D / 3D 交互式知识图谱、社区筛选与时间线演化 |
| `detail.html` | 按需渲染 `wiki/` 与 `references/` 单页正文与关系图 |
| `roadmap.html` | 学习主路线与专题纵深路线图、知识地图进度视图 |
| `hubs.html` | 全站 20 个社区主题排行榜与开源项目覆盖导览 |
| `star-history.html` | 开源仓库星标历史与机构维度对比分析 |

## 本地预览

站点 JSON 与搜索索引不入库，首次本地预览前先生成派生数据：

```bash
make export graph
cd docs && python3 -m http.server 8080
```

## 维护看板与归档（仅仓内保留）

- [技术栈执行清单 v31](checklists/tech-stack-next-phase-checklist-v31.md) — 当前演进阶段看板
- [前端体验优化清单 v1](checklists/frontend-optimization-v1.md) — 展示层交互优化追踪
- [历史执行清单索引](checklists/README.md) — 历史阶段归档 v1–v30
- [展示层变更记录](change-log.md) — 站点演进里程碑
- [历史计划与日志归档](plans/archive/) — 已完成前端改造计划与按季度切分的历史日志
