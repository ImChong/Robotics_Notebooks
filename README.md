# Robotics_Notebooks

机器人技术栈知识库 / Robotics research and engineering wiki.

<!-- Last updated: 2026-04-21 (V21 启动：图谱 170 节点 941 边) -->

[![GitHub Pages](https://img.shields.io/badge/GitHub%20Pages-Live-brightgreen?logo=github)](https://imchong.github.io/Robotics_Notebooks/)
[![Deploy GitHub Pages](https://github.com/ImChong/Robotics_Notebooks/actions/workflows/pages.yml/badge.svg)](https://github.com/ImChong/Robotics_Notebooks/actions/workflows/pages.yml)
[![License](https://img.shields.io/github/license/ImChong/Robotics_Notebooks)](./LICENSE)
[![Knowledge Graph](https://img.shields.io/badge/知识图谱-170节点_941边-blue?logo=d3.js)](https://imchong.github.io/Robotics_Notebooks/graph.html)
[![Sources Coverage](https://img.shields.io/badge/sources覆盖率-99%25-green)](docs/checklists/tech-stack-next-phase-checklist-v21.md)



---

## 适合谁

想系统学人形机器人运动控制 / 强化学习 / 模仿学习，有一定编程基础（Python / C++）和本科数学基础。

不知道从哪开始？直接看 [运动控制成长路线](roadmap/route-a-motion-control.md)。

---

## 这个项目是什么

`Robotics_Notebooks` 是一个**机器人工程知识库**，不是资源收集箱。它把机器人技术栈拆成互联的知识页面——每个概念解释清楚是什么、为什么重要、和哪些概念相关。

**不是**：教科书（不系统讲理论）、笔记堆（有结构和依赖关系）、工具文档。

---

## 从哪里开始

| 你的目标 | 入口 |
|---------|------|
| 可视化探索知识图谱 | [知识图谱](https://imchong.github.io/Robotics_Notebooks/graph.html) |
| 有一条路线照着走 | [运动控制成长路线](roadmap/route-a-motion-control.md) |
| 用强化学习做运动控制 | [强化学习运动控制路径](roadmap/learning-paths/if-goal-locomotion-rl.md) |
| 学模仿学习与技能迁移 | [模仿学习路径](roadmap/learning-paths/if-goal-imitation-learning.md) |
| 学安全控制（CLF/CBF）| [安全控制路径](roadmap/learning-paths/if-goal-safe-control.md) |
| 做接触丰富的操作任务 | [接触操作路径](roadmap/learning-paths/if-goal-contact-manipulation.md) |
| 浏览所有知识页 | [知识页导航总入口](index.md) |
| 搜索特定概念 | `python3 scripts/search_wiki.py <关键词>` |

---

## 项目结构

| 目录 | 用途 |
|------|------|
| `wiki/` | 结构化知识页（概念 / 方法 / 任务 / 对比 / 实体） |
| `roadmap/` | 成长路线与学习路径 |
| `tech-map/` | 技术栈模块依赖关系图 |
| `sources/` | 原始资料输入层（论文 / 博客 / 课程） |
| `references/` | 深挖入口（按主题整理的论文 / repo / benchmark） |
| `schema/` | 知识库维护规则（ingest 流程 / 页面类型规范） |
| `scripts/` | 自动化工具（lint / search / catalog / export） |
| `docs/` | 网站前端 + 执行清单 + 变更记录 |

---

## 知识库维护方法论

本项目采用 [Karpathy 知识库方法论](wiki/references/llm-wiki-karpathy.md)：三层架构（sources → wiki → schema），LLM 负责维护和交叉引用，人类负责资料筛选和方向判断。

常用操作：
```bash
make lint                                 # 健康检查（0 issues 为目标）
make search Q=<关键词>                    # 搜索
make catalog                             # 刷新 index.md
make export                              # 更新前端 JSON（110 页）
make graph                               # 更新知识图谱 + graph-stats.json
make vectors                             # 构建向量索引（V10，需 sentence-transformers）
make anki                                # 导出 Anki 闪卡（V10，exports/anki-flashcards.tsv）
make ingest NAME=<stem> TITLE="..." DESC="..."  # 生成 sources/papers/ 模板
python3 scripts/search_wiki.py <关键词> --related   # 搜索 + 显示关联页面
python3 scripts/search_wiki.py <关键词> --semantic  # 混合 BM25 + 向量搜索（V10）
```

维护操作规范见 [维护操作规范](schema/ingest-workflow.md)。

---

## 知识图谱

`docs/graph.html` 是 Obsidian 风格的知识图谱可视化，由 D3.js 力导向算法驱动：

- **147 个节点**：概念、方法、任务、实体、对比、Query 产物、形式化定义
- **842 条边**：wiki 页面间的内链关系
- 节点支持按类型 / 按社区着色，大小反映连接度（入度 + 出度）
- 悬停 / 点击显示浮动卡片，点击卡片内"打开详情页"跳转对应页面
- 社区模式会在 legend 与右侧详情侧边栏显示社区名称（按社区 hub 命名）
- 支持按类型过滤（含"孤儿"模式）、关键词搜索 + fly-to 定位、缩放平移
- 物理参数调节（排斥力 / 节点大小 / 连接线粗细 / 字体大小）
- 支持亮色 / 暗色主题切换，移动端触控优化

更新图谱数据：`make graph`（自动重新生成并同步至 docs/exports/，含 graph-stats.json）

---

## 执行清单

[技术栈项目执行清单 v21](docs/checklists/tech-stack-next-phase-checklist-v21.md) — 当前阶段目标（触觉力觉闭环 / 通信链路形式化 / 详情页微地图）

历史版本：[执行清单归档目录](docs/checklists/)
cs/checklists/)
