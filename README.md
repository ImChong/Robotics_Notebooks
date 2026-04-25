# Robotics_Notebooks

机器人技术栈知识库 / Robotics research and engineering wiki.

<!-- Last updated: 2026-04-25 (V21 自动更新：图谱 176 节点 984 边) -->

[![GitHub Pages](https://img.shields.io/badge/GitHub%20Pages-Live-brightgreen?logo=github)](https://imchong.github.io/Robotics_Notebooks/)
[![Deploy GitHub Pages](https://github.com/ImChong/Robotics_Notebooks/actions/workflows/pages.yml/badge.svg)](https://github.com/ImChong/Robotics_Notebooks/actions/workflows/pages.yml)
[![License](https://img.shields.io/github/license/ImChong/Robotics_Notebooks)](./LICENSE)
[![Knowledge Graph](https://img.shields.io/badge/知识图谱-176节点_984边-blue?logo=d3.js)](https://imchong.github.io/Robotics_Notebooks/graph.html)
[![Sources Coverage](https://img.shields.io/badge/sources覆盖率-100%25-green)](docs/checklists/tech-stack-next-phase-checklist-v21.md)



---

## 适合谁

想系统学人形机器人运动控制 / 强化学习 / 模仿学习，有一定编程基础（Python / C++） and 本科数学基础。

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
| `wiki/` | **结构化知识库**。包含 Concepts, Methods, Tasks 等核心页面。 |
| `roadmap/` | **成长路线**。规划了从基础到进阶的系统学习路径。 |
| `tech-map/` | **技术地图**。展示模块间依赖关系与技术栈全景。 |
| `sources/` | **原始资料**。Ingest 之前的原始论文摘录、GitHub 仓库导航。 |
| `references/` | **论文/Repo 索引**。按主题分类的深度阅读资源。 |
| `scripts/` | **维护工具**。用于 lint、搜索、索引生成和统计更新。 |
| `docs/` | **展示层**。GitHub Pages 托管的 D3.js 交互式图谱与详情页。 |

---

## 如何贡献

1.  **Ingest 模式**：如果你发现好的论文或 Repo，按照 `schema/ingest-workflow.md` 加入 `sources/`。
2.  **Wiki 完善**：将 `sources/` 提炼为 `wiki/` 页面，并建立 [Link](schema/linking.md)。
3.  **Lint 检查**：运行 `make lint` 确保没有断链或孤儿页。

---

## 许可证

本项目采用 [MIT License](LICENSE)。
