# Robotics_Notebooks

机器人技术栈知识库 / Robotics research and engineering wiki.

<!-- Last updated: 2026-04-14 -->

[![GitHub Pages](https://img.shields.io/badge/GitHub%20Pages-Live-brightgreen?logo=github)](https://imchong.github.io/Robotics_Notebooks/)
[![Deploy GitHub Pages](https://github.com/ImChong/Robotics_Notebooks/actions/workflows/pages.yml/badge.svg)](https://github.com/ImChong/Robotics_Notebooks/actions/workflows/pages.yml)
[![License](https://img.shields.io/github/license/ImChong/Robotics_Notebooks)](./LICENSE)

---

## 适合谁

想系统学人形机器人运动控制 / 强化学习 / 模仿学习，有一定编程基础（Python / C++）和本科数学基础。

不知道从哪开始？直接看 [路线A：运动控制成长路线](roadmap/route-a-motion-control.md)。

---

## 这个项目是什么

`Robotics_Notebooks` 是一个**机器人工程知识库**，不是资源收集箱。它把机器人技术栈拆成互联的知识页面——每个概念解释清楚是什么、为什么重要、和哪些概念相关。

**不是**：教科书（不系统讲理论）、笔记堆（有结构和依赖关系）、工具文档。

---

## 从哪里开始

| 你的目标 | 入口 |
|---------|------|
| 有一条路线照着走 | [路线A：运动控制成长路线](roadmap/route-a-motion-control.md) |
| 用 RL 做 locomotion | [RL Locomotion 学习路径](roadmap/learning-paths/if-goal-locomotion-rl.md) |
| 学模仿学习与技能迁移 | [IL 学习路径](roadmap/learning-paths/if-goal-imitation-learning.md) |
| 浏览所有知识页 | [index.md 导航总入口](index.md) |
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

本项目采用 [Karpathy LLM Wiki 模式](wiki/references/llm-wiki-karpathy.md)：三层架构（sources → wiki → schema），LLM 负责维护和交叉引用，人类负责资料筛选和方向判断。

常用操作：
```bash
python3 scripts/lint_wiki.py              # 健康检查
python3 scripts/search_wiki.py <关键词>  # 搜索
python3 scripts/generate_page_catalog.py # 刷新 index.md
python3 scripts/export_minimal.py        # 更新前端 JSON
```

维护操作规范见 [schema/ingest-workflow.md](schema/ingest-workflow.md)。

---

## 执行清单

[技术栈项目执行清单 v3](docs/tech-stack-next-phase-checklist-v3.md) — 当前阶段目标与待办

历史版本：[v2](docs/tech-stack-next-phase-checklist-v2.md) · [v1](docs/tech-stack-next-phase-checklist-v1.md)
