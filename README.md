# Robotics_Notebooks

机器人技术栈知识库 / Robotics research and engineering wiki.

<!-- Last updated: 2026-07-14 (V29 自动更新：图谱 1632 节点 12619 边) -->

[![GitHub Pages](https://img.shields.io/badge/GitHub%20Pages-Live-brightgreen?logo=github)](https://imchong.github.io/Robotics_Notebooks/)
[![Deploy GitHub Pages](https://github.com/ImChong/Robotics_Notebooks/actions/workflows/pages.yml/badge.svg)](https://github.com/ImChong/Robotics_Notebooks/actions/workflows/pages.yml)
[![Wiki Lint](https://github.com/ImChong/Robotics_Notebooks/actions/workflows/lint.yml/badge.svg)](https://github.com/ImChong/Robotics_Notebooks/actions/workflows/lint.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![Knowledge Graph](https://img.shields.io/badge/知识图谱-1632节点_12619边-blue?logo=d3.js)](https://imchong.github.io/Robotics_Notebooks/graph.html)
[![Sources Coverage](https://img.shields.io/badge/sources覆盖率-98%25-green)](docs/checklists/tech-stack-next-phase-checklist-v29.md)

---

## 在线演示

[![知识图谱交互演示：1564 个知识节点按技术社区着色，悬停节点查看简介，滚轮缩放，并可切换 3D 立体视图](media/graph-demo.gif)](https://imchong.github.io/Robotics_Notebooks/graph.html)

↑ [交互式知识图谱](https://imchong.github.io/Robotics_Notebooks/graph.html)：每个点是一个知识页，颜色代表所属技术社区，连线是页面间的引用关系。悬停看简介与关联边，点击进详情页，支持 2D / 3D 视图切换。

---

## 适合谁

想系统学人形机器人运动控制 / 强化学习 / 模仿学习，有一定编程基础（Python / C++）与本科数学基础。

不知道从哪开始？直接看 [运动控制成长路线](roadmap/motion-control.md)。

---

## 这个项目是什么

`Robotics_Notebooks` 是一个**机器人工程知识库**，不是资源收集箱。它把机器人技术栈拆成互联的知识页面——每个概念解释清楚是什么、为什么重要、和哪些概念相关。

**不是**：教科书（不系统讲理论）、笔记堆（有结构和依赖关系）、工具文档。

---

## 从哪里开始

| 你的目标 | 入口 |
|---------|------|
| 可视化探索知识图谱 | [知识图谱](https://imchong.github.io/Robotics_Notebooks/graph.html) |
| 有一条路线照着走 | [运动控制成长路线](roadmap/motion-control.md) |
| 设计力矩控制关节电机 | [力矩电机设计纵深路线](roadmap/depth-torque-motor-design.md) |
| 学传统模型控制（MPC/WBC）| [传统控制纵深路线](roadmap/depth-classical-control.md) |
| 学安全控制（CLF/CBF）| [安全控制纵深路线](roadmap/depth-safe-control.md) |
| 做接触丰富的操作任务 | [接触操作纵深路线](roadmap/depth-contact-manipulation.md) |
| 让机器人自主从 A 到 B | [导航纵深路线](roadmap/depth-navigation.md) |
| 学模仿学习与技能迁移 | [模仿学习纵深路线](roadmap/depth-imitation-learning.md) |
| 用强化学习做运动控制 | [RL 纵深路线](roadmap/depth-rl-locomotion.md) |
| 让机器人边走边动手 | [Loco-Manipulation 纵深路线](roadmap/depth-loco-manipulation.md) |
| 把人体动作变成机器人参考轨迹 | [动作重定向纵深路线](roadmap/depth-motion-retargeting.md) |
| 做人形全身行为基础模型 | [BFM 纵深路线](roadmap/depth-bfm.md) |
| 让机器人看地形越障 | [感知越障纵深路线](roadmap/depth-perceptive-locomotion.md) |
| 用生成模型造人形动作 | [动作生成纵深路线](roadmap/depth-motion-generation.md) |
| 让机器人听懂指令干活 | [VLA 纵深路线](roadmap/depth-vla.md) |
| 让策略预知动作如何改变世界 | [WAM 纵深路线](roadmap/depth-wam.md) |
| 浏览所有知识页 | [完整页面目录](catalog.md) |
| 搜索特定概念 | `python3 scripts/search_wiki.py <关键词>` |

> 十四条纵深路线按各方向**起点里程碑的历史顺序**排列（与首页按钮一致）：力矩电机设计（磁场定向控制 FOC，1971）→ 传统控制（ZMP 判据，1972）→ 安全控制（CLF，1983）→ 接触操作（阻抗控制，1985）→ 导航（概率 SLAM，1986）→ 模仿学习（行为克隆，1988）→ 强化学习（Q-learning，1989）→ 移动操作（移动操作臂协调控制，1994）→ 动作重定向（Gleicher 动作重定向，1998）→ BFM（DeepMimic 动作跟踪谱系，2018）→ 感知越障（2020s 感知策略浪潮）→ 动作生成（MDM 扩散动作生成，2022）→ VLA（RT-2 确立 VLA，2023）→ WAM（World Action Models 综述形式化，2026）。越靠前的方向理论积淀越深，越靠后的方向越依赖学习方法与算力。

---

## 项目结构

| 目录 | 用途 |
|------|------|
| `wiki/` | **结构化知识库**。包含 Concepts, Methods, Tasks 等核心页面。 |
| `roadmap/` | **成长路线**。规划了从基础到进阶的系统学习路径。 |
| `tech-map/` | **技术地图**。展示模块间依赖关系与技术栈全景。 |
| `sources/` | **原始资料**。Ingest 之前的原始论文摘录、GitHub 仓库导航。 |
| `references/` | **论文/Repo 索引**。按主题分类的深度阅读资源。 |
| `schema/` | **维护规范**（ingest、命名、内链、页面类型、log 与 lint 用 JSON）。索引见 [schema/README.md](schema/README.md)。 |
| `scripts/` | **维护工具**。用于 lint、搜索、索引生成和统计更新；脚本一览见 [scripts/README.md](scripts/README.md)。 |
| `docs/` | **展示层**。GitHub Pages 托管的 D3.js 交互式图谱与详情页。 |
| `docs/checklists/` | **执行清单归档**。当前技术栈推进、前端优化与历史阶段清单。 |

不知道新资料或新页面该进哪个目录？见 [内容目录怎么选](schema/content-directories.md)。

---

## 维护看板

- 当前技术栈执行清单：[V29](docs/checklists/tech-stack-next-phase-checklist-v29.md)
- 前端体验优化清单：[frontend-optimization-v1](docs/checklists/frontend-optimization-v1.md)
- 历史执行清单索引：[docs/checklists/README.md](docs/checklists/README.md)

---

## 如何贡献

详见 **[CONTRIBUTING.md](CONTRIBUTING.md)**（流程与本地/CI 命令）。简述：

1.  **Ingest 模式**：如果你发现好的论文或 Repo，按照 `schema/ingest-workflow.md` 加入 `sources/`。
2.  **Wiki 完善**：将 `sources/` 提炼为 `wiki/` 页面，并建立 [Link](schema/linking.md)。
3.  **Lint 检查**：运行 `make lint` 确保没有断链或孤儿页。

---

## 许可证

本项目采用 [MIT License](LICENSE)。
