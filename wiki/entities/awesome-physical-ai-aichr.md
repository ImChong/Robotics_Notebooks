---
type: entity
tags: [curated-list, physical-ai, embodied-ai, vla, edge-ai, ros]
topic: [ecosystem]
status: complete
updated: 2026-11-06
related:
  - ./awesome-physical-ai-natnew.md
  - ../comparisons/awesome-physical-ai-curated-lists.md
  - ../methods/vla.md
  - ../overview/awesome-physical-ai-technology-map.md
  - ./lerobot.md
  - ./isaac-lab.md
sources:
  - ../../sources/repos/awesome-physical-ai-aichr.md
original:
  - ../../sources/repos/awesome_physical_ai_aichr_readme_reading_ch7q.md
summary: "aichr 组织维护的 Physical AI 策展清单：VLA、3D 视觉、仿真、边缘推理、硬件与 ROS 框架；CC0 列表。"
---

# awesome-physical-ai（aichr）

[`aichr/awesome-physical-ai`](https://github.com/aichr/awesome-physical-ai) 是 **aichr** 组织维护的 Physical AI 资源清单：以 **VLA / 3D 视觉 / 仿真 / 边缘部署 / 硬件** 为主轴，用 📄/💻/🤖 legend 标注条目类型。

## 一句话定义

**Physical AI 宽口径资源索引** — 覆盖 GR00T、π0、Open X-Embodiment、Isaac Lab 与 Jetson/TensorRT 边缘栈的 CC0 Markdown 列表。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLA | Vision-Language-Action | Foundation Models 区主线 |
| Physical AI | Physical Artificial Intelligence | 机器人×ML×具身智能交叉 |
| ROS | Robot Operating System | Frameworks 区 ROS 2 / ros2_ai |
| 3DGS | 3D Gaussian Splatting | 3D Vision 区 NeRF/GS 论文链 |
| OXE | Open X-Embodiment | Datasets 区大规模多本体数据 |

## 为什么重要

- **边缘与硬件显式单列：** Edge AI & Inference、Hardware & Actuation 区集中 Jetson、TensorRT、EtherCAT、触觉传感，补全 **natnew 清单** 偏软件/catalog 的视角。
- **工业 VLA 入口快：** GR00T、π0、Gemini Robotics、Octo 等同屏，适合 **选型扫一眼**。
- **CC0 列表：** 鼓励 fork/重组；与 natnew MIT 仓许可证不同，引用时注意 **条目上游 License**。


## 核心机制

- **组织方式：** README 以 12 个主题章节平铺（Contents 列出），章内再分小节（如 VLA 分 Industry Models / Academic Papers），条目为「名称 + 链接 — 一句话说明」，无打分或分类标签体系。
- **图例（Legend）：** 📄 Paper、🎬 Demo Video、💻 Open Source Code、🤖 Hardware-Verified；其中 🤖 表示「硬件验证」，并非泛指机器人条目。
- **与 natnew 版的结构差异（推断）：** 本清单约 250 行、章节少、偏资源入口；natnew 版有 14 类 taxonomy 与更长注释。

## 工程实践

- **选型扫读：** 先看 Foundation Models 与 Simulators 两章，再按部署目标跳到 Edge AI（Jetson / TensorRT / llama.cpp 量化）与 Hardware（EtherCAT、ROS 2 Control）。
- **入门路径（推断，README 未明示）：** Gymnasium / MuJoCo / Isaac Lab 仿真 → LeRobot 训练 → 边缘部署；课程章列 Stanford CS 224R 与 LeRobot 教程。
- **引用前核对：** 条目多为单行描述，规模、许可与指标（如 Open X-Embodiment 的 1M+ 轨迹）须回到上游页面确认。

## 清单条目覆盖（2026-09-20）

两份清单合并去重后共 **384** 条，每条各有一页可点开（两清单同时出现 33 条）。

完整条目表见 [Physical AI 技术地图](../overview/awesome-physical-ai-technology-map.md)；并集目录见 [awesome-physical-ai-union-catalog.md](../../sources/repos/awesome-physical-ai-union-catalog.md)。

## 核心结构

Foundation Models (VLA) · 3D Computer Vision · Datasets · Benchmarks · Simulators · Edge AI · Hardware · Frameworks · Robot Platforms · Research Labs · Courses · Community

## 局限与使用注意

- **体量与 stars 较小：** 2026-09-19 约 5 stars；更新频率与覆盖需与 [natnew 版](./awesome-physical-ai-natnew.md) 对照。
- **无独立 docs 站：** 全量在 README；无 14 类 CI taxonomy。
- **同名易混：** GitHub 上另有 [`natnew/awesome-physical-ai`](https://github.com/natnew/awesome-physical-ai) — 见 [对比页](../comparisons/awesome-physical-ai-curated-lists.md)。

## 关联页面

- [Physical AI 技术地图](../overview/awesome-physical-ai-technology-map.md)
- [awesome-physical-ai（natnew）](./awesome-physical-ai-natnew.md)
- [Physical AI 策展清单对比](../comparisons/awesome-physical-ai-curated-lists.md)
- [VLA](../methods/vla.md)
- [Isaac Lab](./isaac-lab.md)

## 参考来源

- [sources/repos/awesome-physical-ai-aichr.md](../../sources/repos/awesome-physical-ai-aichr.md)

## 推荐继续阅读

- GitHub：<https://github.com/aichr/awesome-physical-ai>
- [Physical AI 策展清单对比](../comparisons/awesome-physical-ai-curated-lists.md)
