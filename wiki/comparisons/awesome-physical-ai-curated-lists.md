---
type: comparison
tags: [curated-list, physical-ai, embodied-ai, vla]
topic: [ecosystem]
status: complete
updated: 2026-09-20
related:
  - ../entities/awesome-physical-ai-natnew.md
  - ../entities/awesome-physical-ai-aichr.md
  - ../methods/vla.md
  - ../concepts/sim2real.md
  - ../overview/awesome-physical-ai-technology-map.md
  - ../queries/embodied-fm-taxonomy-loop.md
original:
  - ../../sources/repos/awesome_physical_ai_curated_lists_reading_pc7k.md
sources:
  - ../../sources/repos/awesome-physical-ai-natnew.md
  - ../../sources/repos/awesome-physical-ai-aichr.md
  - ../../sources/sites/awesome-physical-ai-natnew-github-io.md
summary: "同名 GitHub 仓 natnew vs aichr 的 Physical AI 策展清单选型：taxonomy、文档站、边缘硬件与维护体量对照。"
---

# Physical AI 策展清单：natnew vs aichr

GitHub 上有两个都叫 **awesome-physical-ai** 的独立仓库，阅读与引用时按 **org 前缀** 区分，别混成一个。

## 一句话定义

**同名异仓对照** — natnew 版偏 **14 类工程 taxonomy + Pages 导航**；aichr 版偏 **VLA/3D/边缘硬件宽口径 CC0 索引**。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Physical AI | Physical Artificial Intelligence | 两清单共同主题 |
| VLA | Vision-Language-Action | 两清单均重点覆盖 |
| CC0 | Creative Commons Zero | aichr 列表许可证 |
| MIT | Massachusetts Institute of Technology License | natnew 仓库许可证 |
| RFM | Robotics Foundation Model | natnew canonical 类之一 |

## 为什么重要

两个同名仓库在检索中极易混淆；引用错仓会把 Safety / Governance 覆盖或边缘硬件覆盖张冠李戴。先分清收录口径，再决定把哪份清单当条目入口。

## 核心原理

两份清单的差别在**组织方式**：natnew 先定 14 个 canonical 类别再往里填条目（并带 docs 站与 tags 注释）；aichr 按技术栈（VLA、3D 视觉、边缘推理、硬件）扁平罗列并用 Legend 图标标注论文 / 代码 / 硬件验证。前者便于按主题系统浏览，后者便于按部署链路速查。（依据：2026-11 通读两份 README，见 `original`。）

## 工程实践

- 引用具体条目时回到条目自身的官方页核对，清单只做入口；两份 README 的描述均为简述，未逐条验证。
- 站内写链接必须带 `natnew` / `aichr` 前缀，避免实体页混用。
- 需要 Safety / Governance / Evaluation 视角时优先查 natnew；需要 Jetson / TensorRT / 触觉传感器视角时优先查 aichr。

## 局限

- 两份均为第三方策展，无统一质量门槛；aichr 体量小、更新节奏与 stars 都低（见对比表）。
- 条目数、stars 等为快照（2026-09-19/20），会漂移；本页分类对照是基于 README 结构的人工归纳，属推断。
- 清单收录不等于推荐，亦不代表条目已被复现。

## 对比表

| 维度 | [natnew/awesome-physical-ai](../entities/awesome-physical-ai-natnew.md) | [aichr/awesome-physical-ai](../entities/awesome-physical-ai-aichr.md) |
|------|------------------------------------------------------------------------|-----------------------------------------------------------------------|
| 维护者 | natnew（个人） | aichr（GitHub org） |
| Stars（2026-09-19） | ~142 | ~5 |
| 列表许可 | MIT | CC0-1.0 |
| 条目规模 | ~229（14 canonical 类） | 较小、按 VLA/3D/Sim 等 ~12 节 |
| 文档站 | ✅ [GitHub Pages Overview](https://natnew.github.io/awesome-physical-ai/docs/overview) | ❌ 仅 README |
| Taxonomy | 14 类 + CONTRIBUTING/CI | README 扁平章节 |
| 特色覆盖 | Safety、Governance、Production、分级 Practice 项目 | Edge AI、Hardware、ROS 2、Research Labs |
| Quick start | Gymnasium → MuJoCo → LeRobot → OpenVLA | 无 staged 路径 |
| 自动化 | issue triage workflow | 标准 CONTRIBUTING |


## 清单条目覆盖（2026-09-20）

两份清单合并去重后共 **384** 条，每条各有一页可点开（两清单同时出现 33 条）。

完整条目表见 [Physical AI 技术地图](../overview/awesome-physical-ai-technology-map.md)；并集目录见 [awesome-physical-ai-union-catalog.md](../../sources/repos/awesome-physical-ai-union-catalog.md)。

## 怎么选

- **系统学习 / 长期跟踪 / 生产视角** → **natnew**：taxonomy 与 docs 站适合按 Sim2Real、Safety、Benchmark 系统浏览。
- **快速扫 VLA + 边缘部署 + 硬件** → **aichr**：工业模型与 Jetson/TensorRT 条目集中。
- **不互斥：** 两清单 URL 不同，可 bookmark 两个 org；站内链接务必写全 `natnew` / `aichr` 实体页。

## 关联页面

- [Physical AI 技术地图](../overview/awesome-physical-ai-technology-map.md)
- [awesome-physical-ai（natnew）](../entities/awesome-physical-ai-natnew.md)
- [awesome-physical-ai（aichr）](../entities/awesome-physical-ai-aichr.md)
- [VLA](../methods/vla.md)
- [Sim2Real](../concepts/sim2real.md)
- [Query：具身大模型分类学选型闭环](../queries/embodied-fm-taxonomy-loop.md) — 本页比的是两份**第三方清单**的收录口径，两边的分类都随各自维护者口味漂移；要按 VLM→VLN→VLA→VLX→WM 的家族分层定选型，以那里的库内 canonical taxonomy 为准，清单只当条目入口

## 参考来源

- [sources/repos/awesome-physical-ai-natnew.md](../../sources/repos/awesome-physical-ai-natnew.md)
- [sources/repos/awesome-physical-ai-aichr.md](../../sources/repos/awesome-physical-ai-aichr.md)
- [sources/sites/awesome-physical-ai-natnew-github-io.md](../../sources/sites/awesome-physical-ai-natnew-github-io.md)

## 推荐继续阅读

- natnew：<https://github.com/natnew/awesome-physical-ai>
- aichr：<https://github.com/aichr/awesome-physical-ai>
