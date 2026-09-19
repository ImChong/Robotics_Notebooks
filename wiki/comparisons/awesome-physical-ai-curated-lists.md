---
type: comparison
tags: [curated-list, physical-ai, embodied-ai, vla]
status: complete
updated: 2026-09-19
related:
  - ../entities/awesome-physical-ai-natnew.md
  - ../entities/awesome-physical-ai-aichr.md
  - ../methods/vla.md
  - ../concepts/sim2real.md
  - ../queries/embodied-fm-taxonomy-loop.md
sources:
  - ../../sources/repos/awesome-physical-ai-natnew.md
  - ../../sources/repos/awesome-physical-ai-aichr.md
  - ../../sources/sites/awesome-physical-ai-natnew-github-io.md
summary: "同名 GitHub 仓 natnew vs aichr 的 Physical AI 策展清单选型：taxonomy、文档站、边缘硬件与维护体量对照。"
---

# Physical AI 策展清单：natnew vs aichr

GitHub 上存在两个均名为 **awesome-physical-ai** 的独立仓库，入库时须按 **org 前缀** 区分，避免链错或重复节点。

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

## 怎么选

- **系统学习 / 长期跟踪 / 生产视角** → **natnew**：taxonomy 与 docs 站适合按 Sim2Real、Safety、Benchmark 系统浏览。
- **快速扫 VLA + 边缘部署 + 硬件** → **aichr**：工业模型与 Jetson/TensorRT 条目集中。
- **不互斥：** 两清单 URL 不同，可 bookmark 两个 org；站内链接务必写全 `natnew` / `aichr` 实体页。

## 关联页面

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
