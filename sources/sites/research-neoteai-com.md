# NeoteAI Research（research.neoteai.com）

- **标题：** NeoteAI Research Blog / 𝒩₀ 三件套项目枢纽
- **类型：** site / research-hub
- **URL：** <https://research.neoteai.com>
- **入库日期：** 2026-07-26
- **机构：** 新智具身智能（NeoteAI）× 复旦 TEAI
- **配套公司站：** [neoteai-com.md](./neoteai-com.md)
- **一句话说明：** 2026-07-25 同日发布 **𝒩₀-Foundation / 𝒩₀-VTLA / 𝒩₀-TWAM** 三篇技术报告项目页，构成「触觉基础设施 → 预测触觉 VTLA → 触觉原生 WAM」研究栈。

## 项目页核查（步骤 2.5 · 2026-07-26）

| 子页 | Technical Report | Dataset | Code | Checkpoints |
|------|------------------|---------|------|-------------|
| [n0-foundation/](https://research.neoteai.com/n0-foundation/) | [PDF](https://research.neoteai.com/assets/n0-foundation-report.pdf) | [OpenNeoData](https://huggingface.co/datasets/NeoteAIEmbodied/OpenNeoData) | [neoteai/N0-Foundation](https://github.com/neoteai/N0-Foundation) | — |
| [n0-vtla/](https://research.neoteai.com/n0-vtla/) | [PDF](https://research.neoteai.com/assets/n0-vtla-report.pdf) | （依赖 Foundation / NeoData） | [neoteai/N0-VTLA](https://github.com/neoteai/N0-VTLA) | 同 Code 链（仓内 roadmap） |
| [n0-twam/](https://research.neoteai.com/n0-twam/) | [PDF](https://research.neoteai.com/assets/n0-twam-report.pdf) | （同上） | [neoteai/N0-TWAM](https://github.com/neoteai/N0-TWAM) | 同 Code 链（仓内 roadmap） |

**开放程度判定：部分开源**

- **已发布：** OpenNeoData（约 **5,000 h** 开源子集，LeRobot v3.0，HF + ModelScope，**门禁 + CC-BY-NC-SA-4.0**）；三仓均有 GitHub 入口与技术报告 PDF。
- **待发布（仓内 Roadmap 写明 By July 31, 2026）：** NeoForce 权重与代码；𝒩₀-VTLA / 𝒩₀-TWAM 模型代码、预训练/后训练权重与训练配方。截至入库日三仓顶层仅 `README.md` + `LICENSE` + `diagrams/`，**无可运行训练/推理入口**。
- **合作署名：** 各页页眉均链 [Fudan TEAI](https://teai.fudan.edu.cn/)。

## 三件套要点（索引）

| 代号 | 定位 | 关键数字（项目页口径） |
|------|------|------------------------|
| **𝒩₀-Foundation** | 传感器 + NeoData + NeoForce + NeoReal/NeoSim | NeoData **>30k h / 1.4M ep / 6 本体 / 450+ 任务**；OpenNeoData **5k h**；π₀.₅+NeoForce NeoReal **32.5% / 47.5** |
| **𝒩₀-VTLA** | 潜空间触觉 token 的 VTLA + ALTER 离线 RL | NeoReal 九任务均 **47.2%**（π₀.₅ **29.4%**）；ALTER 下毛巾折叠 **95%** |
| **𝒩₀-TWAM** | 触觉原生世界–动作模型（非对称 MoT） | UniVTAC **84.5%** · NeoSim **49.4%** · 真机八任务 **46.3%**；~**7.16B** 可训参数 |

## 关联资料

- 论文：[`n0_foundation.md`](../papers/n0_foundation.md) · [`n0_vtla.md`](../papers/n0_vtla.md) · [`n0_twam.md`](../papers/n0_twam.md)
- 仓库：[`n0-foundation.md`](../repos/n0-foundation.md) · [`n0-vtla.md`](../repos/n0-vtla.md) · [`n0-twam.md`](../repos/n0-twam.md)
- Wiki：[`neoteai.md`](../../wiki/entities/neoteai.md) · [`paper-n0-foundation.md`](../../wiki/entities/paper-n0-foundation.md) · [`paper-n0-vtla.md`](../../wiki/entities/paper-n0-vtla.md) · [`paper-n0-twam.md`](../../wiki/entities/paper-n0-twam.md)
