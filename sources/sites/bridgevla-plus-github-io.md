# bridgevla-plus.github.io（BridgeVLA++ 项目页）

- **标题：** BridgeVLA++: A Data-Efficient, Generalizable, and Memory-Augmented VLA Framework for 3D Manipulation
- **类型：** site / project-page
- **URL：** <https://bridgevla-plus.github.io/>
- **配套论文：** [arXiv:2608.05042](https://arxiv.org/abs/2608.05042) — 归档见 [`sources/papers/bridgevla_plusplus_arxiv_2608_05042.md`](../papers/bridgevla_plusplus_arxiv_2608_05042.md)
- **代码：** <https://github.com/BridgeVLA/BridgeVLA> — [`sources/repos/bridgevla.md`](../repos/bridgevla.md)
- **权重：** <https://huggingface.co/datasets/LPY/BridgeVLA>
- **入库日期：** 2026-08-07

## 一句话摘要

BridgeVLA++ 官方站点：用「多视图 heatmap 对齐 + 统一时空记忆」同时覆盖数据高效 3D 操纵、OOD 泛化与记忆依赖任务；展示五套件数字、双臂 RMBench、Franka/Dobot 真机与消融。

## 公开信息要点（截至入库日）

- **页首指标：** RLBench **93.7%**；COLOSSEUM **65.2%**；GemBench **51.1%**；RMBench **96.0%**；MemoryBench **99.7%**；真机记忆任务 **93.3%**。
- **入口：** Paper / Code / Checkpoints（HF）齐全 → **已开源**。
- **架构：** 粗阶段 Temporal memory 𝒯 + 细阶段 Spatial memory 𝒮，注入 VLM patch-token；双臂共享场景记忆。
- **消融：** RMBench 去 𝒯 崩到 ~21%；去 heatmap 解码使 RLBench 从 90.5% 掉到 31.4%。
- **工程代价：** +9.2% 参数；RTX 4090 单步 0.35→0.57 s。

## 为何值得保留

- **步骤 2.5 核查主入口** 与论文/GitHub 三角互证。
- **记忆 VLA 选型对照面：** 相对 KEMO / EventVLA / Chronos，本页给出 **3D heatmap + 时空双记忆** 坐标与可下载权重。

## 关联资料

- 论文归档：[`sources/papers/bridgevla_plusplus_arxiv_2608_05042.md`](../papers/bridgevla_plusplus_arxiv_2608_05042.md)
- 代码归档：[`sources/repos/bridgevla.md`](../repos/bridgevla.md)
- Wiki 实体：[wiki/entities/paper-bridgevla-plusplus.md](../../wiki/entities/paper-bridgevla-plusplus.md)
