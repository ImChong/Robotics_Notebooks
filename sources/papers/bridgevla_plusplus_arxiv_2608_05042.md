# BridgeVLA++（arXiv:2608.05042）

> 来源归档（ingest）

- **标题：** BridgeVLA++: A Data-Efficient, Generalizable, and Memory-Augmented Vision-Language-Action Framework for 3D Manipulation
- **类型：** paper / vla / 3d-manipulation / memory / bimanual
- **arXiv abs：** <https://arxiv.org/abs/2608.05042>
- **PDF：** <https://arxiv.org/pdf/2608.05042>
- **HTML：** <https://arxiv.org/html/2608.05042>
- **项目页：** <https://bridgevla-plus.github.io/> — 归档见 [`sources/sites/bridgevla-plus-github-io.md`](../sites/bridgevla-plus-github-io.md)
- **代码：** <https://github.com/BridgeVLA/BridgeVLA>（`main` = ++；`bridgevla` 分支 = NeurIPS 2025 原版）— [`sources/repos/bridgevla.md`](../repos/bridgevla.md)
- **权重 / 数据：** <https://huggingface.co/datasets/LPY/BridgeVLA>（Apache-2.0）；ModelScope 镜像 `susetiankong/bridgevla_plus`
- **前置论文：** BridgeVLA（arXiv:2506.07961，NeurIPS 2025）
- **机构：** 中国科学院自动化研究所 NLPR / 中国科学院大学；FiveAges；作者 Hongtao Wu / Xiao Ma / Tao Kong 贡献时隶属字节跳动 Seed
- **作者：** Peiyan Li\*、Yuze Zhu\*、Yixiang Chen、Qisen Ma、Yuan Xu、Jiabing Yang、He Guan、Yan Huang†、Hongtao Wu、Xiao Ma、Tao Kong、Liang Wang、Tieniu Tan（\* equal；† corresponding）
- **发表 / 上传：** 标注为 *IEEE Transactions on Pattern Analysis and Machine Intelligence* 稿；arXiv 2608.05042（2026-08）
- **入库日期：** 2026-08-07
- **一句话说明：** 在 BridgeVLA「多视图 2D heatmap 对齐」底座上加统一时空记忆（粗阶段时间记忆 + 细阶段空间记忆），以 +9.2% 参数换 RMBench **96.0%** / MemoryBench **99.7%**，并保持 RLBench / COLOSSEUM / GemBench 数据效率与 OOD。

## 相关资料（策展）

| 类型 | 链接 | 说明 |
|------|------|------|
| 项目页 | [bridgevla-plus.github.io](https://bridgevla-plus.github.io/) | 五套件数字、架构图、真机 rollout |
| 代码 | [BridgeVLA/BridgeVLA](https://github.com/BridgeVLA/BridgeVLA) | 五仿真基准 + 真机 train/deploy |
| HF | [LPY/BridgeVLA](https://huggingface.co/datasets/LPY/BridgeVLA) | checkpoint / pretrain corpus |
| 原版 | [arXiv:2506.07961](https://arxiv.org/abs/2506.07961) | BridgeVLA（NeurIPS 2025） |

## 开源状态（步骤 2.5，截至 2026-08-07）

- **已开源：** 项目页 Paper / Code / Checkpoints；GitHub Apache-2.0；HF / ModelScope 权重；`finetune/*/train.sh` 与 `eval.sh` 可辨识入口。
- **边界：** 真机数据未发布（README：self-collected）；RLBench/PyRep 源码因许可不随仓分发，安装脚本从上游钉选 commit 重建。
- **处理：** wiki 写「已开源」并补 `## 源码运行时序图`。

## 摘要级要点

- **底座 BridgeVLA：** 点云 → 正交多视图图像；先在检测数据上做 2D heatmap 预训练；操纵阶段 coarse-to-fine heatmap → 6D 动作，保住 VLM 输入输出对齐。
- **++ 记忆：**
  - **Temporal 𝒯（粗阶段）：** 关键帧交互历史，回答 *what to do next*
  - **Spatial 𝒮（细阶段）：** 初始较少遮挡点云按当前 zoom 重渲染，回答 *where exactly to act*
  - 注入在 **VLM patch-token 空间**，动作头不变；场景记忆可双臂共享
- **结果（项目页）：** RLBench **93.7%**；COLOSSEUM **65.2%**；GemBench **51.1%**；RMBench **96.0%**（无记忆 base 18.9%）；MemoryBench **99.7%**；Dobot 真机记忆任务 basic **93.3%**（无记忆 20.0%）
- **代价：** +**9.2%** 参数（+269.77M / 2.92B）；单步推理 RTX 4090 **0.35→0.57 s**

## 核心摘录（面向 wiki 编译）

### 1) 消融（项目页）

- 去 heatmap 解码：RLBench **90.5% → 31.4%**
- 喂 per-pixel 3D 位置：**→ 56.2%**
- RMBench 去 𝒯：**96.0% → 21.3%**（≈ 无记忆 base）；去 𝒮 几乎无伤

### 2) 复现入口

```bash
bash scripts/download_checkpoints_hf.sh rlbench paligemma clip
bash finetune/RLBench/train.sh
bash finetune/RLBench/eval.sh
```

## 对 wiki 的映射

- 新建实体页：[wiki/entities/paper-bridgevla-plusplus.md](../../wiki/entities/paper-bridgevla-plusplus.md)
- 交叉：[VLA](../../wiki/methods/vla.md)、[KEMO](../../wiki/entities/paper-kemo-event-driven-keyframe-memory-vla.md)、[EventVLA](../../wiki/entities/paper-eventvla-visual-evidence-memory.md)、[Chronos](../../wiki/entities/paper-chronos.md)、[FM-VLA](../../wiki/entities/paper-fm-vla.md)、[manipulation](../../wiki/tasks/manipulation.md)
