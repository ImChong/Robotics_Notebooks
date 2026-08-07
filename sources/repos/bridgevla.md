# BridgeVLA / BridgeVLA++（BridgeVLA/BridgeVLA）

> 来源归档（repo）

- **标题：** BridgeVLA++ — Data-Efficient, Generalizable, Memory-Augmented 3D VLA
- **类型：** repo / vla / 3d-manipulation / memory / bimanual
- **来源：** BridgeVLA 组织（CASIA NLPR 等）
- **链接：** <https://github.com/BridgeVLA/BridgeVLA>
- **论文：** [arXiv:2608.05042](https://arxiv.org/abs/2608.05042)（++）· [arXiv:2506.07961](https://arxiv.org/abs/2506.07961)（原版 NeurIPS 2025）
- **项目页：** <https://bridgevla-plus.github.io/> — [`sources/sites/bridgevla-plus-github-io.md`](../sites/bridgevla-plus-github-io.md)
- **权重：** HF [`LPY/BridgeVLA`](https://huggingface.co/datasets/LPY/BridgeVLA)；ModelScope `susetiankong/bridgevla_plus`
- **Stars：** ~201（2026-08-07）
- **许可证：** Apache-2.0
- **入库日期：** 2026-08-07
- **一句话说明：** 官方实现：`main` 为 BridgeVLA++（五仿真基准 + 真机）；`bridgevla` 分支保留原版 BridgeVLA。
- **沉淀到 wiki：** 是 → [`wiki/entities/paper-bridgevla-plusplus.md`](../../wiki/entities/paper-bridgevla-plusplus.md)

---

## 开源状态（步骤 2.5）

| 项 | 状态（2026-08-07） |
|----|-------------------|
| 训练 / 评测 | **已开源**（`finetune/*/train.sh`、`eval.sh`、GemBench/memoryBench server+client） |
| 预训练 | `pretrain/pretrain.sh` + HF `pretrain_data` |
| 权重下载 | `scripts/download_checkpoints_hf.sh` / `_ms.sh`（all ~120 GiB） |
| 第三方数据 | `scripts/download_datasets.sh`（RLBench/COLOSSEUM/GemBench/memoryBench/RMBench） |
| 真机数据 | **未发布** |
| 许可证 | **Apache-2.0** |

**结论：** **已开源可运行实现**（仿真全链路）；真机需自采数据。

---

## README 宣称的技术栈 / 入口

| 组件 | 路径 / 命令 |
|------|-------------|
| 分环境安装 | `finetune/RLBench/install_rlbench.sh` 等（按基准选） |
| 权重 | `bash scripts/download_checkpoints_hf.sh rlbench paligemma clip` |
| 训练 | `bash finetune/RLBench/train.sh`（及 Colosseum / GemBench / memoryBench / RMBench / real） |
| 评测 | `bash finetune/RLBench/eval.sh`；GemBench/memoryBench 需双终端 server+client |
| 原版代码 | [`bridgevla` 分支](https://github.com/BridgeVLA/BridgeVLA/tree/bridgevla) |

## 关联资料

- 论文归档：[`sources/papers/bridgevla_plusplus_arxiv_2608_05042.md`](../papers/bridgevla_plusplus_arxiv_2608_05042.md)
- 项目页：[`sources/sites/bridgevla-plus-github-io.md`](../sites/bridgevla-plus-github-io.md)
- Wiki 实体：[wiki/entities/paper-bridgevla-plusplus.md](../../wiki/entities/paper-bridgevla-plusplus.md)
