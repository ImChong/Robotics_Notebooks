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
- **许可证：** Apache-2.0（第三方 vendored 组件如 YARR/PerAct/RVT 衍生文件保留各自原许可；RLBench/PyRep 不随仓分发，安装脚本从上游钉选 commit 重建）
- **入库日期：** 2026-08-07
- **通读：** 已完成
- **核对范围：** `_tree.txt`（932 项）、`README.md`（224 行）、`finetune/bridgevla/mvt/{memory.py, keyframe_disc.py, mvt_single.py, mvt.py}`、`finetune/bridgevla/models/bridgevla_agent.py`（`RVTAgent` 方法清单）、`pretrain/pretrain.py`、`finetune/RLBench/{train.py, eval.py}`、`finetune/RMBench/policy/BridgeVLA_Plus/{bridgevla_rmbench_model.py, deploy_policy.py}`、`finetune/real/rvt_our/eval_flask_app.py`
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

## README 宣称的技术栈 / 入口与源码核对模块

| 组件 | 路径 / 命令 | 核对要点 |
|------|-------------|----------|
| 分环境安装 | `finetune/RLBench/install_rlbench.sh` 等（按基准选） | 按基准建独立 conda 环境；RLBench/PyRep 从上游 commit 重建 |
| 权重 / 数据 | `scripts/download_checkpoints_hf.sh` / `download_datasets.sh` | HF `LPY/BridgeVLA` 与 MS 镜像；RLBench 需配 pre-built keyframe cache |
| 2D 预训练 | `pretrain/pretrain.sh` → `pretrain/pretrain.py` | RoboPoint 120K 语料上训 PaliGemma + `ConvexUpSample` 2D 热图对齐头 |
| 模型主干 | `finetune/bridgevla/mvt/{mvt.py, mvt_single.py}` | `MVT` 调度双阶段 `MVTSingle`（`_paligemma_extract` 提取 16×16 token + `_apply_memory` + `_forward_heads` 预测 6D 旋转/夹爪/碰撞） |
| 统一时空记忆 | `finetune/bridgevla/mvt/{memory.py, keyframe_disc.py}` | `MemoryBlock`（8h×128d，2048d）+ `KeyframeDiscriminator` + `MemoryBank`（`keyframe_gt` 双近邻+边界关键帧槽或 `sliding`） |
| Agent 与策略 | `finetune/bridgevla/models/bridgevla_agent.py` | `RVTAgent`：`update`/`update_gembench`/`update_rmbench` 训练与 `act`/`act_rmbench` 闭环记忆推理 |
| 评测与真机 | `finetune/RLBench/eval.py`；`finetune/RMBench/policy/BridgeVLA_Plus/bridgevla_rmbench_model.py`；`finetune/real/rvt_our/eval_flask_app.py` | RMBench 采用双环境 RPC；真机采用 Flask `/predict` HTTP 服务配对工作站 `eval_client.py` |
| 原版代码 | [`bridgevla` 分支](https://github.com/BridgeVLA/BridgeVLA/tree/bridgevla) | NeurIPS 2025 无记忆基座版 |

## 关联资料

- 论文归档：[`sources/papers/bridgevla_plusplus_arxiv_2608_05042.md`](../papers/bridgevla_plusplus_arxiv_2608_05042.md)
- 项目页：[`sources/sites/bridgevla-plus-github-io.md`](../sites/bridgevla-plus-github-io.md)
- Wiki 实体：[wiki/entities/paper-bridgevla-plusplus.md](../../wiki/entities/paper-bridgevla-plusplus.md)
