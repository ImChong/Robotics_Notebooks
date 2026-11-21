# Click-and-Traverse

> 来源归档（国内具身开源全景 · 源码核对）

- **标题：** Click-and-Traverse
- **类型：** repo
- **机构：** 银河通用（GALBOT · 清华大学）
- **链接：** <https://github.com/GalaxyGeneralRobotics/Click-and-Traverse>
- **论文与项目页：** [arXiv:2601.16035](https://arxiv.org/abs/2601.16035) · <https://axian12138.github.io/CAT/>
- **分类：** 运动RL/技能训练
- **入库日期：** 2026-09-06
- **核查日期：** 2026-11-21
- **一句话说明：** HumanoidPF 杂乱室内全身无碰撞穿行（CAT）的官方完整开源实现：含程序化障碍与势场生成、PPO 专家/DAgger 通才训练、ONNX 导出及 Fast-LIO2+OctoMap 真机部署。
- **核对范围：** 通读 `GalaxyGeneralRobotics/Click-and-Traverse@866ba39` 的 `README.md`、`train_ppo.py`、`cat_ppo/__init__.py`、`cat_ppo/envs/g1/env_cat.py`（配置与类结构）、`cat_ppo/learning/train/pf_utils.py` 与 `procedural_obstacle_generation/main.py`；未在本机启动 CUDA/MJX 训练或 Unitree G1 SLAM 真机部署。
- **通读：** 已完成

## 开源状态

- **已开源**：README Project Status 清单中，程序化障碍与 HumanoidPF 生成、Specialist 训练、预训练 Specialist 模型与场景数据、真机部署代码（`deploy/Click-and-Traverse-SLAM`）、Real2Sim 构建、DAgger 蒸馏代码与 **Generalist v1 模型（42 个随机障碍场景）** 均已发布（Google Drive / 清华云 / Hugging Face `Axian12138/Click-and-Traverse`）；仅“Expanded scene datasets”尚待扩展。

## 关键入口与实现要点（`866ba39`）

- **势场离线生成：** `procedural_obstacle_generation/main.py` 调用 `generate_typical_obstacle` / `generate_random_obstacle` / `generate_pf`，经 `pf_modular.py` 计算 `sdf.npy`、`bf.npy`（梯度场）与 `gf.npy`（渐进引导场）写入 `data/assets/{TypiObs,RandObs}/`。
- **专家与蒸馏训练：** `train_ppo.py`（或批量 `train_batch.py`）支持 `G1Cat` 与 `G1CatPri`（特权观测）以及 `mlp` / `humanoid_transformer` 网络，使用 `wrap_for_brax_training_reset` 包装环境训练；`train_ppo_dagger.py` / `train_dagger_batch.py` 将多专家蒸馏为通才，并经 `cat_ppo.eval.brax2onnx` 自动导出 ONNX。
- **评测与部署：** `cat_ppo.eval.mj_onnx_play`（交互回放）与 `cat_ppo.eval.mj_onnx_test`（批量成功率 JSON）；真机部署位于 `deploy/Click-and-Traverse-SLAM/` 与 `deploy/gx_loco_deploy/`。

## 对 wiki 的映射

- [wiki/entities/cn-os-click-and-traverse.md](../../wiki/entities/cn-os-click-and-traverse.md)
