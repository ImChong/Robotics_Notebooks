# sylvestf/WCM — World Critic Model 官方代码库

> GitHub 开源仓库核对档案

- **仓库地址：** <https://github.com/sylvestf/WCM>
- **许可证：** MIT License
- **关联论文：** [arXiv:2607.29613 World Critic Model](../papers/wcm_world_critic_arxiv_2607_29613.md)
- **关联模型/数据：** <https://huggingface.co/collections/Sylvest/wcm>（已公开 `Sylvest/pick-place-wcm`、`Sylvest/pick-place-wcm-ckpt`、`Sylvest/libero_plus_lerobot`）
- **通读：** 已完成
- **核对范围：** 已通读仓库 `_tree.txt`、`README.md`、四条主 shell 脚本（`1_add_returns.sh`–`4_gen_video.sh`）、训练配置 `configs/train_8gpu.yaml`、核心模块 `world_critic/{train,model,training,evaluate,data,config,checkpoint,distributed,curves,metrics}.py`、`scripts/{add_returns_to_lerobot,make_success_labels,h5_to_lerobot,convert_lerobot_*}.py` 以及可视化工具 `episode_value_video/{cli,pipeline,render,curves,sources}.py`。

---

## 开源完整度核查结论

**状态：部分开源（Critic 本体全链路已开源；下游策略 RL 训练代码与其余权重待逐步释出）**

1. **已完整开源并可运行的部分：**
   - LeRobot v1.6/v2.0/v2.1 → v3.0 转换与基于 `--failure-penalty 300 --normalization global_minmax` 的回报字段注入脚本（`1_add_returns.sh` → `scripts/add_returns_to_lerobot.py`）。
   - 多视角 ViT + 冻结 CLIP 语言融合 + 因果历史 Transformer + 结构化动作隔离 `value_head` + FiLM 门控隐动力学 `ActionConditionedDynamics` + 全局 DDP `SIGReg` 正则化的完整 WCM Critic 模型与分布式训练/断点续训管线（`2_run_train.sh` → `python -m world_critic.train`）。
   - 分布式评估与逐帧 `episode_curves.json`/CSV/PNG 曲线导出（`3_run_eval.sh` → `python -m world_critic.evaluate`）。
   - 价值曲线叠加视频渲染工具（`4_gen_video.sh` → `python -m episode_value_video render`）。
2. **尚未在该仓库释出的部分（Coming Soon）：**
   - README 明确标注 `"We will gradually open-source ckpts of the data points claimed in the paper and the corresponding RL code."`
   - 仓库当前不含论文中的下游策略 PPO / GRPO / FlowSDE / FlowNoise / AWR / RECAP 策略优化闭环代码与真机环境驱动脚本，且 Hugging Face 目前只公开了 Pick-and-Place（ManiSkill 演示/权重）与 LIBERO-Plus LeRobot 转换数据集。

---

## 代码结构与关键实现细节

| 文件路径 | 核心类 / 入口 | 已核对功能与配置 |
| :--- | :--- | :--- |
| `1_add_returns.sh` & `scripts/add_returns_to_lerobot.py` | `conversion_plan` / CLI | 校验并迁移 LeRobot 至 `v3.0`，结合 `--success-labels` 与 `--failure-penalty 300` 计算轨迹回报并做 `global_minmax` 归一化 |
| `configs/train_8gpu.yaml` | Hydra/YAML 默认超参 | `history_size: 3`, `prediction_horizon: 1`, `latent_dim: 384`, `trunk_depth: 6`, `dynamics_depth: 3`, ViT `google/vit-base-patch16-224-in21k`(可微调), CLIP `openai/clip-vit-base-patch32`(冻结), `value_weight: 1.0`, `next_state_weight: 1.0`, `sigreg_weight: 0.01`, `lr: 5.0e-5`, `per_device_batch_size: 32` |
| `world_critic/train.py` | `run()` | 分布式初始化 (`expected_world_size`)、episode 切分清单、`WorldCriticCollator`、`compute_losses` 训练/验证循环与 checkpoint 保存 |
| `world_critic/model.py` | `WorldCriticModel`, `ActionFreeContextTrunk`, `ActionConditionedDynamics`, `GatedDynamicsBlock`, `SIGReg` | `forward()` 先调用 `value_head(context)` 再调用 `dynamics(..., actions)`，从代码拓扑层面强制动作不泄漏进价值分支；提供 `rollout_latent()` 做潜空间多步推演 |
| `world_critic/training.py` | `compute_losses()`, `evaluate_loader()`, `differentiable_global_gather()` | 计算 DDP 全局精确均值 MSE（`value_loss` + `next_state_loss` + 可选 `next_state_vector_loss`）并通过 `dist_nn.all_gather` 计算全局特征 `SIGReg` |
| `world_critic/evaluate.py` | `run()` | 评估 `token_*` 密集指标与每窗口末帧对齐的在线 `value_*` / `next_state_mse`，输出 `episode_curves.json` |
| `episode_value_video/cli.py` & `render.py` | `python -m episode_value_video render` | 读取 `episode_curves.json` 与原视频/LeRobot 分片渲染双曲线对比视频 |
