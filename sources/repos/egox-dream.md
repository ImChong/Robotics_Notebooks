# EgoX 官方仓库：egox-dream/egox

- **类型：** repo
- **地址：** https://github.com/egox-dream/egox
- **项目页：** https://egox-dream.github.io/
- **论文：** https://egox-dream.github.io/assets/egox.pdf
- **核对日期：** 2026-11-13
- **主节点：** [EgoX](../../wiki/entities/paper-egox-mxt-plus.md)

## 代码与入口静态核查

| 路径 | 核实到的职责 |
|---|---|
| scripts/training_scripts/G1-Sim-Dex3/kitchen-Bread/pretrain.sh | 包含 Human-Dex5、其他真实机器人数据和 G1-Sim-Dex3/Kitchen-Cleanup，排除目标 Kitchen-Bread；执行 scripts/train.py |
| scripts/training_scripts/G1-Sim-Dex3/kitchen-Bread/finetune.sh | 从 pretrained_path 权重初始化，训练目标数据 |
| scripts/train.py | 发现 episodes，构造模型配置，调用 egox.training.trainer.train |
| src/egox/data/data.py | HDF5 episode 加载、按本体采样和归一化统计 |
| src/egox/training/trainer.py | load_data、MXT、AdamW、compute_loss、update_ema、保存 checkpoint |
| src/egox/models/mxt/model.py | 模态 tokenizers、共享 trunk、动作头、Dreaming 目标与损失 |
| src/egox/models/mxt/dreaming.py | EmbodimentTokenizer、DreamHead、dream_loss |
| scripts/rollout_scripts/G1-Sim-Dex3/kitchen-Bread/headless.sh | 执行 scripts/rollout.py sim --mode headless 评测 |
| src/egox/inference/policy.py | MXTPolicy 加载 checkpoint、配置、归一化信息，输出动作块 |

## 最短官方流程及开放边界

1. 安装：bash scripts/setup_conda_policy.sh --name egox；仿真用 --sim。
2. 提供 DATA_DIR 并运行上述 pretrain.sh；随后 finetune.sh /path/to/pretrained/policy_last.ckpt。
3. headless.sh --model_path /path/to/training/run：要求 policy_last.ckpt、all_configs.json、dataset_stats.pkl，仿真环境还需 Isaac Sim、Isaac Lab 和机器人资产。

**部分开源：** 源码和命令入口存在；官方 README 明确数据集及 VR 采集流水线尚未发布，训练和部署还需相应外部资产。本文是**源码静态阅读**，不是 GPU/仿真执行记录。
