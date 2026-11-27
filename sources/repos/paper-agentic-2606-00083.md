# Demo2Reward 与 Demo2Reward_RL 官方代码仓库归档

- **论文实体：** [From Demonstrations to Rewards: Test-Time Prompt Optimization for VLM Reward Models](../../wiki/entities/paper-agentic-2606-00083.md)
- **论文归档：** [paper-agentic-2606-00083.md](../papers/paper-agentic-2606-00083.md)
- **官方提示词优化仓库：** https://github.com/cgumbsch/Demo2Reward （默认分支 `main`，截至核对日仓库未附带 `LICENSE` 文件）
- **官方 RL 训练姊妹仓库：** https://github.com/cgumbsch/Demo2Reward_RL （基于 [hengyuan-hu/ibrl](https://github.com/hengyuan-hu/ibrl) 分支扩展，默认分支 `main`，截至核对日仓库未附带 `LICENSE` 文件）
- **开源状态：** 核心代码已开源（提示词离线优化循环与仿真 RL/BC 实验代码均已公开；仿真演示数据集与预训练 BC 策略包通过 Google Drive 外部链接分发；真机 `real_rlinf` 采集与 RL 训练需配合外部 RLInf 与 GELLO 硬件栈，真机数据文件未随仓附带）
- **核对边界：** 静态代码与 README / 文件树核对，未在本地执行安装或运行训练。

## 仓库结构与已核对入口（`cgumbsch/Demo2Reward`）

```text
Demo2Reward/
├── demo2reward.py              # 主入口：运行 Meta-Critic 提示词优化循环并输出最优指令与日志
├── plot_tikz.py                # 评测可视化：对比 Critic 与 GT 预测并输出 TikZ/PNG 图
├── requirements.txt            # 依赖清单（环境要求 Python 3.10 + torch 2.4.0 + flash-attn 2.7.3）
├── datasets/
│   ├── dataset.py              # 通用数据集抽象与前缀切片逻辑
│   ├── metaworld.py            # MetaWorld HDF5 演示读取（Assembly/BoxClose/CoffeePush/StickPull）
│   ├── robomimic.py            # RoboMimic HDF5 演示读取（can/square）
│   └── real_rlinf.py           # 真机 RLInf PKL 演示读取（lid）
├── utils/
│   ├── critic.py               # Critic 评估与目标函数计算
│   ├── evaluate_prompts.py     # 批量评估候选提示词（TPR/TNR/FPR/FNR）
│   ├── prompts.py              # Meta-Critic 与 Critic 系统/用户提示词模板
│   ├── images.py               # 图像缩放与 5 帧均匀采样预处理
│   ├── logging.py              # TensorBoard / W&B 日志与结果序列化
│   └── plots.py                # 曲线与标注辅助绘图
└── vlms/
    ├── vlm_utils.py            # VLM 推理统一封装
    └── qwen_utils.py           # Qwen3-VL（8B/32B）加载与对话模板封装
```

- **典型运行命令（README 核对）：**
  - 优化提示词：`python demo2reward.py --task metaworld_assembly --data_root ./data`
  - 指定更大型 Meta-Critic：`python demo2reward.py --task robomimic_can --meta_vlm qwen3_32b --data_root ./data`
  - 真机数据优化：`--task real_lid --real_robot`

## 姊妹 RL 仓库结构与入口（`cgumbsch/Demo2Reward_RL`）

- **典型入口：**
  - RoboMimic VLM 奖励训练：`train_vlm_rl.py`（配合 `release/cfgs/robomimic_rl/can_ibrl.yaml`、`can_rlpd.yaml`、`square_ibrl.yaml`，参数 `--vlm demo2reward_qwen3_8b --train_end_on_success 0 --reward_at_truncation 1`）
  - MetaWorld VLM 奖励训练：`mw_main/train_rl_vlm_mw.py`（配合 `release/cfgs/metaworld/{ibrl_long,rlpd_long}.yaml` 及 `--bc_policy {assembly,boxclose,coffeepush,stickpull}`）
  - 预训练 BC 与基线：`train_bc.py`、`train_rl.py`、`mw_main/train_bc_mw.py`、`mw_main/train_rl_mw.py`、`tools/dataset_states_to_obs.py`
  - VLM 环境封装与奖励基线：`env/vlm_envs.py`、`env/vlm_prompts.py`、`env/qwen_utils.py`、`env/gvl.py`、`env/roboreward_utils.py`、`env/metaworld_wrapper.py`、`env/robosuite_vlm_env.py`
  - 策略/价值网络与 C++ 回放缓冲区：`bc/bc_policy.py`、`rl/{actor.py,critic.py,q_agent.py,replay.py}`、`common_utils/rela/`（`pybind11` 编译）
