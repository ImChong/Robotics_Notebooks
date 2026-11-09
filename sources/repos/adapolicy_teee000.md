# AdaPolicy 官方源码（teee000/AdaPolicy）

> 来源归档（repo）

- **标题：** AdaPolicy: Action-Driven Anticipatory Robot Policies via Next-Chunk Prediction
- **类型：** repo
- **源码：** <https://github.com/teee000/AdaPolicy>
- **主论文：** <https://arxiv.org/abs/2610.34280>
- **项目页：** <https://teee111.github.io/AdaPolicy/>
- **核对日期：** 2026-11-09
- **开源状态：** 已开源（MetaWorld 50 任务与 RoboTwin 2.0 50 任务训练/评测实现）；真机训练/部署代码及完整 LIBERO 运行链路未见于当前仓库，不能据此宣称全部复现。
- **代码核对范围：** README、metaworld/run_train_ema.sh、metaworld/train_metaworld_ema.py、metaworld/models/model_runner_ema.py、metaworld/models/vla_model_fm_ema.py、metaworld/models/vla_model_fm.py、metaworld/scripts/eval_metaworld_parallel.py、robotwin/train_robotwin_ema.py、robotwin/models/model_runner_ema.py、robotwin/models/model_fm_ema.py、robotwin/robotwin_infer.py；核对训练/EMA/损失/评测入口，未声称逐行审计整个仓库。
- **沉淀到 wiki：** [AdaPolicy / 动作驱动预判策略](../../wiki/entities/paper-adapolicy.md)

## 已核验的运行入口

1. 安装：README 中 Python 3.10、PyTorch 2.7.1、DINOv3 与 MetaWorld/MuJoCo 依赖；按 README 获取 ModelScope HDF5 演示数据与视觉权重。
2. MetaWorld 数据：\`metaworld/dataloader/dataset.py\` 构造动作块及 \`next_action_sequence\`；\`metaworld/utils/compute_norm_stats_metaworld.py\` 生成归一化统计。
3. MetaWorld 训练：在 \`metaworld/\` 目录运行 \`./run_train_ema.sh all 40\`，脚本调用 \`train_metaworld_ema.py --config configs/metaworld.yaml\`；训练产生模型与 EMA shadow checkpoint。
4. 模型前向：\`models/model_runner_ema.py::VLAWrapper.forward\` 提取冻结视觉特征、归一化当前/下块动作、取 EMA 参数，并调用 \`vla_model_fm_ema.py::calc_flow_matching_loss\`。
5. 梯度与 EMA：两次 flow matching 前向通过 \`torch.func.functional_call\` 调用第二次模型；第二次使用不带梯度的 EMA 权重，但保留 \`obs_tokens_evolved\` 的计算图。训练循环在 \`loss.backward()\` / \`optimizer.step()\` 后调用 \`ema_helper.update(action_model)\`。
6. MetaWorld 评测：\`./run_train_ema.sh eval checkpoints_vla/<run>/checkpoint_epoch_40.pt all 50\` → \`scripts/eval_metaworld_parallel.py\`。
7. RoboTwin：\`robotwin/train_robotwin_ema.py\` 使用同类 EMA + next-chunk 损失；\`robotwin/robotwin_infer.py::RobotWinInference\` 提供推理接口，评测另有 \`robotwin/eval.sh\`。数据获取/预处理步骤 README 指向 LiLa-WAM。

## 关键源文件

- [MetaWorld 训练入口](https://github.com/teee000/AdaPolicy/blob/main/metaworld/train_metaworld_ema.py)
- [MetaWorld EMA Runner](https://github.com/teee000/AdaPolicy/blob/main/metaworld/models/model_runner_ema.py)
- [MetaWorld flow matching + next-chunk 损失](https://github.com/teee000/AdaPolicy/blob/main/metaworld/models/vla_model_fm_ema.py)
- [RoboTwin EMA Runner](https://github.com/teee000/AdaPolicy/blob/main/robotwin/models/model_runner_ema.py)
- [RoboTwin 损失实现](https://github.com/teee000/AdaPolicy/blob/main/robotwin/models/model_fm_ema.py)
- [MetaWorld 评测脚本](https://github.com/teee000/AdaPolicy/blob/main/metaworld/scripts/eval_metaworld_parallel.py)

## 与其他资料互链

- [论文原文归档](../papers/adapolicy_arxiv_2610_34280.md)
- [项目页归档](../sites/adapolicy.md)
