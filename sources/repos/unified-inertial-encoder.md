# Unified Inertial Encoder — 官方源码仓库

> 来源归档（repo；核查日期 2026-11-16，静态阅读，未执行）

- **代码：** <https://github.com/amazon-far/Unified-Inertial-Encoder>
- **项目页：** <https://unified-inertial-encoder.github.io/>；[项目页归档](../sites/unified-inertial-encoder.md)
- **论文：** <https://arxiv.org/abs/2611.10137>；[论文归档](../papers/unified-inertial-encoder_arxiv_2611_10137.md)
- **许可证：** Apache-2.0（第三方组件另有许可证）
- **权重：** <https://huggingface.co/Open-IO/Unified-Inertial-Encoder>（EMA bf16，55.6M 参数，缺优化器状态）
- **真机数据：** <https://huggingface.co/datasets/Open-IO/humanoid-g1-odometry>
- **wiki：** [论文详情](../../wiki/entities/paper-unified-inertial-encoder.md)

## 官方可运行入口（经 README 和关键代码静态核对）

| 作用 | 入口 | 具体链路 |
|---|---|---|
| 数据预处理 | `python -m src.scripts.{nymeria_imu_process,naturalmotion_imu_process,bonesseed_soma_process,bonesseed_imu_process}` | 对应 `src/config/data/*.yaml`，生成各数据集缓存 |
| 训练 | `python -m src.train experiment=exp15_full_train_C` | `src/train.py` Hydra 实例化数据、`FoundationIMU`、loss，Lightning `training_step` 走 `model(batch, mask)` → 多任务损失 → checkpoint/EMA |
| 模型前向 | `src/model/model.py:FoundationIMU.forward` | tokenizer → adaptor → transformer trunk → 速度、IMU 去噪与语言特征解码 |
| 惯性里程计评测 | `python -m src.eval_foundation --ckpt ... --data_from ... --dataset Humanoid_SuperOdom_NPZ_Processed_Language --json real_g1.json` | `src/eval.py:load_model/inference` → 全站点前向 → `pelvis_restrict` → `Velocity_Metric`、`Trajectory_Metric` |
| 语言检索 | `python -m src.scripts.evaluate_language_classification <CKPT>` | 单独评测 IMU↔语言检索，不等同于 `src.eval_foundation` |
| 性能测时 | `python -m src.scripts.effiency_profiling --ckpt <CKPT> --device cuda:0 --trt` | 4 s、12 IMU、200 Hz 的合成输入，用于 TensorRT profile；不是机器人端闭环 |
| 架构消融 | `bash run.sh` | 依次跑 `exp18_arch_fullattn`、`exp18_arch_norope`、`exp18_arch_discrete` |

## 复现前提与口径

- 官方 README 要求 Python 3.14、与硬件匹配的 PyTorch / CUDA GPU；FlashAttention 可选。原始 Nymeria、Natural Motion、BONES-Seed 数据不随代码发布，需要分别获取。
- 发布的 `unified_encoder.ckpt` 只含 EMA 权重与 `.hydra/`，可以评测/微调，但不能无优化器状态地无损续训。
- README 训练配方跑 64 epochs，论文结果选择第 20 epoch、step 6420。**Real G1 测试片段未参与梯度训练，但测试分数参与 checkpoint 选择**（官方 README 与模型卡）；应按这一事实解读零样本报告。
- 仓库包含训练/离线评估和 TensorRT profiling，但以上**不构成已验证的 G1 机载实时闭环接口**。本次只静态核对关键入口与代码，没有运行测试、训练或下载数据集。

## 一手来源

- [官方 README](https://github.com/amazon-far/Unified-Inertial-Encoder)
- [训练代码](https://github.com/amazon-far/Unified-Inertial-Encoder/blob/main/src/train.py)
- [前向代码](https://github.com/amazon-far/Unified-Inertial-Encoder/blob/main/src/model/model.py)
- [评测代码](https://github.com/amazon-far/Unified-Inertial-Encoder/blob/main/src/eval_foundation.py)
- [公开模型卡](https://huggingface.co/Open-IO/Unified-Inertial-Encoder)
