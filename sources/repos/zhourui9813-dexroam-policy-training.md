# zhourui9813/DexRoam-Policy-Training

> DexRoam 官方 VLA 训练仓库来源归档

- **类型：** repo
- **链接：** <https://github.com/zhourui9813/DexRoam-Policy-Training>
- **项目页：** <https://dexroam.github.io/>
- **论文：** <https://arxiv.org/abs/2609.35761v1>
- **源码核对：** 2026-11-11，main 分支（`git/trees/main` SHA `a82d2c98dc0dff00c17e1aa4fb2bb93d1c5655fe`）
- **数据：** <https://huggingface.co/datasets/zhourui9813/DexRoam_Realworld_Data>，公开 50 条 Push Chair & Close Laptop 的**机器人** HDF5 episode
- **一句话说明：** 先把 Astribot HDF5 转为 LeRobot v2.1，再用相同 53-D 状态 / 56-D 全身动作接口分别微调 GR00T N1.7 与 π0.5。

## 已核实可执行链路

1. `data_scripts/dexroam_hdf5_to_lerobot.py` 将 HDF5 的本体状态、相对动作、双目图像、任务描述转换成 LeRobot v2.1。`observation.state` 为 **53-D**；`action` 为 **56-D**；存储 parquet、H.264 视频及 `meta/stats.json` 等。
2. **GR00T：** 官方指南 `docs/gr00t_training.md`：`cd dexroam_gr00t && bash scripts/run_train_gr00t.sh`；底层 `gr00t/experiment/launch_finetune.py` 解析 `FinetuneConfig`，构建数据配置与训练配置，调用 `gr00t.experiment.experiment.run()`。推理服务另由 `scripts/run_gr00t_server.sh` 启动。
3. **π0.5：** `docs/pi05_training.md` 先要求 `bash scripts/run_compute_norm_stats.sh` 再 `bash scripts/run_train_pi.sh`；底层 `scripts/train.py` 读取 `dexroam_train_config.py` 的配置并调用数据加载、模型更新与 checkpoint 路径；独立服务由 `scripts/run_pi_server.sh` 启动。
4. 两个训练路径 **分别需要环境与预训练模型**；示例数据与代码不足以独立重现论文五个真机任务的完整训练、评估。

## 论文训练配置（原文 Appendix D）

| Backbone | GPU | 训练步数 | 全局 batch | 学习率 |
|---|---:|---:|---:|---:|
| GR00T N1.7-3B | 4 | 40,000 | 24 | 1e-4 |
| π0.5 | 2 | 30,000 | 4 | 2.5e-5 |

GR00T 冻结预训练语言/视觉 backbone、更新 projector 和 diffusion decoder；π0.5 微调全部参数。训练说明与论文配置属于两类信息，不能把 README 命令视作已跑通全量训练的证据。

## 开源状态与限制

**训练源码已公开，数据仅示例开放。** 公开数据卡提供 50 条单任务机器人记录；论文五任务人类和机器人示教的完整数据以及真机评估环境并未由此样例一并公开。未在此执行 GPU 训练或部署。

## 对 wiki 的映射

- [DexRoam 唯一论文实体](../../wiki/entities/paper-dexroam-mobile-bimanual-manipulation.md)
- [采集 / 对齐代码归档](./zhourui9813-dexroam.md)
- [论文全文](../papers/dexroam_arxiv_2609_35761.md)
