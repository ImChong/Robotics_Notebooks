# V-JEPA 2：原文通读与实现复核（2026-11-15）

- **类型：** paper
- **原文：** <https://arxiv.org/html/2506.09985>
- **版本：** 2026-11-15 获取的 HTML；具体正文、附录和代码范围如下。
- **代码：** <https://github.com/facebookresearch/vjepa2>
- **核对日期：** 2026-11-15
- **核对范围：** arXiv HTML 正文 §1–9、附录 A–E、全部表格与图注；另通读当前官方 README、app/main.py、app/scaffold.py、Droid 配置、能量景观 notebook 及 world_model_wrapper.py、mpc_utils.py；核对 app/vjepa_droid/train.py 的模型构建和 train_step。未运行 GPU 训练、探针或真机实验；不将 V-JEPA 2.1 当作本文配方。
- **通读：** 已完成

本次复核单独存档；[历史摘录](vjepa2_arxiv_2506_09985.md)保留原有日期和发布状态，不能代替下述版本及协议核对。

## 原文与实现核对

### 两阶段训练的精确含义

- 预训练只回归 masked token 的 EMA 目标表征，编码器至 ViT-g 1B，预训练预测器约 22M；300M 指后训练的动作条件预测器，不是预训练预测器。VM22M 同时含视频与图像，ImageNet 图片重复为 16 帧；不是 22M 条带动作机器人轨迹。
- AC 的“无标注”是无任务名、成功标签和奖励，仍需 RGB 与每帧 7D 末端状态；相邻状态构造动作。论文描述 4 秒、4 fps、16 帧，冻结编码器，300M / 24 层 / hidden 1024 预测器以 teacher-forcing L1 加两步自回归 L1 训练，只用 Droid 左外参相机。当前公开配置名和 `dataset_fpcs` 为 8 帧，不能拿文件名倒推论文 16 帧协议。
- 零样本指部署实验室与任务未另采训练数据；不是没有机器人后训练，也不是不需要图像目标。Pick-and-place 人工给出抓住、搬近与放下三张目标图，按 4/10/4 步定时切换，不是自主发现长程子目标。

### 评测条件与结果

| 指标 | 论文条件 | 结果与边界 |
|------|----------|------------|
| SSv2 top-1 | ViT-g 384、64 帧、2 temporal × 3 spatial crops、训练 attentive probe | 77.3%；不是冻结模型零训练分类 |
| EK100 action mean-class recall@5 | 提前 1 秒，32 帧 / 8 fps，训练 probe | 39.7%；仅 encoder 已有 39.1%，预测器额外贡献 0.6 pp |
| 视频问答 | 88.5M 图文/视频文对齐数据，Llama 3.1 8B | PerceptionTest 84.0%（额外 SFT，test），TempCompass 76.9%；受控 18M + Qwen2-7B 实验另计 |
| Lab 1 / Lab 2 cup pick-and-place | 各 10 trials，图像子目标、blocking controller | 80 / 80%；box 为 80 / 50%，不能只写最高 80% |
| 规划时间 | 单 RTX 4090，H=1，800 samples × 10 rounds、top 10 | 16 秒/动作；Cosmos 80 samples × 10、H=1 为 4 分钟，采样预算不同 |

两实验室平均 grasp cup / box 为 65 / 25%，pick-and-place 为 80 / 65%；小样本实验不等于普遍可靠性。论文相机未经显式标定，但先人工试选位置，动作轴误差随视角明显变化；附录提出旋转校正设想，明确未在报告实验中执行。视觉目标与单步短视规划无法证明无需子目标的长程能力。

### 当前源码核对

- `app.main:process_main` 读 YAML、初始化分布式环境，经 `app.scaffold.main` 动态调用 `app.vjepa_droid.train.main`。配置中的权重与 Droid CSV 均为待替换本地路径。`train_step` 在 `no_grad` 下逐帧编码，计算 teacher-forcing 与 `auto_steps=2` 自回归损失，只更新预测器路径。
- `energy_landscape_example.ipynb` 从本地 `franka_example_traj.npz` 读观测与状态；PyTorch Hub 加载编码器/预测器，计算笛卡尔网格的 latent L1 能量。末尾通过 `WorldModel.infer_next_action → cem → step_predictor` 返回动作，不含真实机器人驱动。
- notebook 为 CPU 演示显式使用 rollout=2、samples=25、cem_steps=2，不是论文 800×10、H=1 的速度或质量复现。当前 `cem` 固定旋转动作维度为 0，对 xyz 按轴裁剪 `[-maxnorm,maxnorm]`，不应将实现写成严格 L1-ball 投影；论文的动作约束描述单独保存。
- README 现收录 V-JEPA 2.1 的 dense predictive loss、多层自监督及新权重；这些不属于 arXiv:2506.09985。原版 V-JEPA 2 与 2-AC 的配置和权重必须成套选择。主体 MIT，README 列出的部分数据增强/worker 工具使用 Apache 2.0。

## 对 wiki 的映射

- [V-JEPA 2 详情](../../wiki/entities/paper-vjepa2.md)
