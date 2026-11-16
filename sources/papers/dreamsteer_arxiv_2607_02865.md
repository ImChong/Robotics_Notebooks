# DREAMSTEER: Latent World Models can steer VLA Policies during deployment without any finetuning

- **类型：** paper
- **原始链接：** <https://arxiv.org/abs/2607.02865>
- **全文版本：** <https://arxiv.org/pdf/2607.02865v1>
- **核对日期：** 2026-11-13
- **核对范围：** arXiv v1 PDF，全文 §1–§6 与补充 §7–§10（架构、训练、价值聚合、候选构造、平台和任务）；参考文献浏览。图表以文字层核对，未观看演示视频，未运行训练或真机实验。
- **通读：** 已完成
- **项目页：** <https://dream-steer.github.io/>

## 通读核对

- §3 / 补充 §7：冻结 DINOv2 潜空间、分解时空 attention、异构动作/状态 tokenizer；teacher forcing 与 open-loop sampling MSE 交替训练，共享动力学。WM 训练并非免训练，免的是目标环境微调。
- §3.2 / 补充 §8–§9：π₀ 关节动作经 FK 转为 Cartesian delta，5 个策略样本加 8 个方向/夹爪原语；VLAC（InternVL2-2B）累加相邻帧进度分数，单左视图评分，选一块执行。
- §4.3：每 5 控制步 steering 一次，H=10；13 候选串行约 13 秒，单候选约 1 秒；0.08/0.59/0.37 秒分别是策略/WM/价值耗时，0.59 秒不是整套控制周期。
- 表 2：4 物体各 20 次，完整方案 53/80=66.25%，仅多策略采样排序 34/80=42.50%，单采样 19/80=23.75%；原语或随机选均 0。
- 表 3：IF 45/80=56.25%，判据为与指定对象持续接触或尝试抓取，不等于完整 pick-and-place 成功。§5 指出候选覆盖、单视图误排与延迟限制。

## 对 wiki 的映射

- [DREAMSTEER: Latent World Models can steer VLA Policies during deployment without any finetuning](../../wiki/entities/paper-dreamsteer-vla-deployment-steering.md)
