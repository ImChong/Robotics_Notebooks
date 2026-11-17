# HOVER 官方 Isaac Lab 实现

- **类型：** repo
- **代码：** https://github.com/NVlabs/HOVER
- **项目页：** https://hover-versatile-humanoid.github.io/
- **论文：** https://arxiv.org/abs/2410.21229

[HOVER 详情](../../wiki/entities/paper-bfm-14-hover.md)

## 官方原文复核（2026-11-17）

- **版本：** `8088f6cfb42a8f307dc614735197796a86ce8490`。
- **核对范围：** README 全文；对照 train_student_policy.py 和 student_policy_trainer.py 的入口、rollout、oracle 标注及 update；未运行 GPU 训练、推理或真机实验。
- **通读：** 已完成

Isaac Lab 2.0 移植，代码 Apache-2.0。AMASS/SMPL、重定向数据及 teacher checkpoint 需另备。默认训练 OmniH2O specialist；generalist 需配置 DISTILL_MASK_MODES_ALL。实现的动作损失为二范数均值，论文给出平方 L2；不是已运行的等价性证明。
