# SUAVE 来源归档

- **论文：** https://arxiv.org/abs/2610.04009
- **HTML v1：** https://arxiv.org/html/2610.04009v1
- **PDF：** https://arxiv.org/pdf/2610.04009
- **作者：** Rhythm Syed, Jean Mercat, Sedrick Keh, Kushal Arora, Paarth Shah, Aykut Onol, Mengchao Zhang, Tony Dear
- **机构：** Columbia University；Toyota Research Institute
- **论文类型：** arXiv preprint，2026-10-02
- **代码/项目：** arXiv v1 正文未列独立项目页或官方 GitHub 链接。

## 方法摘录

SUAVE 从 MMaDA-8B 初始化，使用 32 层 Transformer，增加 256 个动作 token。文本经 LLaDA tokenizer，视频经冻结 MAGViT-v2 tokenizer，动作离散到 256 档；三种模态共享 embedding/output head。Masked diffusion 少步并行补全目标 token。推理时改变待 mask 模态位置，即可选择视频预测、动作策略或视频+动作生成。无动作人类视频将 action positions 设为 mask 并排除动作损失。

## 评测摘录

- **LIBERO：** Cotrain 平均成功率 95.9%，π0.5 为 97.4%。
- **LIBERO-Plus 零样本：** Cotrain 71.3%，π0.5 为 84.7%；robot initial-state 类别为 82.5%。
- **LIBERO-Plus 训练内：** Cotrain 86.1%。
- **DOMINO：** Cotrain SR Level 1/2/3 分别 18.7%/12.3%/8.2%，论文报告各等级最高。
- **真机：** 单臂 7-DoF xArm7；RTX 5090 上 1,030 ms 生成两个视觉 subgoal 与 5 个动作（约 1 秒），闭环约 2.5 策略查询/秒。

## 限制

仅单臂末端执行器动作空间；该版本不生成自然语言、不评估广义语义泛化。冻结视觉 tokenizer 限制重建质量，部分 co-training 收益统计上不可区分。当前论文正文未列官方实现仓库。
