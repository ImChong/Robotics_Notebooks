# LAPT-Scale：Scaling Embodied Policies with Latent Action Pre-Training on Human Videos

> 来源归档（paper）

- **标题：** LAPT-Scale: Scaling Embodied Policies with Latent Action Pre-Training on Human Videos
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2610.13732>（v1，2026-10-09，cs.CV）
- **PDF：** <https://arxiv.org/pdf/2610.13732v1>
- **会场：** CoRL 2026（正文页脚「10th Conference on Robot Learning (CoRL 2026), Austin, Texas, USA」）
- **作者与机构：** Zhiyuan Xiang、Peidong Liu、Mingyang Li 等；京东 Joy Future Academy、哈工大深圳、清华 SIGS、港科大（广州）
- **代码 / 项目页：** 正文未给出（核对范围内未见任何仓库或项目页链接）
- **入库日期：** 2026-11-12
- **一句话说明：** 在 23.3K 小时第一人称人类视频上训练连续潜在动作模型（LAM），用其后验均值标注 5.1K 小时清洗子集作为 VLA 的 Stage-1 预训练目标，再经 250 小时目标机器人数据对齐与任务后训练；仅把 Stage-1 人视频从 1K 增至 5K 小时，真机三任务平均成功率 63.3% 升至 85.0%。

## 核对范围与通读

- **核对范围：** 用 pdftotext 通读 arXiv:2610.13732v1 PDF 全部 25 页：正文第 1–9 页、参考文献、附录 A–H（数据构成、训练与算力、基线、评测协议、扩展真机缩放、LAM 诊断、Stage-1 潜在 rollout、泛化评测）全部文字与表格 1–19；图 1 与图 4 直接看图，其余图（图 2、3、5–12）只读了图注与正文对它们的描述，未逐图细看。未读：参考文献所引的其它论文（JoyAI-RA、VITRA、DreamDojo、LARY 等）、未运行任何代码；论文无补充材料或视频之外的附件可读。
- **通读：** 已完成

## 方法摘要（章节对应）

1. **LAM（§3.2）：** 时空 VAE，相邻帧 (x_t, x_t+1) 经带可学习 action query 的 Transformer 编码器输出对角高斯后验，采样 d=32 的连续潜在动作 z_t；解码器以当前帧补丁嵌入加 W_z z_t 预测下一帧。损失 L_LAM = L_perc（MSE+LPIPS）+ λ_dino·L_dino（冻结 DINOv2 的补丁与 CLS 特征）+ λ_flow·L_flow（冻结 RAFT 光流，目标光流停梯度）+ β·L_KL（式 1–4）。
2. **Policy（§3.3）：** VLM 骨干加 Perceiver 式动作专家，沿用 JoyAI-RA 架构，条件流匹配训练（式 5）。Stage 1 目标为 LAM 后验均值潜在动作块；Stage 2 切到 250 小时机器人动作块；Stage 3 任务后训练。三阶段共用同一动作专家，无独立人视频头或适配器；推理时输出可执行机器人动作，LAM 编码的动作只作 Stage-1 监督。
3. **数据（§3.1、附录 A）：** LAM 训练 23.3K 小时，来源 Egocentric-10K 10.0K、EgoDex 0.7K、Ego4D（EgoVid 式过滤）1.8K、Xperience 1.1K、EgoVerse 1.2K、自采 8.5K（表 8）；Stage-1 清洗子集 5.1K 小时（EgoDex 0.5K、Ego4D 0.8K、Xperience 0.4K、EgoVerse 0.5K、自采 2.9K，Egocentric-10K 为 0 因缺手部姿态标注，无法与手部运动基线公平比）。
4. **算力（表 9）：** LAM 32 张 H200 训练 154.3 小时（4,938 GPU 小时）；标注 5.1K 小时约 1,700 GPU 小时（3 视频小时/GPU 小时）；Stage 1 约 43h、Stage 2 约 50h（表 10，均 100K 步，批 1024）。
5. **LAM 超参（附录 B.1、表 9）：** 24 层编码器与 24 层解码器、模型维 1024、16 头、patch 16、分辨率 320×240；KL 权重 1e-6、LPIPS 0.01、DINO 0.003、光流 3.3e-5；AdamW、学习率 2.5e-5、100 万步。

## 关键结果（表号）

- **RoboTwin 2.0（表 1）：** Easy 90.0 / Hard 89.3；列出的基线 π0 65.9/58.4、π0.5 82.7/76.8、Motus 88.7/87.0、LingBot-VLA 88.6/86.7、JoyAI-RA 88.6/87.2。
- **RoboCasa-GR1 Tabletop（表 2）：** 64.8；GR00T-N1.6 47.6、Qwen3PI 43.9、ABot-M0 58.3、JoyAI-RA 63.2。评测 24 任务×50 rollout（附录 D.2）。
- **真机三任务（表 3，AgiBot G1，每任务 20 次）：** LAPT-Scale Mouse Pad 85.0 / Headphones 95.0 / Cup 75.0，均值 85.0；JoyAI-RA 均值 80.0；π0.5 56.7；Diffusion Policy 45.0。
- **Stage-1 缩放（图 4、表 14、表 15）：** 0 小时对照 55.0；LAPT-Scale-LAM 1K/3K/5K 小时为 63.3/80.0/85.0；DreamDojo LAM 为 61.7/75.0/76.7；手部运动标注为 58.3/75.0/55.0。
- **Stage-1 移除对照（表 4、表 16）：** 无 Stage 1 为 55.0；无 Stage 1 且延长 Stage 2 做算力匹配为 53.3；有 5K 小时 Stage 1 为 85.0。
- **LAM 对齐消融（表 5）：** MSE-only 的 RoboTwin Hard 为 74.2、去 LPIPS 82.5、去 DINO 84.1、去光流 77.8、完整 89.3；LARY 线性探针 0.675 / 0.691 / 0.698 / 0.685 / 0.713。去光流掉 11.5 个点，而 PSNR 34.30 与完整 34.42 相近。
- **LAM 诊断（表 7、表 17）：** 教师强制下 VQ-LAM（表 17 中记为 LAPA）PSNR 18.3416，MSE-only 33.9844，LAPT-Scale-LAM 34.4217；LARY 准确率 0.232 / 0.675 / 0.713。自回归 rollout 下 28.8291 对 28.3290（MSE-only）。
- **四个更难真机任务（表 6、表 19）：** 每任务每域 20 次、联合策略每任务 200 条演示；Seen / Position / Illumination / Background / 均值：LAPT-Scale 85.0/75.0/70.0/70.0/75.0，π0.5 85.0/50.0/45.0/40.0/55.0，Diffusion Policy 60.0/20.0/15.0/15.0/27.5。
- **Stage-1 潜在 rollout（表 18）：** 真值潜在动作 PSNR 30.7705，VLA 预测潜在动作 25.5371。

## 作者自述的局限（§5）

- 视觉潜在动作不保证对应机器人可独立控制的自由度，目标机器人对齐仍必需。
- 视频操作信号弱、具身差距大、遮挡重、自我相机运动主导时可能失效。
- 真机仅 7 个任务、1 个机器人平台，每条件 20 次试验，成功率精度有限，部署鲁棒性需更广验证。

## 对 wiki 的映射

- [paper-lapt-scale](../../wiki/entities/paper-lapt-scale.md)
