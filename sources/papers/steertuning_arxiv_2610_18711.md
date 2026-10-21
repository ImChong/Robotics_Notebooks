# Steertuning Policies from a Few Human Videos（arXiv:2610.18711）

> 来源归档（ingest）

- **标题：** Steertuning Policies from a Few Human Videos
- **arXiv：** <https://arxiv.org/abs/2610.18711>（HTML：<https://arxiv.org/html/2610.18711>）
- **项目页：** <https://steertuning.github.io/>（归档：[steertuning-project.md](../sites/steertuning-project.md)）
- **代码：** 项目页有 Code / Model 按钮但无 URL，无 GitHub / Hugging Face 链接（2026-10-17 核查）
- **作者：** Billy Yan、Isaac Tu、Kelly Lee、Blake Chang、Kenny Ma、Kanad Patel、Dhawal Kabra、Neer Patel、Kevin Wu、Lerrel Pinto、Irmak Guzey
- **机构（项目页）：** 纽约大学（New York University）
- **入库日期：** 2026-10-17
- **文章来源：** 具身智能小站 2026-10-17 公众号文章
- **一句话说明：** 在约 100 小时第一人称人类交互视频上预训练 4 亿参数扩散策略，新任务仅用 5 条示范做 LoRA 微调并在采样时向最近示范转向（steering）。

## 开源状态（步骤 2.5）

- **项目页核查（2026-10-17）：** 摘要称数据集、模型权重、代码与机器人视频"将"发布于项目站；按钮无链接，无发布日期。
- **结论：** **宣称将开源 / 未列链接**。

## 摘录 1：方法（arXiv HTML）

- 预训练数据 D_wild：Aria Gen 2 第一人称录像 109.6 小时（摘要取整为约 100 小时），10 Hz，3.69M 帧，48.7k 条自动指令；手-场景接触（55 cm 内）切段，Qwen3-VL-8B 写指令，Gemini 2.5 Flash 给关键点，SAM3 出物体掩码。
- 模型：400M 参数 DiT；5,000 点点云经冻结 Utonia 编码；物体掩码 ResNet-18；冻结 SigLIP 文本；T=100 训练步，推理 8 步 DDIM；动作为 52 个 3D 点（双手 21 关键点 + 3 腕朝向点，头部 4 点），15 步 stride 3（约 4.5 s）。
- 微调：rank 8、scaling 16 的 LoRA，5 条示范，2,000 步约 2 分钟。
- Steering：把选中示范动作加噪到当前噪声级，按 x_t ← w·a*_t + (1−w)·x_t 混入；w=0.3 最佳；示范按物体 3D 质心布局的最近邻选取。

**对 wiki 的映射：** [`paper-steertuning.md`](../../wiki/entities/paper-steertuning.md)；与 [LoRA](../../wiki/concepts/lora.md)、[扩散策略](../../wiki/methods/diffusion-policy.md) 互链。

## 摘录 2：结果

- 12 个任务 / 12 个新环境（9 个在租来的 Airbnb、3 个在实验室）总成功率 71%。
- 实验室三任务（Trash / Flower / Bread）：Steertuning 9/10、7/10、9/10（均值 8.3/10）；仅 steering 2/10、2/10、3/10；仅微调 2k 步 0/10；全量微调 150k 步 7/10、8/10、8/10（文中另写 128k，原文不一致）。
- 消融：点云编码 PointNeXt 6.0 / PTv3 7.7 / Utonia 8.3；解码器 UNet 7.3 / DiT 8.3；示范选择 VINN 5.3 / 腕位姿 5.7 / 物体布局 8.3（均为 /10）。
- 整机：YOR 移动底盘 + Wuji 20 自由度灵巧手；Gemini ER 1.6 做任务分解，OK-Robot 式导航。

**对 wiki 的映射：** 评测与结论小节。
