# MimicDroid: In-Context Learning for Humanoid Robot Manipulation from Human Play Videos

> 来源归档（论文全文与附录通读核对 · `arXiv:2509.09769v1`）

- **标题：** MimicDroid: In-Context Learning for Humanoid Robot Manipulation from Human Play Videos
- **类型：** paper
- **状态：** ICRA 2026 Oral (Top-5%) · `arXiv:2509.09769v1`（2025-09-11）
- **原始链接：**
  - arXiv：<https://arxiv.org/abs/2509.09769>
  - 项目页：<https://ut-austin-rpl.github.io/MimicDroid/>
- **基准代码：** <https://github.com/UT-Austin-RPL/mimicdroid-robocasa/tree/latest>
- **数据集：** <https://huggingface.co/datasets/Rutav/MimicDroidDataset>
- **作者：** Rutav Shah\*, Shuijing Liu\*, Qi Wang\*, Zhenyu Jiang\*, Sateesh Kumar, Mingyo Seo, Roberto Martín-Martín, Yuke Zhu
- **机构：** The University of Texas at Austin; Amazon Consumer Robotics; NVIDIA
- **入库日期：** 2026-11-19
- **核对范围：** 通读 `arXiv:2509.09769v1` 全文与 Appendix A–F（§I–§VII & Tables I–V，1349 行）、项目页与 GitHub 基准仓库 `UT-Austin-RPL/mimicdroid-robocasa`。
- **通读：** 已完成
- **一句话说明：** MimicDroid（ICRA 2026 Oral，arXiv:2509.09769）完全脱离昂贵遥操作，仅从连续无标注人类玩耍视频（human play videos）通过相似行为片段检索构建 Meta-ICL 上下文–目标轨迹对，并结合 WiLoR 手腕姿态重定向与随机块遮挡训练出测试时 1–3 示例自适应的人形操作策略，真机 GR1 成功率约为 Vid2Robot 的两倍。

## 核心论文摘录（全文核对）

### 1. 数据与自监督 Meta-ICL 轨迹对构造

- **原始玩耍视频规模（Table II）：** 仿真（空间鼠标控制 Abstract 自由双手，3 视角，每段 20 min）8 小时 / 320,000 帧；真机（单目静止 RGB 相机单手自由玩耍，每段 10 min）1 小时 / 80,000 帧（正文 §IV 写作 45 min，附录 Table II 记 1 hour，80k 帧）。
- **动作与本体标签抽取：** 真机调用现成 3D 手部姿态估计 `f = WiLoR` 提取每帧腕部 6-DoF 位姿 + 夹爪开合信号 $h_t$，以 $t+1$（$k=1$）未来手部位姿作为动作标签 $a_t$。
- **上下文–目标配对检索：** 随机采样目标片段 $\sigma^{\text{tgt}}$，用冻结 `DINOv2`（$g$）逐帧视觉特征做时间平均池化并拼接动作序列组成片段表征 $\phi(\sigma) = [\frac{1}{T}\sum g(s_t), a_1, \dots, a_T]$，按余弦相似度检索 Top-$k$ 最相似片段作上下文 $\sigma^{\text{ctx}}$（实现细节 $k=10$，消融显示 $k=50$ 更佳）。

### 2. 模型结构与跨具身桥梁（Appendix C–D）

- **骨干继承：** 基于开源 **ICRT**（4 层 LLaMA 风格 Transformer，RMSNorm，RoPE，768 维；8×A5000 训练 200 epochs，AdamW `lr=5e-4`，batch size `256`）。
- **编码与训练设置：** 冻结 **CrossMAE** 视觉编码 + 3 层 Visual Attention Pooling（`768 → 256`）；输入频率降采样 `4×`，上下文由 3 段 context + 1 段 target 构成（仿真序列长 512 token，真机 256 token）；L1 行为克隆损失预测 `l=32` 步动作块（rollout 执行前 `16` 步）。
- **消除视觉与运动学鸿沟：**
  - **随机块遮挡（Visual Masking）：** 训练时以概率 `p=0.8` 随机擦除 1–16 个（附录记 0–16）面积 1%–4% 的图像块，防止策略过拟合人类手部外观，使 Abstract→GR1 迁移掉点从 `-17%` 降至 `-3%`（真机上 0.53 接近 EgoMimic SAM 手部掩码 0.58 且免分割器）。
  - **任务空间重定向：** 笛卡尔腕部位姿输出经逆运动学（`WHOLE_BODY_MINK_IK` 等）映射到人形手腕，保持任务意图。

### 3. 评测结果（L1 / L2 / L3 分级基准与真机）

- **仿真 12 任务三级基准（每级 4 任务，Table I & V）：**
  - L1（见过物体、见过环境，位置变化）：Abstract **0.73**，GR1 **0.59**（Vid2Robot 0.44 / 0.41；PEFT 0.47 / 0.29）。
  - L2（新物体、见过环境）：Abstract **0.39**（去掉视觉遮挡 0.51），GR1 **0.44**（Vid2Robot 0.40 / 0.23；PEFT 0.35 / 0.21）。
  - L3（新物体、新厨房布局，常需新运动轨迹如不同水龙头开关方向）：Abstract **0.27**，GR1 **0.26**（PEFT 在 L3 坍塌至 `0.00 / 0.01`）。
- **真机 GR1 评估（每任务 10 rollouts，4 任务/级）：** L1 / L2 / L3 分别为 **0.53 / 0.23 / 0.08**，约为 Vid2Robot（**0.28 / 0.08 / 0.00**）的两倍。
- **扩展与失败拆解：** 仿真数据从 64k/128k 扩至 320k 帧时 L1/L2/L3 持续上升；上下文示例从 1 增至 3 提升，4–6 个因训练上下文上限略降；失败主要由任务误判（**26%**）、抓取失败（**16%**）与其他如未完全合拢柜门（**8%**）构成。

## 对 wiki 的映射

- 主沉淀与升级：[`wiki/entities/paper-notebook-mimicdroid-in-context-learning-for-humanoid-robo.md`](../../wiki/entities/paper-notebook-mimicdroid-in-context-learning-for-humanoid-robo.md)
- 项目页归档：[`sources/sites/mimicdroid-ut-austin.md`](../sites/mimicdroid-ut-austin.md)
- 代码仓归档：[`sources/repos/ut-austin-rpl-mimicdroid-robocasa.md`](../repos/ut-austin-rpl-mimicdroid-robocasa.md)

## 当前提炼状态

- [x] `arXiv:2509.09769v1` 全文通读与核对完成
- [x] 支撑 `wiki/entities/paper-notebook-mimicdroid-in-context-learning-for-humanoid-robo.md`
