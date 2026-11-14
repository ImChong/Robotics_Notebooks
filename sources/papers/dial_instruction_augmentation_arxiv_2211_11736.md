# DIAL — Robotic Skill Acquisition via Instruction Augmentation with Vision-Language Models

> 原始论文与项目页完整核对归档（注意区分 2026 年同名但不同研究的 `dial_arxiv_2603_29844.md` 与 `loco_manip_161_survey_147_dial.md`）

- **标题：** Robotic Skill Acquisition via Instruction Augmentation with Vision-Language Models
- **类型：** paper
- **会议 / arXiv：** RSS 2023 · [arXiv:2211.11736v3](https://arxiv.org/abs/2211.11736)（2023-07-01；首版 2022-11）
- **作者与机构：** Ted Xiao*, Harris Chan*, Pierre Sermanet, Ayzaan Wahid, Anthony Brohan, Karol Hausman, Sergey Levine, Jonathan Tompson（Robotics at Google, University of Toronto）
- **项目页：** <https://instructionaugmentation.github.io/>
- **代码 / 数据开源状态：** **未开源**（项目页 `dial.html` 仅提供论文 PDF 与轨迹视频 `https://youtu.be/lbaqjJJMAFg`；未公开训练代码、FT-CLIP 权重或 80k 内部厨房遥操作数据）
- **入库日期：** 2026-11-14
- **复核日期：** 2026-11-14
- **核对范围：** `2211.11736.txt` 全文（Sec. I–VI 及 Appendix A–H，共 1759 行：含 Table I–VII、Fig. 1–12、Listing 1–3 与 60 条未见测试指令全表）及项目官网 `dial.html`（`https://instructionaugmentation.github.io/` 全文与链接）
- **通读：** 已完成
- **沉淀到 wiki：** [`wiki/methods/dial-instruction-augmentation.md`](../../wiki/methods/dial-instruction-augmentation.md)

## 一句话摘要

DIAL（Data-driven Instruction Augmentation for Language-conditioned control）在仅占 3.5%（2,800 条轨迹 / 5,600 条人工事后描述 $D_A$）的少量标注上对比微调 CLIP，再用微调后的 VLM 为 80,000 条无事后自由文本的轨迹 $D_B$ 自动筛选候选指令生成弱监督数据集 $D_C$，最终使 RT-1 架构 BC 策略在 60 条完全未见指令上从 42.5% 提升至 60.0%。

## 核心三阶段流程（Sec. III & Appendix C/F/G）

1. **阶段 1 — 少量众包标注上微调 VLM（FT-CLIP）：**
   - 从 OpenAI ViT-B/32 CLIP 权重初始化；对每条轨迹只取首帧 $s_0^n$ 与末帧 $s_T^n$（640×512 第三视角肩部 RGB），经图像编码器 $I_{\text{enc}}$ 后拼接再过单隐层 MLP $f_\theta$（隐维 200）得到 512 维轨迹视觉嵌入 $z_{s^n}$。
   - 文本侧用 $T_{\text{enc}}$ 编码众包人工给出的事后描述 $l^n$，在批大小 64 上做对称交叉熵（InfoNCE）对比训练 100k 步，联合微调 $I_{\text{enc}}$、$T_{\text{enc}}$ 与 $f_\theta$（10% 验证集选最优 checkpoint）。
   - Appendix A.2（Fig. 10）证实：**冻结原生 CLIP** 或 **从随机权重从头训练 CLIP** 的标注检索准确率均显著劣于「OpenAI 预训练初始化 + $D_A$ 微调」。
2. **阶段 2 — 基于视觉上下文的指令重标记（$D_B \to D_C$）：**
   - 候选池 $\mathcal{L}$ 共 **18,719** 条文本（去重后合并自 $D_A$ 中 9,393 条众包描述 $L_{CC}$ 与 GPT-3 按 Listing 1 扩写的 9,336 条含「可控幻觉」空间/场景变化指令 $L_{GPT3}$）。
   - 用余弦相似度 $d(z_l, \hat{z}_s)$ 对 80,000 条遥操作轨迹 $D_B$ 打分排序，对比两种截断策略：
     - **Top-$k$**（$k \in \{1, 3, 10\}$ → 80k / 240k / 800k 条重标样本）
     - **Min-$p$**（按 CLIP 温度 $\alpha$ 做 softmax 后保留概率 $\ge p$ 的指令，$p \in \{0.10, 0.20, 0.30\}$ → 128,422 / 38,516 / 17,013 条样本；最多取 $1/p$ 条，置信度不足可返回 0 条）。
   - 在 $p=0.20$ 下，CLIP 从 18,719 条候选中实际选中 **3,675** 条唯一指令（86.3% 来自 $L_{CC}$，13.7% 来自 $L_{GPT3}$）。
3. **阶段 3 — 下游语言条件行为克隆（RT-1 + FT-CLIP）：**
   - 在 $[D_A, D_B, D_C]$ 或子集上训练 RT-1 架构策略（连续关闭环像素到离散动作 token），关键改动是把 RT-1 原有的 USE 文本编码替换为 **FT-CLIP 文本编码器**（Appendix A.3 Table V 证明 FT-CLIP 表征在各数据配置下均优于 USE 与冻结 CLIP）。

## 关键实验结果（Sec. IV–V, Table I–V）

- **数据规模：** 真实办公厨房 7-DoF 移动操作机器人，$D_B$ 含 80,000 条仅带 551 类结构化命令（Table VI）的遥操作轨迹；仅抽 2,800 条（3.5%）各标 2 条自由语言描述构成 5,600 对 $D_A$。
- **未见测试集（Table I & VII）：** 60 条完全不在 $D_A \cup D_B \cup D_C$ 中的新指令（共 **> 1,300** 次真机评估）：**34 条 Spatial**（左/中/右等空间位置）、**16 条 Rephrased**（颜色/泛称改写）、**10 条 Semantic**（如移开孤立物体、用海绵擦苹果等原集未包含语义）。
- **主结果（Table II）：**
  - 仅 $D_A$（`IL`）：总体 **27.5%**（Spatial 30.0% / Rephrased 40.0% / Semantic 7.7%）。
  - 仅 $D_B$（`RT-1`）：总体 **33.8%**（38.0% / 40.0% / 15.4%）。
  - `RT-1 + IL`（$D_A + D_B$，无增强）：总体 **42.5%**（46.0% / 60.0% / 15.4%）。
  - `DIAL`（$D_A + D_C$，去掉 $D_B$ 结构标签）：总体 **47.5%**（58.0% / 46.7% / 15.4%）。
  - `DIAL`（$D_A + D_B + D_C$，$p=0.20$）：总体 **60.0%**（Spatial **68.0%** / Rephrased **66.7%** / Semantic **30.8%**），相对无增强基线提升 **>41%**。
- **对比无视觉信息的增强基线（Table III）：** 同样基于 $\{D_A, D_B\}$ 增强时，不看图像的高斯噪声（33.8%）、词级同义替换（27.5%）与纯 GPT-3 句级改写（30.0%）全部**低于**不增强的 `None`（42.5%），只有借助视觉接地的 `DIAL` 提升至 **60.0%**。
- **准确率–规模权衡（Table IV & Fig. 7）：** 离线人工评估发现 CLIP 第 20 位候选准确率已跌至 20.0%；增加 $k$ 或降低 $p$ 会引入噪声——$k=10$（800k 样本、57.0% 准确率）策略反而跌至 35.0%，而 **Min-$p = 0.20$**（38,516 样本、68.8% 重标准确率）取得最高 60.0% 策略成功率（$p=0.30$ 76.0% 准确率→56.3%）。

## 局限与后续方向（Sec. VI）

- 仅依赖首末两帧 $(s_0, s_T)$，既无法描述依赖中间过程时序一致性的技能，也受限于现有轨迹中真实包含的底层运动技能集合（若轨迹未示范相应动作，单纯改写指令仍会失败，见 Fig. 9）。
- 依赖事先收集或 LLM 扩写的静态候选池 $\mathcal{L}$ 做检索打分，尚未接入开放式视频字幕生成或强化学习策略协同迭代。

## 参考链接

- 论文全文（本地归档源）：`2211.11736.txt`（<https://arxiv.org/abs/2211.11736>）
- 项目官网（本地归档源）：`dial.html`（<https://instructionaugmentation.github.io/>）
- 轨迹演示视频：<https://youtu.be/lbaqjJJMAFg>
