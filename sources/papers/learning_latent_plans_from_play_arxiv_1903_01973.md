# Learning Latent Plans from Play（arXiv:1903.01973）

> 来源归档（ingest · 全文通读）

- **标题：** Learning Latent Plans from Play
- **作者：** Corey Lynch、Mohi Khansari、Ted Xiao、Vikash Kumar、Jonathan Tompson、Sergey Levine、Pierre Sermanet（Google Brain）
- **发表：** CoRL 2019（3rd Conference on Robot Learning, Osaka, Japan）
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/1903.01973>（v2, 20 Dec 2019）
- **论文所列项目页：** <https://learning-from-play.github.io/>
- **入库日期：** 2026-11-19
- **核对范围：** 通读 arXiv:1903.01973v2 全文（§1–§5 及附录 A.1–A.5、Algorithm 1–2、Fig. 9 网络结构详图）；未在本次离线读取中访问外部网站 `learning-from-play.github.io` 或拉取其代码/数据集压缩包。
- **通读：** 已完成
- **一句话说明：** 从人类无任务标签、无分段、无复位的遥操作「玩耍（play）」连续日志中自监督学习目标条件潜在规划空间（seq2seq CVAE），在 18 项桌面视觉操控任务上以单一通用策略超越 18 个逐任务训练的专家 BC 策略，并自发涌现抗扰与失败后重试行为。

## 开源与公开口径（步骤 2.5 核对说明）

- 论文摘要明确声明：`Videos, code and data are available at learning-from-play.github.io`。
- 本次归档仅基于论文全文 PDF/TXT；项目主页 HTML 与代码压缩包未在本批次本地抓取，具体代码分支与数据许可需打开项目页进一步核查。

## 核心方法与技术细节（均出自原文）

### 1. 为什么用 Play 数据替代传统 LfD 与随机探索（§1–§3）

- **Task-agnostic control 问题定义：** 将「任务」从离散标签推广为连续的当前–目标状态对 $(s_c, s_g)$，求单一目标条件策略 $\pi_\theta(a|s_c, s_g)$。
- **Play 数据性质：** 人类操作员凭好奇心连续遥操作（MuJoCo HAPTIX VR），不切分任务片段、不打任务标签、不复位场景。Play 不是随机动作，而是受物体 **affordance**（可操作性）驱动的蓄意目标行为：在人类脑中给定 $(s_c, s_g)$，从多峰先验 $p(b|s_c, s_g)$ 中抽取高阶行为并执行。
- **量化覆盖度对比（Fig. 2c/d，App. A.4.3）：** 将 11 维物体交互状态空间（方块 6D 位姿 + 推拉门/抽屉连续开合 + 3 个按钮传感器）每维离散成 10 个分箱并统计访问量：在同等 90 分钟采集时长内，**Play 数据覆盖的交互空间区域是 18 任务专家演示（1800 条）的 4.2 倍，是随机探索的 14.4 倍**；并在扩展至 7 小时的采集过程中几乎保持线性增长。

### 2. Play-GCBC 与 Play-LMP（§3.1–§3.2，Algorithm 1–2，App. A.1–A.2）

- **自监督切片机制：** 从连续日志 $\mathcal{D}=\{(O_1, a_1),\dots,(O_T, a_T)\}$ 随机采样长度 $\kappa \sim U(\kappa_{\text{low}}, \kappa_{\text{high}})$ 的窗口 $\tau$，首帧作为当前状态 $s_c$、末帧作为保证可达的合成目标 $s_g$，中间动作作监督标签。
- **Play-GCBC 的局限：** 直接训练 $\pi_{\mathrm{GCBC}}(a_t|s_t, s_g)$ 面临 **多峰性（multimodality）** 冲突——同一 $(s_c, s_g)$ 间存在多种合理路径（例如手抓方块放杯、抛入、弹入），均值/简单条件拟合会拉扯梯度。
- **Play-LMP（Seq2Seq CVAE）：** 将「推断高层计划」与「闭环执行计划」解耦，使策略头只需执行给定潜计划 $z$ 下的单峰控制：
  1. **Plan recognition（计划识别后验 $q_\phi(z|\tau)$）：** 双向 RNN `Venc`（前向 2048 + 后向 2048 → 256 维对角高斯均值 $\mu_\phi$ 与方差 $\sigma_\phi^2$），读取整段编码后序列 $\tau^*=\Phi(\tau)$ 并重参数化采样 $z$；**仅用于训练，测试时丢弃**。
  2. **Plan proposal（计划提议条件先验 $p_\theta(z|s_c, s_g)$）：** 4 层 2048 维 ReLU MLP `CGenc`，仅看当前与目标状态 $(s_c, s_g)$ 预测 256 维高斯分布 $(\mu_\psi, \sigma_\psi^2)$，表征连接 $(s_c, s_g)$ 的全部可行高阶行为分布。
  3. **Plan & goal-conditioned policy（动作解码器 $\pi_{\mathrm{LMP}}(a_t|s_t, s_g, z)$）：** 2 层 2048 维 RNN，输出 8-DoF 末端动作（3D 位置 + 3D 欧拉角 + 2 指角度，各维量化为 256 bins）的 **离散化 Logistic 混合分布（MODL）**。
- **训练目标（Eq. 2–4）：** $\mathcal{L}_{\mathrm{LMP}} = \mathcal{L}_\pi + \beta \mathcal{L}_{\mathrm{KL}}$，其中 $\mathcal{L}_\pi = -\frac{1}{\kappa}\sum_t \log \pi_{\mathrm{LMP}}(a_t|s_t, s_g, z)$，$\mathcal{L}_{\mathrm{KL}} = \mathrm{KL}(\mathcal{N}(\mu_\phi,\sigma_\phi^2)\,\|\,\mathcal{N}(\mu_\psi,\sigma_\psi^2))$；采用 $\beta < 1$ 避免 VAE 后验坍塌（posterior collapse）。
- **视觉编码与目标设计（App. A.2，Fig. 9）：** 像素实验输入 `299×299×3` RGB + 8 维本体感知；图像经 3 层卷积（`8×8 s4` → `4×4 s2` → `3×3 s1`）+ `spatial softmax`（64 个 2D 空间关键点）+ `512 → 64` 全连接得到 64 维视觉特征，与 8 维本体特征拼接为 72 维 `visuo-proprio` 状态；**目标 $s_g$ 仅使用 64 维视觉特征**（不依赖目标时刻的关节/夹爪内状态），方便用户仅用目标图像指定任务。
- **测试期闭环推理（§3.2，Fig. 1b）：** 以 **30 Hz** 读取观测并闭环采样动作 $a_t \sim \pi_{\mathrm{LMP}}(a_t|s_t, s_g, z)$，每 $\kappa = 32$ 步（约 **1 Hz**）用 $CG_{\mathrm{enc}}(s_c, s_g)$ 重新采样 $z$ 实现在线重规划（replan）。

### 3. 实验结果与涌现行为（§4，App. A.3–A.5）

- **环境与数据（Playground，App. A.3）：** 桌面含推拉门、抽屉、长方体方块、三色灯光按钮与垃圾桶；像素实验使用 7 h 无标注 play，状态实验使用至多 3 h play；对比基线包括 18 个分任务专家 BC（每任务 100 条演示、共 1800 条约 90 min）与 Multitask BC。
- **18 项任务平均成功率（Fig. 6a）：**
  - **像素输入（3 个随机种子均值±区间）：** `BC (labeled)` **66.5% ± 12.1**、`Play-GCBC` **58.7% ± 11.6**、**`Play-LMP` 69.4% ± 10.8**。
  - **真值状态输入：** `BC` **70.3%**、`Multitask BC` **66.2%**、`Play-GCBC` **77.9%**、**`Play-LMP` 85.5%**；单任务最高超越 BC 达 50 个百分点以上（Fig. 7）。
- **数据效率与抗扰（Fig. 8、Fig. 6b、App. A.4.1–A.4.2）：** 仅用 **30 分钟** 无标注 play 训练的 `Play-LMP`（**71.8%**）即可超过用 **90 分钟** 18 任务精选专家演示训练的 18 个 BC 策略（**70.3%**）；在初始末端位置加至多 `0.4 m` 扰动时，play 训练模型掉点远缓于专家 BC；测试中多次出现 **自发重试（emergent retrying）**（如先碰倒立柱方块后再拾起、推拉门三次尝试关上），以及未见任务标签却按功能任务（推门/抽屉/按钮/抓取）自然聚类的潜空间 t-SNE（Fig. 4）。
- **原文局限（App. A.5）：**
  1. 单一目标图像无法规定执行**方式**（如「缓慢拉抽屉」）；
  2. 训练/测试同房间分布假设，跨房间/新物体泛化留待未来工作；
  3. 要求 play 数据对各物体交互不要严重失衡；
  4. 后验与先验均为单峰高斯，对更复杂的潜计划分布表达力受限。

## 对 wiki 的映射

- [Learning from Play (LMP)](../../wiki/methods/learning-from-play-lmp.md) — **主沉淀页**
- [HER](../../wiki/methods/her.md) — 目标条件事后重标记对比（RL vs 自监督模仿）
- [Imitation Learning](../../wiki/methods/imitation-learning.md) — 无标签 play 范式坐标
- [Manipulation](../../wiki/tasks/manipulation.md) — 多技能桌面视觉操控
