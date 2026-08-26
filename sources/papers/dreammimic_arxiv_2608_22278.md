# DreamMimic（世界模型辅助的视觉全身 Mimic）

> 来源归档（ingest）

- **标题：** DreamMimic: Learning Visuomotor Whole-Body Loco-Manipulation via World Model
- **类型：** paper
- **原始链接：** <https://arxiv.org/abs/2608.22278>
- **机构：** Independent（Jie Yin）；清华大学（Xingyu Lai）
- **会议：** IROS 2026（仓库标题标注）
- **项目页：** <https://dreammimic.github.io/> — 归档见 [`sources/sites/dreammimic-github-io.md`](../sites/dreammimic-github-io.md)
- **代码：** <https://github.com/DreamMimic/DreamMimic> — 归档见 [`sources/repos/dreammimic.md`](../repos/dreammimic.md)
- **入库日期：** 2026-08-26
- **一句话说明：** 把特权 HOI 教师（InterMimic 式 specialist→generalist）蒸馏成深度+分割学生；Dreamer 风格 RSSM 不作规划，只提供预测表征与 H=3 动作条件潜对齐；PCG 按师生奖励比调节教师 rollout 比例。OMOMO 视觉学生 Succ. **92.2%**；代码截至入库日仅占位 README。

## 核心摘录（MVP）

### 1) 视觉全身 loco-manip：特权信号部署不可用

- **摘录要点：** 接触丰富人形 loco-manipulation 在部分可观测下难学。多数全身控制依赖物体位姿、交互图、接触标签；部署 visuomotor 拿不到这些。单阶段像素 Dreamer 在 HumanoidBench 类高维任务上弱。DreamMimic 目标：学生只看本体、紧凑目标（物体位姿 + 短视界机体轨迹）与世界模型特征。
- **对 wiki 的映射：**
  - [DreamMimic](../../wiki/entities/paper-dreammimic.md) — 问题与 POMDP 设定。
  - [VisualMimic](../../wiki/entities/paper-notebook-visualmimic.md) — 同为视觉全身蒸馏，接口是关键点而非 RSSM。
  - [Loco-Manipulation](../../wiki/tasks/loco-manipulation.md) — 任务族。

### 2) RSSM 表征 + 多步潜蒸馏 + PCG

- **摘录要点：** 教师走 InterMimic 特权 RL。学生不直接吃原图：CNN+MLP 编码深度/分割与本体 → RSSM \((h_t,s_t)\)；策略读 \(h_t\) 与辅助头（奖励、特权、接触、物体状态）。\(\mathcal{L}_{\text{latent}}\) 从同一后验分叉，用师生均值动作各滚 H=3 步先验并对齐。PCG：按 \(\pi=\hat r_S/(\hat r_T+\epsilon)\) 把教师环境比 \(\rho\) 从 \(\rho_{\max}\) 衰减到 \(\rho_{\min}\)，模仿系数固定。另有 InterMimic 式失败参考缓冲课程。
- **对 wiki 的映射：**
  - [DreamMimic](../../wiki/entities/paper-dreammimic.md) — 方法与损失。
  - [InterMimic](../../wiki/entities/paper-bfm-15-intermimic.md) — 特权教师配方。
  - [World Action Models](../../wiki/concepts/world-action-models.md) — 对照：此处 WM 服务蒸馏，不是 Joint WAM。

### 3) OMOMO / BEHAVE 数字与跨本体定性

- **摘录要点：** 同教师、同学生输入下，DreamMimic 在 SMPL-X OMOMO 达 Succ. **92.2%** / Time 184.18 / \(E_r\) 5.4 cm / \(E_o\) 8.8 cm，高于 ResNet/ViT/CNN + DAgger+RL（72.6–76.5%）与单阶段 Dreamer（0%）。去多步潜蒸馏掉到 **70.6%**。BEHAVE 上 PCG 与 naive annealing 成功率同为 **72.7%**，但跟踪误差与持续时间更好。G1 42 DoF 与 Isaac Gym→Lab 仅为定性；无真机。感知是仿真 GT 深度+分割。
- **对 wiki 的映射：**
  - [DreamMimic](../../wiki/entities/paper-dreammimic.md) — 指标与消融读法。

### 4) 开源状态（截至 2026-08-26）

- **摘录要点：** 项目页写 **Code (Coming soon)**；GitHub `DreamMimic/DreamMimic` 仅 `README.md` + MIT LICENSE，正文 “Codes coming soon!”。按步骤 2.5：**宣称将开源 / 占位仓**，无可运行训练或推理入口。
- **对 wiki 的映射：**
  - [DreamMimic](../../wiki/entities/paper-dreammimic.md) — 源码时序图标不适用。

## 当前提炼状态

- [x] arXiv 方法/实验与项目页对齐
- [x] GitHub 占位状态已核查
- [x] wiki 映射：`wiki/entities/paper-dreammimic.md` 新建
