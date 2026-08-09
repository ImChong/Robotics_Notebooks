# CommNav（arXiv:2607.01044）

> 来源归档（ingest）

- **标题：** Robots Ask the Way: Communication-Enabled Social Navigation
- **缩写：** **CommNav**（任务）/ **COMM**（通信模块）/ **Habitat 3.0c**（仿真扩展）
- **类型：** paper / social-navigation / human-robot-communication / embodied-ai / deep-rl
- **arXiv：** <https://arxiv.org/abs/2607.01044>
- **会议：** IEEE/RSJ International Conference on Intelligent Robots and Systems（IROS）2026（官方仓 bibtex）
- **项目页：** 无独立 `*.github.io`；官方入口为 GitHub README
- **代码：** <https://github.com/S4b3/CommNav>（截至 2026-08-09：**under preparation**，无可运行训练/评测入口）
- **作者：** Valentino Sacco\*、Luca Scofano\*、Indro Spinelli、Fabio Galasso（\* equal contribution）
- **机构：** 罗马第一大学（Sapienza University of Rome）
- **入库日期：** 2026-08-09
- **一句话说明：** 提出 **Communication-enabled Social Navigation（CommNav）**：多居住者场景下机器人主动询问非目标路人，获取「是否见过 / 何时 / 位置 / 方向 / 说话者轨迹」等稀疏线索以定位目标；扩展 Habitat 3.0 → **Habitat 3.0c**，并用预训练 **COMM** 模块把结构化或自然语言线索回归为类 PointGoal 的目标估计，接入 DDPPO；相对无通信基线 Episode Success **+10 pp**；口语指令与结构化数据在 ES 上统计接近。

## 核心论文摘录（MVP）

### 1) 问题与总贡献（Abstract / §I）

- **链接：** <https://arxiv.org/abs/2607.01044>
- **核心贡献：** 既有社交导航侧重避障与轨迹适应，缺少**主动向人打听**的机制。CommNav 把「找人」写成多智能体导航：机器人接近视野内路人并请求线索，而非穷举房间搜索。贡献三点：
  1. 任务形式化 CommNav + 通信内容 \(\mathcal{S}=\{x_h,x_t,\mathbf{x}_l,\mathbf{x}_d,\mathbf{x}_p\}\)（相对说话者的 egocentric 坐标）；
  2. 仿真 **Habitat 3.0c**（多人体、信息交换、ORCA、\(p=0.25\) 忽略机器人）；
  3. **COMM** 模块：通信代理任务预训练后冻结/微调，把稀疏交互映射为 \(\hat{\mathbf{x}}_{\text{target}}\)，再与 ResNet 视觉流融合进导航策略。
- **对 wiki 的映射：**
  - [CommNav 论文实体](../../wiki/entities/paper-commnav.md)
  - [Habitat-Sim](../../wiki/entities/habitat-sim.md)
  - [社会导航（正负示范）](../../wiki/entities/paper-notebook-learning-social-navigation-from-positive-and-neg.md)

### 2) COMM 架构与语言路径（§III-B / Fig. 2）

- **链接：** arXiv HTML Methodology
- **核心贡献：**
  - 结构化路径：轨迹 \(\mathbf{x}_p\in\mathbb{R}^{100\times3}\) 经 Spatio-Temporal MLP（3 头、\(H=64\)）嵌入，与 \([x_h,x_t,\mathbf{x}_l,\mathbf{x}_d]\) 拼接后 4 层 MLP 回归目标位置。
  - 语言路径 \(\text{COMM}_{\mathcal{L}}\)：QWEN3-8B 将 \(\mathcal{S}\) 口头化为 \(\mathcal{L}\)，冻结 BERT 编码后同样回归 \(\mathbf{x}_{\text{target}}\)；约 7000 条合成指令。
  - 策略侧：先无通信训稳健视觉导航（DDPPO），再对齐 COMM；无交互时 COMM 填占位，策略回退纯视觉。
  - 代理数据：基线训练约 60M 步收集 **2.4M** 通信实例。
- **对 wiki 的映射：**
  - [CommNav 论文实体](../../wiki/entities/paper-commnav.md) — 流程总览
  - [强化学习](../../wiki/methods/reinforcement-learning.md)

### 3) Habitat 3.0c 与主结果（§III-C / Table I）

- **链接：** Experiments
- **核心贡献：** 单人 Habitat 3.0 上 DDPPO/SDA 的 ES 约 0.40/0.43；多人 3.0c 无通信掉到 0.14/0.16。仅打开 interaction 的 DDPPO **无增益**（ES 仍 0.14），说明「有传感器≠会用」。**COMM** 达 S **0.78**、ES **0.24**（相对 DDPPO+Int **+10 pp** ES），CR 降至 0.51。\(\text{COMM}_{\mathcal{L}}\) ES 0.20；人类口语 \(\text{COMM}_{\mathcal{L}(Human)}\) ES **0.23±0.01**，与结构化 COMM 接近。消融（Table II）：去掉 \(x_h\) 或 \(\mathbf{x}_p\) 对 ES 打击最大。三人全/半速（Table III）通信仍抬高 S/ES。
- **对 wiki 的映射：**
  - [iCrowdNav](../../wiki/entities/paper-icrowdnav.md) — 人群视觉避障对照（不问路）
  - [HUMEMBR](../../wiki/entities/paper-humembr.md) — 找人任务对照（多日记忆，非即时问路）
  - [导航纵深路线](../../roadmap/depth-navigation.md)

### 4) 开源与部署边界（§IV-D / GitHub）

- **链接：** <https://github.com/S4b3/CommNav>
- **核心贡献：** 论文写「Code can be found at」GitHub；仓库 README（2026-07-01）明确 **under preparation**，训练/评测/Habitat 3.0c 配置/COMM 实现/生成通信数据均 **coming soon**。评测为仿真、无噪声传感与完美 egocentric grounding；真机需人分割/识别等 oracle 假设外模块。单轮交互、同意主体伦理边界见文末。
- **对 wiki 的映射：**
  - [CommNav 仓库归档](../repos/commnav.md)
  - [Sim2Real](../../wiki/concepts/sim2real.md)

## 对 wiki 的映射（汇总）

- 沉淀实体页：[`wiki/entities/paper-commnav.md`](../../wiki/entities/paper-commnav.md)
- 仓库归档：[`sources/repos/commnav.md`](../repos/commnav.md)
- 互链参考：[Habitat-Sim](../../wiki/entities/habitat-sim.md)、[iCrowdNav](../../wiki/entities/paper-icrowdnav.md)、[社会导航笔记](../../wiki/entities/paper-notebook-learning-social-navigation-from-positive-and-neg.md)、[HUMEMBR](../../wiki/entities/paper-humembr.md)、[导航·SLAM 栈总览](../../wiki/overview/navigation-slam-autonomy-stack.md)、[导航纵深](../../roadmap/depth-navigation.md)、[强化学习](../../wiki/methods/reinforcement-learning.md)

## BibTeX（官方仓）

```bibtex
@inproceedings{sacco2026robots,
  title     = {Robots Ask the Way: Communication-Enabled Social Navigation},
  author    = {Sacco, Valentino and Scofano, Luca and Spinelli, Indro and Galasso, Fabio},
  booktitle = {IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year      = {2026}
}
```
