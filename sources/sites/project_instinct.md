# Project Instinct（清华 IIIS / 上海期智研究院）

- **类型**：研究项目站群 / 原始摘录（官网子页 + 公开预印本入口）
- **收录日期**：2026-05-12
- **主站**：<https://project-instinct.github.io/>

## 一句话

面向 **人形全身控制** 的公开研究线，站点叙事强调从 **算法、仿真环境、数据整理到实机部署** 的一体化框架，并展示 **感知增强跑酷 / 野外徒步**、**全身接触丰富动作（Shadowing）** 等子课题。

## 为什么值得保留

- 把 **「感知 locomotion」与「通用动作跟踪」** 两条范式在工程上拧在一起，对理解当前人形 **非足端接触 + 深度外感受 + 单策略多地形** 路线有参照价值。
- 子页提供 **可直接引用的 BibTeX / arXiv 号**，便于与仓库内 [Sim2Real](../../wiki/concepts/sim2real.md)、[Whole-Body Control](../../wiki/concepts/whole-body-control.md)、[Terrain Adaptation](../../wiki/concepts/terrain-adaptation.md) 等页交叉。

## 主站摘录（一级叙事）

来源：<https://project-instinct.github.io/>（2026-05-12 抓取）

- 标题口径：**Instinct-Level intelligences for whole-body control**。
- 摘要口径：提出覆盖 **算法、环境、数据集整理与部署** 的统一框架，目标为 **人形机器人上的本能级（Instinct-Level）智能**。
- 列出的研究子项（页面小节标题级）：
  - **Embrace Collisions** — Humanoid Shadowing for Deployable Contact-Agnostics Motions（CoRL 2025）
  - **Deep Whole-Body Parkour**（In Submission）
  - **Hiking in the Wild** — Scalable Perceptive Parkour Framework for Humanoids（In Submission）
- 站点还强调：**机载深度**、**位置自校正**、**从跟踪策略蒸馏下游用例** 等展示块（以页面视频/小标题为准，本库不转存媒体）。

## 子页 1：Embrace Collisions

来源：<https://project-instinct.github.io/embrace-collisions/>

- **作者（页面）**：Ziwen Zhuang；Hang Zhao。
- **会议**：Conference on Robot Learning (CoRL) 2025。
- **问题设定**：传统人形多将机体视为 **双足移动操作平台**，接触集中在 **脚与手**；人体还会用 **躯干、臀、背** 等与环境接触（坐、起身、地面滚动），带来 **不可预测接触序列**，使 **MPC 难以前瞻规划**；零样本 Sim2Real RL 又依赖 **GPU 刚体仿真加速** 与 **碰撞检测简化**；缺乏 **极端躯干运动** 数据时，**终止条件、运动指令与奖励** 设计变难。
- **方法口径（页面摘要）**：离散 **高层运动指令** → **实时低层电机动作**；在 **GPU 加速刚体仿真** 中训练 **全身控制策略**，在 **随机接触、大基座旋转、不可行指令** 下仍跟踪指令；面向实机 **实时** 部署。
- **独立预印本入口（便于 curator 对照全文）**：arXiv:2502.01465 — <https://arxiv.org/abs/2502.01465>  
- **会议论文 proceedings 线索（开放获取）**：PMLR v305 条目 *Embrace Contacts* 系列同一工作线 — <https://proceedings.mlr.press/v305/zhuang25b.html>（标题与 Embrace Collisions 站点略有措辞差异，以 PDF 为准）。

## 子页 2：Deep Whole-Body Parkour

来源：<https://project-instinct.github.io/deep-whole-body-parkour/>

- **作者**：Ziwen Zhuang, Shaoting Zhu, Mengjie Zhao, Hang Zhao（单位标注：清华 IIIS、上海期智）。
- **核心主张**：现有路线二分——**感知 locomotion** 擅地形但多限于 **足式步态**；**通用动作跟踪** 能复现复杂技能但常 **忽视环境可通行性**。本文将 **外感受** 并入 **全身动作跟踪闭环**，在 **非平坦地面** 上完成 **腾跃、侧滚翻** 等高动态 **多接触** 行为；用 **单一策略** 跨 **多种动作与多种地形特征** 训练，强调 **把感知接入控制回路** 的收益。
- **预印本**：arXiv:2601.07701 — <https://arxiv.org/abs/2601.07701>（页面内 BibTeX 与站点一致）。

## 子页 3：Hiking in the Wild

来源：<https://project-instinct.github.io/hiking-in-the-wild/>

- **标题**：Hiking in the Wild: A Scalable Perceptive Parkour Framework for Humanoids。
- **问题**：野外徒步需要从 **反应式本体感知** 过渡到 **主动外感受**；**建图类方法** 易受 **状态估计漂移** 影响（页面举例 LiDAR 对 **躯干抖动** 敏感）；部分端到端方案 **可扩展性 / 训练复杂度** 不足，**虚拟障碍** 等技巧 **个案化**。
- **机制（页面用语）**：
  - **落脚点安全**：可扩展 **Terrain Edge Detection** + **Foot Volume Points**，抑制 **边沿灾难性打滑**。
  - **Flat Patch Sampling**：通过 **可行导航目标采样** 缓解 **奖励投机（reward hacking）**。
- **学习范式**：**单阶段强化学习**，**原始深度 + 本体** 直接映射到 **关节动作**，**不依赖外部状态估计**；宣称野外 **全尺寸人形** 上复杂地形 **最高约 2.5 m/s**，并 **开源训练与部署代码**（以仓库 README 为准）。
- **预印本**：arXiv:2601.07718 — <https://arxiv.org/abs/2601.07718>（页面 BibTeX 一致）。

## 对 wiki 的映射

- 升格页面：[wiki/entities/project-instinct.md](../../wiki/entities/project-instinct.md)

## 参考链接（索引）

- 项目主页：<https://project-instinct.github.io/>
- Embrace Collisions：<https://project-instinct.github.io/embrace-collisions/> · arXiv:2502.01465：<https://arxiv.org/abs/2502.01465>
- Deep Whole-Body Parkour：<https://project-instinct.github.io/deep-whole-body-parkour/> · arXiv:2601.07701：<https://arxiv.org/abs/2601.07701>
- Hiking in the Wild：<https://project-instinct.github.io/hiking-in-the-wild/> · arXiv:2601.07718：<https://arxiv.org/abs/2601.07718>
