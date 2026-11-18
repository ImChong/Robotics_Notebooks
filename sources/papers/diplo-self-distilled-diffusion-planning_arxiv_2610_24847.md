# DiPLO: Self-distilled Diffusion Planning for Loco-manipulation（arXiv:2610.24847）

> 来源归档（ingest）｜论文原文核对，不复制全文

- **标题：** DiPLO: Self-distilled Diffusion Planning for Loco-manipulation
- **中文标题：** DiPLO：面向人形移动操作的自蒸馏扩散规划
- **类型：** paper
- **作者：** Shafeef Omar, Sebastian Sanokowski, Elham Daneshmand, Dian Yu, Daniel Cremers, Cordelia Schmid, Majid Khadiv
- **机构：** Technical University of Munich（慕尼黑工业大学）、McGill University（麦吉尔大学）、Inria（法国国家信息与自动化研究所）、Institute for Advanced Study, TUM
- **原始链接：** <https://arxiv.org/abs/2610.24847>
- **HTML 全文：** <https://arxiv.org/html/2610.24847v1>
- **PDF：** <https://arxiv.org/pdf/2610.24847v1>
- **项目：** <https://atarilab.github.io/diplo-website/>
- **版本：** arXiv:2610.24847v1，2026-10-17
- **入库日期：** 2026-11-18
- **核对范围：** 通读 arXiv HTML v1 正文 §I–VI（运动表示、SBTO 数据增广、DiT/SO(3) 扩散、PPO tracker、SD:DPO、DA:PPO、表 I–VII 消融及真机）及附录 A–M（网络配置、目标和奖励、观测、超参数与泛化）；参考文献仅核查相关方法条目。与官方项目页及其 GitHub 网站 README 交叉核对。未运行训练/推理、未独立复现实验。
- **通读：** 已完成

## 核心摘录

### 1. 问题：少量示范下，兼顾动作可执行性与目标泛化

人形 pick–carry–place 同时涉及行走平衡、抓持接触及目标物体的位姿变化。直接从头强化学习难探索；仅按 IK 重定向的人体示范又不保证动力学可行。预训练时使用强运动学任务先验可使模型稳定，但部署时不能要求事先知道每个目标对应的完整动作序列。

### 2. 数据与预训练（§IV-A–B）

- OmniRetarget 来源 **29 段**约 **2.7 分钟**人类运动；先 IK 重定向，再由 DynaRetarget 的 SBTO 对不同初态和随机种子精炼，形成动力学可行轨迹；结合镜像及物体四分之一转对称性，扩至 **7,184** 条机器人–物体运动。
- DiT 规划器以机器人/物体状态历史、物体目标及训练时的 IK 参考为条件，预测 **64 帧 / 1.28 s** 未来机器人–物体状态，采用 **10 步去噪**；旋转在 SO(3) 上扩散；附录 B：4 层、4 头、宽 256、约 **5.1M** 参数。
- 单一 PPO tracker 在 mjlab 中学跟踪可行轨迹，输出关节参考位置的残差；50 Hz 跟踪器，200 Hz 关节 PD。与扩散规划器分别训练。

### 3. 两段后训练（§IV-C–D）

- **SD:DPO（无任务奖励蒸馏）**：student 使用状态历史和目标，将 IK 条件置空；在自身闭环 rollout 收集的状态上，用同一噪声运动与同一去噪步比较有/无 IK 参考的 denoising 预测；EMA self-teacher 提供停止梯度的目标。另有数据预训练损失与 anchor 约束，避免已学到的有参考行为漂移。**该阶段无需奖励函数或 critic**。
- **DA:PPO（目标对齐）**：冻结 tracker，把生成器每次去噪视作扩散增强 MDP 的决策，通过仿真执行 chunk 得到目标/运动学跟踪奖励，借 PPO 与最大熵正则把回报传回去噪过程。部署时仍只使用状态历史和目标，不输入 IK 运动学先验。

### 4. 实验：明确分开 survival 与任务成功率（§V）

论文定义：**survival** 是未摔倒且未掉落物体的 rollout 比例；**task success** 是 *在存活 rollout 中*，末端物体位置偏差不超过 **15 cm**、朝向偏差不超过 **30°** 的比例（即条件成功率），不要误称无条件端到端成功率。

| 评测 / 模型 | Survival | Task success |
|---|---:|---:|
| pick-place Student（预训练，无 IK 参考） | 57.7 ± 9.2% | 45.4 ± 9.0% |
| Student + SD:DPO | 100.0 ± 0.0% | 55.1 ± 6.0% |
| Student + SD:DPO + DA:PPO | 99.5 ± 0.5% | **98.8 ± 1.2%** |
| Kicking：最终模型 | 100.0 ± 0.0% | 68.1 ± 8.3% |
| Push/Pull：最终模型 | 100.0 ± 0.0% | 52.7 ± 6.5% |

- 消融：SO(3) 旋转扩散 survival **96.5%** vs 欧氏 6D 的 **85.9%**（表 II，预训练）。
- 同条件下 motion-state diffusion + tracker 相对 action diffusion：**96.5% vs 87.5%** survival；不能跨训练条件直接比较（表 III）。
- 未见目标泛化：0.5–5 m 目标位置/角度网格上 **88%** 任务成功率（论文 §V-G）；项目页 20 m 行走是展示案例，非此网格成功率的统计样本。
- 真机 G1 29 DoF：同一目标重复 **10/10**、不同目标 **5/5** 成功（表 VII）；规划器 GPU 进程约 **12.5 Hz**，tracker 经 CPU ONNX Runtime **50 Hz**，PD **200 Hz**；运动捕捉系统提供基座位置和箱体位姿，尚非仅用机载视觉的闭环。

### 5. 局限与开放边界

- 真机仍依赖外部 mocap 感知物体位姿；仅报告有限目标与尝试次数，没有多物体形态和开放世界场景的大样本证明。
- 后训练依赖仿真及冻结的跟踪器，奖励可用 IK 参考或目标反馈；不同奖励下的成功率和跟踪误差不等价。
- 截至 2026-11-18，**项目主页 Code/Dataset 仍写 soon**；官方 GitHub 的 diplo-website 是网站源码，**不是训练/部署算法实现**。该网站 README 链到一个 Hugging Face 数据集地址，但本次无法核实其实际公开访问及内容，故不据此宣称数据已开源。

## 对 wiki 的映射

- [DiPLO 独立论文实体](../../wiki/entities/paper-diplo-self-distilled-diffusion-planning.md)
- [Loco-Manipulation Stage 4](../../roadmap/depth-loco-manipulation.md)
- 同团队上游：[DynaRetarget](../../wiki/entities/paper-notebook-dynaretarget-dynamically-feasible-retargeting-us.md)

## 当前提炼状态

- [x] 论文全文、附录与项目页核对，单独区分模拟/真机数据
- [x] 主干 Mermaid 论文方法流程图
- [x] 源码开放状态核查；因未公开算法代码，源码运行时序图不适用
- [ ] 本地复现与真机脱离 mocap 的评测（无官方代码）
