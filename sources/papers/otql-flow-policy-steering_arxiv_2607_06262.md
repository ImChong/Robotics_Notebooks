# OTQL：Optimal Transport Q-Learning for Flow Policy Steering and Acceleration（arXiv:2607.06262）

> 来源归档（paper）

- **标题：** Optimal Transport Q-Learning for Flow Policy Steering and Acceleration
- **作者：** Andreas Sochopoulos, Esmeralda S. Whitammer, Nikolaos Tsagkas, João Moura, Michael Gienger, Sethu Vijayakumar（爱丁堡大学；Honda Research Institute Europe）
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2607.06262>（v1，2026-07-07，cs.RO）
- **PDF：** <https://arxiv.org/pdf/2607.06262v1>
- **项目页：** <https://ansocho.github.io/otql-flow>（只含论文链接与作者主页链接，未见代码仓库链接；本页未找到官方代码仓库）
- **会场：** 论文页脚为 CoRL 格式，具体收录信息以 OpenReview 为准（本环境未能打开）
- **入库日期：** 2026-11-12
- **一句话说明：** 用「优势加权的条件最优传输流匹配」（wCOT-CFM）对次优 flow 策略做 RL 后训练，同时让流变直，只需 2 到 3 步积分，免去蒸馏。
- **核对范围：** 通读 arXiv v1 PDF 全文：摘要、§1 引言、§2 背景、§3 方法（含算法 1）、§4 实验（表 1、表 2、图 2 到 5）、§5 结论与局限、附录 A（OT 定义）、B（相关工作）、C（消融，文字部分）、D（实现细节与全部超参表）、E（硬件）。图 5 渲染后目视核对，图 6（消融曲线）与图 1、图 2、图 3 只读了正文与图注描述，未逐点读数；参考文献列表只作略读，未逐条核对被引文献。未运行任何代码，也没有可运行的官方仓库。
- **通读：** 已完成

## 核心摘录

1. **问题（摘要、§1）：** flow / diffusion 策略受限于演示质量与推理慢；已有 flow 的 RL 方法多需多步积分，或靠蒸馏得到单步策略（如 FQL），训练昂贵。
2. **方法（§3.1）：** 在 minibatch 熵正则 OT（式 2）里，(a) 把条件拼进代价（式 3、4，系数 alpha）近似条件 OT，(b) 用 exp(lambda·A) 重加权目标边缘（式 5），源条件也从同一倾斜分布采样；从最优耦合 Γ* 采 (z, a) 对，再用标准 CFM 损失（式 1）训练。该损失称 wCOT-CFM。
3. **后训练流程（§3.2，算法 1）：** 初始化 v_theta 为基策略；缓冲区 B 先用基策略 rollout 填充；Q 用 TD 损失训练（目标动作由 Euler 积分采得）；优势 A = Q(s,a) − 对 K 个 flow 采样动作的 Q 均值；用 A 作负能量做 wCOT 耦合并更新流。动作按 chunk 处理，Q 函数是 chunked 的。
4. **两个实用修正（§3.2）：** 权重截断 w_max；对较大 chunk 且 Q 尖峰的任务，推理与 Bellman 目标处从 N 个候选动作里取 Q 最大者（拒绝采样，N=5）。
5. **实验设置（§4、附录 D）：** OGBench 8 个环境 x 5 个任务（共 40）离线 RL；OGBench 的 cube-double 与 humanoidmaze-medium 各一个任务做离线到在线；LeRobot 的两个 MuJoCo 任务在线；4 个真实任务，KUKA IIWA 14 + Robotiq 2F-85，2 个 RealSense D415。
6. **离线结果（表 1）：** OTQL 8 环境平均 0.59，QAM 0.55，QAM-F 0.54，DSRL 0.48，FQL 0.44；OTQL 使用 NFE=2，QAM 等归入 NFE>=10 组。
7. **真实机器人（§4.3、表 2、图 5）：** 单任务基策略仅 10 条演示，每次 10 个 rollout 后离线训练 20k 步，累计 50 个 rollout，平均成功率 36% 到 86%，NFE 10 到 3；SmolVLA 在 tape in penholder 与 move knight 上 14/30 到 24/30、9/30 到 22/30，用 30 个 rollout，VLM 冻结。
8. **作者自述局限（§5）：** 要求策略是 neural-ODE 生成模型，不能当黑箱；维度升高时 minibatch OT 误差变大，加速与适应变弱；沿用真实 RL 的人工打标与人工复位问题。

## 对 wiki 的映射

- [paper-otql-flow-policy-steering](../../wiki/entities/paper-otql-flow-policy-steering.md)
