# Transformer Transformer: A Unified Model for Motion-Conditioned Robot Co-design

> 来源归档（ingest）

- **标题：** Transformer Transformer: A Unified Model for Motion-Conditioned Robot Co-design
- **类型：** paper（运动条件机器人共设计 / 跨具身控制 / 扩散 Transformer）
- **状态：** arXiv preprint（2026-07）
- **原始链接：**
  - 项目页：<https://transformer-transformer.github.io/>
  - arXiv abs：<https://arxiv.org/abs/2607.25798>
  - PDF：<https://arxiv.org/pdf/2607.25798> · 项目页镜像 <https://transformer-transformer.github.io/static/paper.pdf>
  - 代码：<https://github.com/real-stanford/transformer-transformer>
  - 权重/数据（小包）：<https://real.stanford.edu/transformer-transformer/>
  - 训练 Zarr（HF）：<https://huggingface.co/datasets/hqhuy/transformer-transformer>
- **作者：** Huy Ha, C. Karen Liu, Shuran Song
- **机构：** Stanford University；Columbia University
- **入库日期：** 2026-07-30
- **核对范围：** arXiv:2607.25798v1（2026-07-28，cs.RO）PDF 全文 26 页逐段读完：正文 §1–§5、附录 §7（容量消融）、§8（奖励定义与表 2–18）、§9（RoboToken 细节、数据、推理超参、表 19–20）及图注；参考文献只看了与本文直接相关的条目，未逐条查阅被引论文；真机视频（项目页）与代码仓库本次未重新核对（相关事实沿用下方「相关资料」归档，标注为沿用）。
- **通读：** 已完成
- **一句话说明：** 用 **RoboTokens** 统一编码刚体关节机器人的 embodiment / state / action，训练单一 **DiT（Transformer Transformer）**，同一权重既可做 **motion→完整机体生成**，也可做 **跨具身跟踪控制**；推理时用自有动力学预测把任意可微奖励转成引导信号（**Dynamics Self-Guidance**），相对 CMA-ES 在三设计空间上更快、更高回报，并真机制造优化后的 ALOHA 抛布展开（跟踪误差 −73%、峰值关节速度 −30%）。

## 逐节要点（2026-11 通读后重写，章节号为论文章节）

### 1) 问题与贡献（§1）

- 目标：给定一组目标末端轨迹和一组用户奖励，自动生成「完整机体」（运动学、几何、动力学参数）。奖励可同时依赖机体（尺寸、质量）和行为（关节速度、跟踪精度）。
- 三项贡献：RoboTokens 统一表征；掩码建模下同一网络兼任 generator / critic / controller；以「动力学建模」而非单奖励为训练目标，推理时把动力学预测转成奖励分数去引导机体扩散。
- 脚注 1：第一个 Transformer 指变形机器人，第二个指自注意力架构。

### 2) RoboToken（§2.1，附录 §9.1、表 19）

- 5 类机体 token：link、固定关节、滑动/转动关节、球关节、电机；状态 token 4 类（固定关节无状态）加动作 token。关节用两个 link ID 指向所连 link，电机用 joint ID 指向所控关节，无电机指向的关节即被动关节（文中以 Cassie 腿机构为例）。
- 一致性：统一变换约定、拆分集总惯量、把变换折叠进关节 token（§9.1）。属性编码：布尔/枚举/整数用二进制，跨数量级的连续量用 log 或 signed log。
- 紧凑性：Fig. 3 称比 MJCF 文本 token 紧凑 27–110 倍；§9.1 说明文本 token 数用 GPT-4o tokenizer 对 MJCF 计数。§3：MuJoCo Menagerie 11 个机器人（质量 0.65 kg 到 67.5 kg，DoF 6 到 35）均可 token 化为长度 28–101 的序列。
- 范围（§2 Scope）：刚体关节、基元（primitive）几何；任务表示限于 UMI 末端轨迹；结构强度、外观类奖励不在范围内。

### 3) 架构与训练（§2.2，§9.2，表 20）

- DiT + DDIM；每种 token 类型一个线性投影加 ID 学习式位置嵌入；SE(2) 平面增广；每样本从 episode 子采样 8 个时间步（motion-to-robot 随机采样，control 连续采样）。
- 两种用法：Motion-to-Robot 同时扩散机体与动力学 token，Zeroth-Order Optimizer 并行扩散 n 个候选并用奖励排序取最优；Cross-Embodiment Control 条件于机体、当前状态与目标运动，预测专家动作。
- 扩散步数：motion-to-robot 用 100，control 用 5（§9.2）。基础超参（表 20）：hidden 256、8 层、4 头、LR 5e-3、batch 64、50 epoch x 16384 步；大模型（§7）hidden 与头数翻倍、12 层、LR 1e-4，参数量 63.6M，小模型 11.6M。

### 4) Dynamics Self-Guidance（§2.3，§9.4）

- 对可微奖励，把其对机体 token 的梯度按 classifier-guided DDIM 注入每步去噪；n 条并行引导扩散后由模型排序取最优。作者对比：先前工作需要另训动力学模型 [23] 或可微仿真器 [24]。
- §3 Guidance Substitutes for Search：单样本下 DSG 在多轨迹双臂行上领先 Zeroth Order 242 到 341 奖励；采样预算足够大时两者无法区分（附录 §8）。
- 超参（§9.4）：η=1.0 的随机 DDIM 显著改善 DSG（对 Zeroth 无影响）；引导尺度 ViperX 50、四足 100、双臂 0.2，多轨迹 ViperX 500。§8.4/8.5：机体 token 奖励（尺寸、质量）的梯度远大于动力学 token，需把引导系数降到 0.005（报告奖励仍用 0.1）。

### 5) 数据与设计空间（§2.4，§9.3）

- 目标运动：UMI 扩展夹爪采集 76 条单末端轨迹（56 训练 / 20 验证），另用 UMI 双臂洗碗示范（验证 26 条）。
- 三个设计空间：ViperX 300S 固定基（运动学）、四足操作臂（动力学/WBC）、移动双臂（任务复杂度）；ALOHA 设计空间用于真机。
- 控制数据：非腿式用 Mink DiffIK（80 ms 固定前瞻）；腿式对 7 个二元离散选择各训一个 PPO 专家共 128 个，每个约 16 小时 A100（§9.3.2）；连续变化参数进入策略观测。
- 数据规模（§9.3.4）：ViperX 3.8M episode / 2B 步 / 38 万机体；四足 1.3M / 500M / 13 万；移动双臂 50K / 69M / 5 千。
- 数据质量（§9.3.3）：滤掉短于 100 步的 episode；ViperX 与 ALOHA 空间的训练机体偏向高性能——对训练轨迹跑 CMA-ES（种群 5、3 代），把采到的所有机体都加入训练数据。

### 6) 结果（§3，附录表 2–18）

- 对比 CMA-ES（9 个种子，RTX 4090）：Fig. 5 (a,b,d)（ViperX 单/多轨迹、双臂）上更快且不差；论文正文未声称四足（c）占优。表 4/8/11/15（四足单轨迹）中 CMA-ES 的位置误差（1.9/2.0/2.1/2.3 cm）均低于 Zeroth/DGS（2.5 到 4.6 cm）。
- 表 5：移动双臂多轨迹 tracking-only，CMA-ES 11505.2 s，Zeroth 20.7 s，DGS 30.8 s；位置误差 1.7 / 1.8 / 1.7 cm。
- 表 3：ViperX 多轨迹，CMA-ES 位置误差 7.1 cm / 663.8 s，DGS 2.4 cm / 43.5 s。
- 测试时算力（§3）：并行种子增多通常提升设计，但作者明说「性能并不稳健地随测试时算力增长」，DSG 可精炼约一分钟。
- 自验证控制（§3，Fig. 8）：四足上学到的控制器与 RL 专家奖励 Pearson r=0.53，有离群。§7.1 表 1（移动双臂，650 episodes）：Mink 奖励 1064.3，大模型 957.0，小模型 884.9；位置误差 4.8 / 5.9 / 6.6 cm。§7.2：大模型更好，同样并行样本数下运行时间约 4 倍。
- 真机（§3，Fig. 9–10）：ALOHA 按 Tracking Velocity 奖励优化后做布料抛展，最大关节速度 2.57 到 1.82 rad/s（-30%），位置跟踪误差 13.0 到 3.5 cm（-73%）；设计为更长连杆、手臂倒挂在工作空间后方做下摆而非上抛。论文只报告了这一次对比，未给重复次数或方差。

### 7) 局限与作者自述（§3、§4 Limitations、§5）

- 生成设计只在训练分布内插，不外推（不能从仅四足训练生成六足，也不能超出训练链长范围）（§3）。
- 几何限基元；未含场景/物体/触觉；数据生成（尤其腿式控制）成本高（§4 Limitations）。
- 预测奖励与实际奖励的相关性在四足上低于 ViperX/双臂（Fig. 7）。

## 与清单摘要及旧页面说法的核对备忘

- 清单摘要「Diffusion transformer over unified embodiment, state and action tokens that generates robot designs for target motions」与论文一致（摘要与 §2）。
- 摘要里「humanoids」：正文只在 Menagerie 的 11 个机器人 token 化中涉及人形，没有人形设计空间的优化实验；三个优化设计空间是 ViperX、四足、移动双臂。
- 旧页面「ALOHA2」：论文只写 ALOHA（引用 [60,61]），未写 ALOHA2。
- 旧页面「约 1 分钟后平台期」：论文原话是性能并不稳健地随算力增长，未写明 1 分钟处饱和。

## 相关资料（交叉核对）

| 资料 | 链接 | 备注 |
|------|------|------|
| 项目页归档 | [transformer-transformer-github-io.md](../sites/transformer-transformer-github-io.md) | Demonstrate/Generate/Validate、真机对比、局限 |
| 代码归档 | [transformer-transformer.md](../repos/transformer-transformer.md) | 入口脚本、ckpt、HF 数据 |
| 共设计对照 | [Shape Your Body](shape_your_body_arxiv_2606_00702.md) | VGDS：价值梯度搜连续参数；本文：扩散生成完整机体+控制 |
| 硬件语境 | [ALOHA wiki](../../wiki/entities/aloha.md) | 真机抛布平台 |
| UMI | 项目引用 Chi et al. RSS 2024 | 示范轨迹来源 |

## 当前提炼状态

- [x] 项目页 + arXiv HTML/摘要方法与结果已摘录
- [x] 开源核查：GitHub 全栈 + 官方 ckpt/数据（2026-07-30）
- [x] 论文 PDF 全文通读（2026-11）
- [x] wiki 映射：`wiki/entities/paper-transformer-transformer.md`
