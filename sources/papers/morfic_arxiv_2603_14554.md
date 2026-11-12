# MorFiC: Fixing Value Miscalibration for Zero-Shot Quadruped Transfer（arXiv:2603.14554）

> 来源归档（paper）

- **标题：** MorFiC: Fixing Value Miscalibration for Zero-Shot Quadruped Transfer
- **类型：** paper
- **作者：** Prakhar Mishra, Amir Hossain Raj, Xuesu Xiao, Dinesh Manocha（马里兰大学学院公园分校 / 乔治梅森大学）
- **arXiv：** <https://arxiv.org/abs/2603.14554>（v1，2026-03-15，cs.RO）
- **PDF：** <https://arxiv.org/pdf/2603.14554v1>
- **代码：** 文中未给出仓库或项目页链接（全文检索无 github / http 链接）
- **入库日期：** 2026-11-12
- **一句话说明：** 在 PPO 上只改 critic：用由机器人物理与控制参数编码出的形态潜变量 z，对 critic 隐层做 FiLM 调制，使单一共享 critic 能表示不同形态的价值，仅在 Go2 单源机器人（带形态随机化）上训练后零样本迁移到其他四足。
- **核对范围：** 已下载 v1 PDF（11 页），核对首页标题与编号后逐段通读正文（摘要、第 I 至 IX 节）、全部表（Table I 至 VII）、附录 A（奖励项）、B（超参数与架构）、C（域随机化）与参考文献列表；图 1 至图 5 的图注读过，图 3、图 4、图 5 看了渲染页面（只定性判读，未取原始数据点）。未读：作者未提供代码、补充视频或项目页，故未核对实现；未做数值复现。
- **通读：** 已完成

## 开源状态

- 文中未给出代码或项目页链接（2026-11-12 全文检索确认）；未单独联网核查作者主页或仓库，故只能写「论文未给出」。

## 核心摘录

1. **问题（§I、§IV-A）：** 作者主张多形态训练中共享 critic 对「同一输入、不同形态下不同未来回报」的价值目标取平均，产生形态相关的价值偏差，进而使 PPO 的优势估计失准（这是论文的论证与经验观察，文中未给出理论证明）。
2. **方法（§IV-C/D、附录 B）：** 形态向量 m（11 维，每维归一化到 [-1, 1]，§IV-B）经 2 层 MLP（dm→128→64，ELU）得 z。actor 与 critic 都接收 z，但 critic 经 FiLM 注入：每个 critic 隐层 `h ← h(1+γ)+β`，FiLM 头 64→128→2H，末层权重与偏置零初始化，FiLM 输出缩放 0.1（Table VII）。PPO 其余不变（§IV-E，Table VI：γ=0.99，λ=0.95，ε=0.2，lr 1e-3 自适应，KL 目标 0.01）。
3. **训练设置（§V-E、Table I、附录 C）：** IsaacGym，Go2 单模板加形态随机化（质量、关节限位，Table I，均匀分布）；4000 并行环境，dt=0.005，总计 4 亿步，RTX 4090 笔记本 GPU；域随机化含摩擦 [0.5, 1.25]、每 15 s 推力最大 1.0 m/s、动作延迟历史最多 6 步。
4. **仿真零样本结果（§VI-A、Table IV）：** 目标机器人 Go1、A1、AlienGo、B1、Cheetah、ANYmal。MorFiC 的稳定平均速度 v̄：Go1 2.171，A1 2.286，AlienGo 0.993，B1 0.245，Cheetah 1.193，ANYmal 0.364 m/s；训练机 Go2 上 1.911（低于 PPO 变体 1 的 1.942 与变体 3 的 2.174）。摘要中的「A1 +16.1%、Cheetah 约 2 倍、B1 约 5 倍」与 Table IV 相符（对比 A1 的 PPO 变体 1，Cheetah 与 B1 的 PPO 变体 3）。
5. **对比变体（§V-C）：** PPO（无形态条件）、仅 actor 得 z、actor 与 critic 均拼接 z、MorFiC（critic 经 FiLM）。四者奖励、形态与域随机化、网络参数一致（§VI-D）。另在 Table III 与 URMA、URMAv2、LocoFormer、ManyQuadrupeds、GenLoco、BodyTransformer、RL2AC、Embodiment Scaling 做特性对照表（不是同条件数值对比）。
6. **价值校准证据（§VI-B、图 3、图 4）：** 图 3(a) 用「解释方差 EV」、(b) 用「优势噪声代理」按目标机器人比较；图 4 为 Go1 部署中的奖励与价值轨迹。文中均为定性读法，无显著性检验。
7. **真机（§VII）：** Go1 上约 1.4–1.7 m/s、10 次试验成功率超过 80%；Go2 约 1.1–1.3 m/s，同一份 Go2 策略权重；另以 A1 为训练机零样本部署到 Go1 与 Go2，速度区间相近，成功率 70%（试验次数文中未写）。
8. **局限（§VIII-B）：** 极端分布外形态（ANYmal，尤其 B1）迁移显著下降；只研究四足与速度跟踪运动，未评估双足或人形。

## 对 wiki 的映射

- [paper-morfic](../../wiki/entities/paper-morfic.md)
