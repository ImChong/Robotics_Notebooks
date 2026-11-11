# PA-SaR（arXiv:2611.11648）

> 来源归档（paper）

- **标题：** PA-SaR: Perception-Aware Learning, Control, and Coordination for Multi-Robot Search and Rescue
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2611.11648>
- **PDF：** <https://arxiv.org/pdf/2611.11648v1>（v1，2026-11-08，cs.RO）
- **代码：** https://github.com/svenschoonebeek/Perception-Aware-Search-and-Rescue
- **作者 / 机构：** Dimsa、Grouev、Keerberg、Laplagne、Schutgens（并列一作）、Schoonebeek、Jamshidnejad；代尔夫特理工大学（TU Delft）
- **入库日期：** 2026-11-11
- **一句话说明：** 把「真值环境 / 机器人各自感知 / 模拟受害者行为」分开建模的多机器人搜救仿真框架，含 MPC 与 RL 局部控制、感知安全屏蔽、事件触发的集中式任务分配与受害者信任感知的引导；全部结论限于论文自建仿真。
- **核对范围：** 读完 v1 PDF 正文第 1–4 节（引言、2.1–2.3 建模与控制器、3.1–3.6 全部实验、结论、数据可用性声明）与附录 A（表 A1–A4）、附录 B（表 B1 节选参数，含环境生成、传感 / 火情、MPC 权重、拍卖系数、受害者信任参数）；图仅看图注与正文引用，未逐图细看像素；参考文献只核对了与本页相关的条目标题，未逐条追原文；公式 (1)–(17) 读了定义与符号，未做独立推导复算；代码仓库 README 仅一行标题，仓库内容未核实。
- **通读：** 已完成

## 开源状态

- 论文「Data availability statement」写明源码在 https://github.com/svenschoonebeek/Perception-Aware-Search-and-Rescue（论文原文，第 4 节之后）。
- 本次检查：仓库 README 只有标题 `# Perception-Aware Search and Rescue`；本会话无法读取仓库文件树，代码、权重、数据的实际开放程度**未核实**。未运行任何代码。

## 核心摘录

1. **框架（第 2 节）：** 观测—决策—执行循环；区分真值环境 G(k)（式 1，含可通行 / 易损 / 火 / 受害者 / 机器人层）、机器人感知 Q_r（式 2，含置信度层与探索层）、模拟受害者（GT–BDI 改编，信任 θ_v，式 7）。
2. **局部控制（2.3.1–2.3.2）：** 名义 MPC、固定管 TMPC、SDD-TMPC（网格版：只要估计的位移概率为正就施加一格管，管大小是二值的）；循环 PPO 策略（11 通道观测）+ 感知屏蔽 + TMPC 备份。论文明确「未提供递归可行性证明，也无无条件真值安全保证」。
3. **监督协调（2.3.3）：** 集中式监督者维护共享信念（后到报告覆盖，不做平均或置信加权融合），前沿目标 + 修改版顺序单项拍卖（式 12–13），事件触发重分配；受害者引导用「接近 / 带领」两策略，由引导分 s_guide（式 15）切换。
4. **实验（第 3 节）：** 全部为自建离散网格仿真；匹配种子；无真机。
5. **关键数字：** 见 wiki 页「核心原理」与「与文章说法的核对」两节的表号。

## 对 wiki 的映射

- [paper-pa-sar](../../wiki/entities/paper-pa-sar.md)
