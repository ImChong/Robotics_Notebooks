# Kelly 轨迹优化教程通读（直接配点法）

> 来源归档（原文通读 · 支撑 `wiki/methods/trajectory-optimization.md`）

- **标题：** An Introduction to Trajectory Optimization: How to Do Your Own Direct Collocation
- **作者：** Matthew Kelly（Cornell University）
- **类型：** paper / tutorial / trajectory-optimization
- **原文：** SIAM Review 59(4), 2017, DOI 10.1137/16M1062569；PDF 取自作者主页 <https://www.matthewpeterkelly.com/research/MatthewKelly_IntroTrajectoryOptimization_SIAM_Review_2017.pdf>（44 页，pdftotext 全文）
- **一句话说明：** 以四个算例（块移动、倒立摆小车摆起、五连杆双足步态、最小功块移动）讲清梯形 / Hermite–Simpson 直接配点的转写、初始化、网格细化、调试，并概述其他方法。
- **核对范围：** 通读正文 §1–§10（含 §5 实践、§6–§8 三个算例、§9 背景与方法对比、表 1–2）及附录 A–E 的推导与参数；附录 F（五连杆双足运动方程与撞击映射推导）与参考文献仅浏览，未逐式核对。未阅读 Betts 专著、Posa 等接触隐式原论文，相关表述仅依赖 wiki 页既有内容。
- **通读：** 已完成

## 核心摘录（均出自原文）

- **问题形式（§1.4，式 1.1–1.9）：** 目标含 Mayer 项（边界）与 Lagrange 项（路径积分），合称 Bolza 形式；约束含动力学、路径约束、边界约束、状态 / 控制上下界、时间界。直接法先离散再优化（转写 transcription），得到非线性规划（NLP）；若约束线性且目标二次则退化为 QP，更适合实时（§1.6）。
- **梯形配点（§2–§3）：** 用梯形求积把 x_{k+1}−x_k = ½h_k(f_k+f_{k+1}) 作为配点约束；控制是线性样条，状态是二次样条（控制为 n 阶样条则状态为 n+1 阶）。
- **Hermite–Simpson（§4）：** Simpson 求积给出 x_{k+1}−x_k = h_k/6(f_k+4f_{k+½}+f_{k+1})，中点状态由插值式 x_{k+½}=½(x_k+x_{k+1})+h_k/8(f_k−f_{k+1}) 得到；状态为三次 Hermite 样条（一阶连续）。分离形式（中点状态作决策变量）适合段数少，压缩形式适合段数多。
- **初始化（§5.1）：** NLP 只保证局部最优，初值决定落入哪个局部极小甚至能否可行；常用起点是状态空间直线插值；复杂问题先简化（换简单目标、去掉大部分约束、动力学换成纯运动学）求解，再逐步加回约束并以上一解作初值。
- **网格细化与误差（§5.2–§5.3）：** 先粗网格 / 低阶，再加密或升阶（hp 自适应）；用动力学残差 ε(t)=ẋ−f 在各段的积分 η_k 估计转写误差。
- **调试（§5.4–§5.5）：** 解不唯一（奇异弧）时加小正则项；abs / min / max 等非光滑、随机数、变步长积分、迭代求根、查表插值、含接触求解器的时间步进仿真器都会破坏"一致性"（每次调用算术操作序列相同、输出随输入光滑），需改用松弛变量、平滑、定步长等；判断全局最优只能靠多初值与多转写方法比对的启发式。
- **算例结果（§6–§8）：** 小车摆起（Hermite–Simpson，25 段）FMINCON 5.91 s / 71 迭代；五连杆双足周期步态，5 段粗网格 0.96 s、25 段细网格 21.3 s（解析梯度，数值梯度 79.8 s）；最小功块移动的解是 bang-bang（不连续），样条难以精确表示，需网格加密、平滑或多相法；松弛变量精确但变量多、收敛慢，tanh 平滑越轻越准但越刚性。
- **背景与方法对比（§9）：** 轨迹优化给出开环解，须配稳定控制器；动态规划给出闭环策略并可全局最优，但受维数灾难。间接法精度高但收敛域小、需初始化伴随变量。单次打靶适合控制简单、路径约束少；多次打靶更稳健；配点法适合动力学与控制需同等精度、控制结构未知（如机器人关节力矩）；正交配点谱收敛；DDP 向后传播最优控制；多相法处理已知接触序列，穿越接触（through-contact，互补约束）处理接触序列未知的情形。软件见表 1–2（GPOPS-II、PSOPT、DIRCOL、DIDO；FMINCON、SNOPT、IPOPT）。

## 对 wiki 的映射

- [`wiki/methods/trajectory-optimization.md`](../../wiki/methods/trajectory-optimization.md)

## 参考来源

- Kelly, M. (2017). *An Introduction to Trajectory Optimization: How to Do Your Own Direct Collocation*. SIAM Review 59(4). <https://epubs.siam.org/doi/10.1137/16M1062569>
