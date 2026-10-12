# Point It, Strike It 论文来源归档

## 元数据

- **题目：** Point It, Strike It: Direction-Conditioned Dynamic Manipulation of Deformable Linear Objects
- **作者：** Yi Yang, Xiang Fei, Lehong Wang, Zilin Dai, Ruogu Li, Jiting Cai, Liyao Chang, Xinyi Yang, Henry Kou, Ruijie Fu, Lu Li, Howie Choset
- **单位（据 HTML 版）：** 卡内基梅隆大学机器人研究所（CMU RI）；上海交通大学；哈佛大学 SEAS（Zilin Dai）
- **预印本：** https://arxiv.org/abs/2610.09573
- **HTML：** https://arxiv.org/html/2610.09573v1
- **版本：** v1，2026-10-07，cs.RO
- **项目页：** https://deformx.github.io/DeformY （归档见 [deformy-rope-project.md](../sites/deformy-rope-project.md)）
- **代码：** 截至 2026-10-12 未发布（项目页与 arXiv 均无仓库、数据链接，也无"即将发布"声明）
- **核查日期：** 2026-10-12

## 论文要解决的问题

单次挥绳击打：目标是指定绳端 3D 位置，以及绳端到达时的方向。同一目标有很多种挥法可达（多解），回归会平均不兼容的动作；仿真与真实绳的参数不一致，还会造成落点和到达方向偏差。

## 方法与训练信息

1. **目标与动作**：目标 g=(p*, d*)，p* 为 3D 位置，d* 为到达方向（cosθ, sinθ），到达角为绳端速度在以臂为中心球面的目标切平面上的投影。动作向量 w（21 维：5 个关节各 4 个等时间间隔路点 + 挥动时长，最后一个腕关节不含），经截断三次样条解码为关节轨迹，时长按关节速度限制调整。
2. **DeformX 2.0 仿真器**：GPU 加速，稳定的 Cosserat 杆求解器（位置与旋转分开更新，闭式姿态），对每个绳段加垂直于切线的二次交叉流阻力；Isaac Sim 中每 60 Hz 控制步 100 个子步、16 次约束迭代。1 m、20 段绳在 RTX 4090 上：DeformX 单 CPU 线程仿真/墙钟比 0.22，DeformX 2.0 为 2.3（1 环境）、1,780（1,024 环境）、4,710（8,192 环境）；论文称每 GPU 每小时超过 100 万条轨迹、比 DeformX 快 20,000 倍以上。
3. **TRACE 数据生成**：基于交叉熵方法（CEM），代价 J = L_target + L_spd + L_lim + L_curv；max 归一化损失 ≤1 的挥动被接受；曲率项偏好绳伸直、绳端运动平滑的可重复挥法；每个新目标从绳端轨迹最接近的已存挥动热启动，起点是一次手调并真机验证的挥动，另有生成树顺序精修与事后重标记。工作空间覆盖率：TRACE 98.3%，带示例的 CEM 49.5%。
4. **条件流匹配策略**：6 个 adaLN-Zero 残差块、宽度 512、约 11.3M 参数；目标与流时间注入每个块；推理 64 个噪声样本、50 步 Euler，按 J 取最优。训练集规模正文未给出（原文未报告）。
5. **RECAP 校准**：每根绳执行 8 次固定校准挥动，动捕记录绳端轨迹；用 CEM 辨识 11 个仿真参数；增量 CFM 策略（在扰动参数的仿真绳与装置上训练）输出修正动作，运行时生成 8 个修正，连同名义动作在拟合后的仿真中重放，执行代价最低者；每个目标单次开环击打。

## 论文报告的结果

- **真机**：墙装 UR5e 握 0.65 m 刚性杆 + 1.0 m 绳；工作空间为 30° 锥、半径 1.6–2.3 m、最大高 2.5 m；三根绳（绿色编织绳、黑色塑料链、黄色编织绳）；每根绳每种目标 25 个目标 × 3 次挥动。
- **3D 位置目标**（三绳均值）：SR@5 为 Base 72%、Baseline（Wiggle&Go 式参数估计）74%、RECAP 87%；SR@10 为 92/94/98%；平均距离 4.5/4.2/3.1 cm。
- **4D（位置+方向）目标**：SR@5 为 28/28/52%；SR@10 为 50/51/79%；平均距离 10.6/10.2/5.6 cm；平均角误差 8.2°/8.5°/5.1°。4D 成功要求绳端到达方向误差 ≤10° 且沿该方向速度至少 1 m/s。
- **仿真（5,700 个留出目标）**：本文策略一次采样 73.5%、best-of-64 92.1%、偏差 3.6 cm、角误差 2.4°；朴素 FM 67.9% / 88.9% / 5.3 cm / 3.0°；回归 21.4% / 24.0 cm / 12.4°；最近邻 33.6% / 9.2 cm / 6.4°。
- **数据生成消融**：CEM+示例覆盖 49.5%、best-of-64 68.5%；TRACE 无示例覆盖 96.8%、best-of-64 83.9%；TRACE 覆盖 98.3%、best-of-64 92.1%、平均曲率代价 0.33。

## 复现与解释边界

- 仅单次挥绳击打；校准后开环，不能响应挥动中的扰动；不含持续接触、障碍或多阶段。
- 依赖仿真生成与排序候选，真实系统超出仿真动力学范围时可能退化。
- 页面文字有两处明显笔误（关节速度限 "22 rad/s"、到达速度 "11 m/s"），数值不采信。
- 无代码、无数据。

## Robotics_Notebooks 映射

- 详情节点：wiki/entities/paper-point-it-strike-it.md
- 汇总页：wiki/overview/frontier-manipulation-2026-10-05-10-09.md
- 项目页归档：sources/sites/deformy-rope-project.md

## 一手来源

- arXiv 摘要：<https://arxiv.org/abs/2610.09573>
- arXiv HTML：<https://arxiv.org/html/2610.09573v1>
- 项目页：<https://deformx.github.io/DeformY>
