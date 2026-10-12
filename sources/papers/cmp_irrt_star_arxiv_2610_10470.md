# CMP-IRRT* 论文来源归档

## 元数据

- **题目：** CMP-IRRT*: A Perception-Assisted Height-Adaptive Planner for Quadruped Robots
- **作者：** Mingfan Zhao, Wendong Mao, Zhongfeng Wang
- **单位：** arXiv 页面未列机构；周报标注为中山大学深圳校区（未在 arXiv 页面核实）
- **预印本：** https://arxiv.org/abs/2610.10470
- **HTML：** https://arxiv.org/html/2610.10470v1
- **版本：** v1，2026-10-07，cs.RO；已被 ACML 2026（第 18 届亚洲机器学习会议）接收
- **代码：** https://github.com/MingfanZhao/height-adaptive-planner （摘要声明公开；核查结论见 [sources/repos/height-adaptive-planner.md](../repos/height-adaptive-planner.md)）
- **项目页：** 无

## 论文要解决的问题

二维规划器通常把所有障碍都当作不可穿越的实心块，浪费了四足机器人能跨越低矮障碍的能力；采样规划（RRT*/IRRT*）在算力紧张时收敛慢。论文用俯视 RGB 估计障碍相对地面的高度，让规划器“高障碍绕行、低障碍可跨越”，并用学习的采样引导加速。

## 方法与训练信息

1. **感知：** 标定过的俯视 RGB；Depth Anything V2（针对目标场景微调）估计深度，经地面参考平面标定为地面相对高度图；YOLOv11 检测障碍区域；每个障碍在检测框中心取一个代表性顶面高度（假设顶面近似平坦的矩形障碍）。
2. **高度碰撞判定：** H_α = min{P_foot·max{cos θ}, H_round}，P_foot 为最大足端抬升，θ 为大腿关节角，H_round 为机器人最大可跨越高度；估计高度 ≥ H_α 的障碍视为阻断，低于则可穿越。原文未给 P_foot、H_round、H_α 的具体数值。
3. **CMP 引导网络：** PointNet++ 风格主干产生 128 维特征，经 6 个 Mamba 块，并与通道注意力加权的 PointNet++ 分支融合；输入为约 2048 个均匀采样的自由空间点（不足时用最远点下采样），二维图以 z=0 的三维点云表示。训练数据 1000 个随机世界训练 + 150 个测试，A* 生成参考路径，路径点 η 半径内为目标引导区（η、损失函数原文未给）。
4. **引导进 IRRT*：** 以概率 p_g 从引导集合采样，否则在有解时从信息椭圆采样、无解时从自由空间采样（p_g 未给）；当最优代价改善到一定因子 α 时在聚焦椭圆内重算引导；NeuralConnect 在引导不连通时建半径图并用 BFS 检查起终点连通性，最多重试 5 次。
5. **输出：** 规划结果包含速度、偏航率与“绕行(A)/穿越(B)”模式；规划在执行前完成，无在线重规划。

## 论文报告的结果

- **基准：** Center Block 与 2D Random World（沿用 NIRRT* 设置）；对比 RRT*、IRRT*、NIRRT*。
- **2D Random World（Table 1）：** 首解时间 / 节点数 / 迭代数 / 最终代价（18 s 后）：RRT* 0.17 s / 241.86 / 299.04 / 233.01；IRRT* 0.19 / 251.61 / 312.75 / 232.63；NIRRT* 1.81 / 102.92 / 151.69 / 206.51；CMP-IRRT* 1.37 / 99.49 / 146.55 / 203.41。注意 RRT*/IRRT* 首解更快，但节点和最终代价较差。
- **引导网络消融（Table 2，对 PointNet++ 归一化）：** +Mamba：节点 0.984、迭代 0.980；CMP（Mamba+通道注意力）：0.919、0.920，即约少 8.1% 节点、8.0% 迭代。
- **配对复评（Table 3，150 张图×3 随机种子=450 次）：** NIRRT* 首解 1.813±0.770 s、最终代价 207.27±26.79；CMP-IRRT* 1.380±0.650 s、204.11±25.73；成功率均 100%。
- **可穿越规划：** 4 张构造地图，1500 次迭代，10 次平均，每图至多两个可穿越障碍，启用高度条件穿越后路径缩短 7.7%–16.3%（论文归因于高度条件可穿越性，而非 CMP 引导本身）。
- **Go2 演示（Table 4，固定混合障碍场景）：** Binary-CMP 10/10 完成、8.56±0.11 m；Oracle 高度 CMP 10/10、7.25±0.09 m（短 15.3%）；预测高度 CMP 18/20 完成、7.71±0.13 m，2/20 需干预；无摔倒与机身碰撞。阈值测试：0.7H_α 10/10 穿越，0.9H_α 9/10（1 次干预）；高度判断在可穿越情形 18/20 正确，在应阻断情形 20/20 正确，错误均为保守误阻断。
- **训练与评测硬件：** RTX 4090，Python 3.9.16，PyTorch 1.13.1；Go2 的运动控制器与机载算力原文未指明。

## 复现与解释边界

- 需要标定的俯视观测且场景静态或缓变；不处理机载第一视角感知与动态重规划。
- 单目高度估计有噪声；高度只取检测框中心一点，对不规则/遮挡障碍不稳健。
- 统计上，CMP 引导相对 NIRRT* 的提升幅度较小（首解约快 0.43 s、代价差约 3），未做显著性检验。
- 官方仓库是“最小发布”，见 repos 归档：README 描述的包目录在仓库中不存在。

## Robotics_Notebooks 映射

- 详情节点：wiki/entities/paper-cmp-irrt-star-height-adaptive-planner.md
- 仓库归档：sources/repos/height-adaptive-planner.md
- 汇总页：wiki/overview/frontier-humanoid-quadruped-2026-10-05-10-09.md

## 一手来源

- arXiv 摘要：<https://arxiv.org/abs/2610.10470>
- arXiv HTML 全文：<https://arxiv.org/html/2610.10470v1>
- 官方仓库：<https://github.com/MingfanZhao/height-adaptive-planner>
