# It Takes Two to Tangram（TangramBench / GOMA，arXiv:2611.08814）

> 来源归档（paper）

- **标题：** It Takes Two to Tangram: Multi-Agent Object Centric Reinforcement Learning for Assembly
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2611.08814>
- **PDF：** <https://arxiv.org/pdf/2611.08814v1>（v1，2026-11-06 提交，cs.RO，标注 Preprint；已投 IEEE，见首页脚注）
- **项目页：** <https://tangrambench.org/>
- **代码：** 论文脚注 1 写明 camera-ready 版将链接含生成器、资产与基准划分的仓库；项目页 Code 按钮在抓取到的 HTML 中仍是占位（`code: ""`），**代码未公开 / 未核实可用**
- **入库日期：** 2026-11-09
- **机构（论文首页）：** 达姆施塔特工业大学 Intelligent Autonomous Systems Lab 与 Control and Cyber-Physical Systems Laboratory；Hessian.AI；DFKI（SAIROL）；Robotics Institute Germany。作者：Paul Knauß, Kay Pompetzki, Nico Bohlinger, Niklas Funk, Cristiana de Farias, Rolf Findeisen, Jan Peters
- **核对范围：** 下载 v1 PDF（首页标题与编号 2611.08814 已核对），用 pdftotext 逐段通读正文 I–VI 节（引言、相关工作、TangramBench、GOMA 方法、实验含 sim2real、结论）、Table I、Table II（附录超参）与参考文献列表；图 1–5 只读到图注与文字描述，曲线图的具体数值未逐点读取（正文给出的数字才引用）；论文无其他补充材料；项目页只读 HTML 静态文本（未看视频）。
- **通读：** 已完成

## 核心摘录（均出自论文）

1. **问题（III 节）：** N 块刚性平面零件，位形 q_i=(x,y,θ)∈SE(2)；给定初始位形与已指派的目标位形，要求不穿插地把全部零件移到目标。成功判据：中心距目标 ≤1 cm 且朝向误差 <0.1 rad。目标位姿放在 70×36 cm 平面上，初始位姿在 60×34 cm 区域内采样，零件中心间距至少 18 cm。
2. **基准三档（III 节，Fig. 2）：** Trio（3 块：两大一中三角形）、Cluster（7 块，紧凑但较开放）、Puzzle（7 块，边对边拼接、封闭槽位）。两者中位零件间距同为 37 mm，但中位「围合度」Cluster 9%、Puzzle 57%。训练配置约 1M / 1.1M / 955k，留出评测配置各 258 个。
3. **评测实现：** MuJoCo、网格碰撞；零件被直接以体坐标系平移 / 角速度驱动（无机器人运动学）。指标含成功率、位置与偏航误差、零件间接触率、完成时间、计算时间、5 cm 内零件比例。
4. **GOMA（IV 节，Fig. 3）：** 策略与评论家为两个独立的图网络（边 / 节点 / 全局三类更新，残差 MLP）。策略头在零件间共享，输出 tanh 缩放后的均值与共享的可学习标准差（对角高斯）；评论家输出整个场景一个标量。训练为 MAPPO 的逐智能体裁剪目标：每块零件有自己的 GAE 优势，评论家回归各零件回报目标的均值。
5. **奖励（Eq. 5–6）：** 个体项含朝目标的速度投影进度、朝向误差 1−cos、到位稀疏奖励、零件间接触惩罚、工作区边界惩罚；共享项为最大位置误差惩罚与全部到位的完成奖励。接触不终止回合。
6. **实验设置（V-A）：** MuJoCo，GPU 上 1024 个并行实例；物理步长 10 ms，策略 10 Hz。
7. **结构消融（V-B，Trio，5 个种子）：** 单智能体 PPO（MLP 拼接全部零件）几乎无成功；MLP-MAPPO 约 90% 零件进入 5 cm，但成功率约 50% 即平台期；GOMA 与 GOMA-J（图 + 单一联合团队奖励）都收敛到约 90% 成功。作者据此认为关系建模（GNN）是关键，而逐智能体优势作为归纳偏置的增益有限。
8. **奖励消融（V-C，Cluster，5 个种子）：** 去掉接触惩罚则接触频繁；去掉团队奖励对避碰与粗到位影响小，但最终装配成功率更低。Fig. 5 用 7 块训练的策略评测 3/5/6/7 块，比较团队奖励随件数增多的作用（正文未给具体数字）。
9. **与经典规划对比（V-D，Table I）：** 对手为吸引式 APF、联合 RRT-Connect（RRT-J，21 维位形空间）、逐件 RRT（RRT-P），后两者各有一次规划与每 5 s 重规划两种。规划器只给「起始布局通过其碰撞检查」的 258 个中的 117 个，且回合 150 s。
   - Cluster：GOMA 成功 75%、误差 1.0 cm、接触 4%、中位时间 8 s、推理 0.68 ms；RRT-J+replan 98%、0.3 cm、接触 12%、21 s、49 ms；RRT-J 73%；APF 3%；RRT-P 3%，+replan 9%。
   - Puzzle：GOMA 41%、1.7 cm、接触 4%、24 s、0.68 ms；RRT-J+replan 88%、1.5 cm、接触 37%、43 s、1.1 s；RRT-J 58%（6.8 s 计算）；APF、RRT-P 及其重规划 0%。
10. **真机（V-E）：** 仅定性零样本演示：Franka FR3 + CRISP 笛卡尔阻抗控制器 + 定制方形插销工具；RealSense D435i（场景）与 D405（腕部）、FoundationPose + HSV 分割估位姿。单机械臂一次只动一块，故对未完成零件轮询（round-robin）顺序执行策略输出，每次查询前由感知重建交互图。**没有报告真机成功率或试验次数。**
11. **结论自述的范围（VI 节）：** 目标位姿与零件 - 目标指派均为给定，不解决由轮廓反推 Tangram 分解；真机顺序执行，未直接验证并发协调；抽象掉了推挤等非抓取操作。
12. **超参（Table II）：** 隐藏维度 256，消息传递 1 轮，γ=0.99，λ=0.9，裁剪 0.1，每次更新 3 个 epoch，批量 4096，滚动长度 100，并行环境 1024，策略学习率 2×10^-5→10^-6。

## 致谢 / 声明

- 论文声明 Claude（Anthropic）被用于语法润色、实验编码辅助与 Fig. 1 的展示图生成，输出经作者审核。

## 对 wiki 的映射

- [paper-tangrambench-goma](../../wiki/entities/paper-tangrambench-goma.md)
