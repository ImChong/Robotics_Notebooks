# gmapping_tro_2007_icra_2005

> 来源归档（paper）

- **标题：** Improved Techniques for Grid Mapping with Rao-Blackwellized Particle Filters（IEEE Transactions on Robotics，2007）；前作 Improving Grid-based SLAM with Rao-Blackwellized Particle Filters by Adaptive Proposals and Selective Resampling（IEEE ICRA，2005）
- **类型：** paper（无 arXiv 版本；以作者主页预印本为准）
- **作者 / 机构：** Giorgio Grisetti（弗莱堡大学；罗马第一大学 La Sapienza）、Cyrill Stachniss（弗莱堡大学；T-RO 版署名另有苏黎世联邦理工 ETH）、Wolfram Burgard（弗莱堡大学）（以两份预印本首页署名为准）
- **原始链接：**
  - T-RO 预印本：<http://www2.informatik.uni-freiburg.de/~stachnis/pdf/grisetti07tro.pdf>
  - ICRA 预印本：<http://www2.informatik.uni-freiburg.de/~stachnis/pdf/grisetti05icra.pdf>
  - T-RO 官方记录：DOI 10.1109/TRO.2006.889486（卷期页码取自通行引用，预印本未印出，未核实）
- **代码：** <https://github.com/ros-perception/openslam_gmapping>（OpenSLAM gmapping 的 ROS 社区镜像；README 自述「fork from gmapping at openslam.informatik.uni-freiburg.de … includes a few patches」）；ROS 封装见 <https://github.com/ros-perception/slam_gmapping>。仓库源码本次未读，未核实其实现是否与论文逐项一致。
- **入库日期：** 2026-11-11
- **版本：** 作者主页预印本（T-RO 版 12 页）；非出版社排版版
- **核对范围：** 用 pdftotext 抽取并通读 T-RO 预印本全文 §I–§VIII（引言、RBPF 总述、改进提议分布与自适应重采样、算法 1、实现细节、复杂度与表 I、实验 §VI-A 至 §VI-F 与表 II、表 III、相关工作、结论），参考文献仅浏览；通读 ICRA 2005 预印本全文（含其实验 §V 与图 6、图 7 文字描述）。图只读图注与正文对图的描述，未逐图目视，PDF 图像未单独核对。未读：ICRA 版与 T-RO 版之外的补充材料（预印本无附录）、仓库源码、ROS 封装 slam_gmapping 的参数与实现。未运行任何代码。
- **通读：** 已完成
- **一句话说明：** 在栅格地图 RBPF-SLAM 中，用「扫描匹配 + 里程计」构造每粒子的高斯提议分布，并用有效粒子数 Neff 触发选择性重采样，把达到拓扑正确地图所需粒子数降到先前方法的约十分之一（论文 Table II）。

## 核心摘录

### 1) 问题（T-RO §I、§II）
- RBPF 把后验分解为 p(m | x1:t, z1:t) p(x1:t | z1:t, u1:t-1)（式 1）：轨迹用粒子滤波，每个粒子带一张自己的栅格地图，地图由「已知位姿建图」解析得到（§II）。
- 作者指出的两个难点：所需粒子数（复杂度）与重采样可能淘汰正确粒子（粒子耗散 / particle depletion）（§I）。
- 论文把 RBPF 用于 SLAM 归功于 Murphy 与 Doucet 等（参考文献 [6, 32]），Montemerlo 等把它扩展到地标地图（FastSLAM，[28, 29]）；栅格地图的 RBPF 前人工作是 Eliazar & Parr（DP-SLAM）与 Hähnel 等（§VII）。

### 2) 改进提议分布（T-RO §III-A 至 §III-C，式 9 至式 19；ICRA §IV-A）
- 以往用里程计运动模型作提议分布，激光很准时似然函数窄、运动模型宽，大量粒子落在低似然区域（§III-A，图 1）。
- 本文做法：对每个粒子，先用扫描匹配（实现中是 CARMEN 的 vasco，梯度下降，§IV）在初始猜测附近找似然局部最大，在其周围采 K 个点，按观测似然乘运动模型加权，算出高斯 N(μ, Σ)（式 15 至 17），再从该高斯采样新位姿，权重乘以归一化因子 η（式 19）。
- 与 ICRA 版区别：ICRA 版在似然显著区间 L 内把运动模型近似为常数，只靠激光；T-RO 版同时保留里程计项，在长直走廊、空旷场地这类特征贫乏处更稳（§VI-D：模拟 4 m 量程激光，计入里程计 100% 成功，不计入 50%）。
- 与 FastSLAM-2 的关系：论文自述计算方式与 FastSLAM-2 类似，但 FastSLAM-2 依赖预定义地标，本文直接用原始激光数据与栅格地图（§VII）。

### 3) 自适应（选择性）重采样（T-RO §III-D，式 20；ICRA §IV-B）
- 用 Neff = 1 / Σ(w̃_i)^2 衡量粒子集对后验的近似好坏，Neff < N/2 才重采样；其余时刻不重采样（§III-D）。触发后做的是普通的按权重有放回抽样（§II 步骤 3），论文没有描述按权重排序的步骤。
- 论文对 Neff 行为的观察（§VI-C，图 8）：走未知区域时缓降；走已知区域时基本恒定；闭合回环时部分粒子对得上自己的地图、部分对不上，Neff 骤降从而触发重采样。

### 4) 数字与实验（T-RO §VI）
- 更新频率：机器人移动超过 0.5 m 或旋转超过 25 度做一次滤波更新（§III-C、§VI-F）。
- 数据集与粒子数（§VI-A）：Intel Research Lab（28 m × 28 m）15 个粒子；Freiburg 校园（室外，约 250 m × 250 m，轨迹 1.75 km，图 5）30 个粒子；MIT Killian Court（250 m × 215 m，行驶 1.9 km，多个嵌套回环，图 6）60 个粒子可得拓扑正确但有时出现「双层墙」，80 个粒子得高质量图。
- Table II：达到至少 60% 次运行拓扑正确所需粒子数，本文 Intel 8、MIT 60、Freiburg 20；Hähnel 等的方法 40、400、400；论文概括为「约少一个数量级」（§VI-B）。
- 成功率（图 7）：每个设定 20 次运行，由非作者的检查者盲评拓扑正确性；MIT-2 曲线关闭自适应重采样，成功率低于开启时（§VI-C）。
- 复杂度（§V，表 I）：计算提议、更新栅格地图、算权重、判断是否重采样均为 O(N)；重采样要复制整张地图，最坏 O(N·M)（M 为地图大小）；论文指出可借 DP-SLAM 的地图表示降低，但「不是本文目标」。
- 运行时（§VI-F）：2.8 GHz 标准 PC，30 个粒子；Intel 数据集 45 min 的日志用不到 30 min 处理完，内存约 150 MB（地图约 40 m × 40 m，分辨率 5 cm）；表 III 平均耗时：提议 + 权重 + 地图更新 1910 ms，判断是否重采样 41 ms，重采样 244 ms。
- ICRA 版（§V）：在 Pioneer2 AT + Sick PLS 上在线运行；称不超过 100 个粒子即可处理 250 m × 250 m 环境，Pentium IV 2.8 GHz、0.5 m/s 下 50 个粒子可实时。

### 5) 作者自述的局限与失败情形（T-RO）
- §III-C：扫描匹配只返回距初始猜测最近的单个极大值，多峰时可能漏掉其他峰；作者靠频繁更新与限制搜索区域，认为可视为单峰；极端杂乱且里程计噪声很大时滤波「至少理论上」可能过度自信；作者说实验中从未遇到，并给出方案（跟踪多个模式）但未实现。
- §III-C、§VI-E：扫描匹配失败（观测差、与已有地图重叠太小）时退回原始运动模型；Intel 等室内数据未出现，MIT 数据中因行人挡在玻璃走廊里出现过一次；Freiburg 校园中机器人进入空停车场（几乎全是最大量程读数）时也退回里程计。
- §VI-A b)：校园环境「部分违反环境为平面的假设」，且有灌木、草地、行人与车辆等动态物体；论文称仍得到准确地图。
- §VI-A c)：MIT 数据集用 60 个粒子时有时出现人工「双层墙」，需 80 个粒子。
- §V：每个粒子各存一张栅格地图，重采样复制开销 O(N·M)。

## 对 wiki 的映射

- [paper-gmapping](../../wiki/entities/paper-gmapping.md)

## 未核实

- 仓库源码是否与论文一致、ROS 封装 slam_gmapping 的默认参数：未读。
- 论文之外对 GMapping 在更长时间、更大场景下漂移表现的独立评测：未检索。
