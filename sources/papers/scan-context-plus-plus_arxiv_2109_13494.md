# Scan Context++（arXiv:2109.13494）

> 来源归档（paper）

- **标题：** Scan Context++: Structural Place Recognition Robust to Rotation and Lateral Variations in Urban Environments
- **作者：** Giseop Kim（KAIST）、Sunwook Choi（NAVER LABS）、Ayoung Kim（首尔国立大学）（首页署名）
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2109.13494>
- **PDF：** <https://arxiv.org/pdf/2109.13494v1>（v1，2021-09-28；页眉为 IEEE Transactions on Robotics 稿件格式，卷期为占位 "VOL. X"）
- **代码：** https://github.com/gisbi-kim/scancontext （论文首页脚注给出；仓库 README 仍以 IROS 2018 版 Scan Context 为主，未见 "++" 字样）
- **入库日期：** 2026-11-11
- **一句话说明：** 把 Scan Context 扩成"极坐标（抗旋转）+ 笛卡尔（抗横向平移）"两种描述子，加检索键 / 对齐键两个一维子描述子与描述子增强，提供拓扑检索加 1-DOF 半度量初值。

## 通读状态

- **核对范围：** arXiv v1 PDF 正文逐段读完（摘要、§I–§IX 全部小节，含 Table I–V 及 §VIII 各讨论小节）；Table VI–IX 与各 PR / DR 曲线只核对了文字结论与可抽取的数字，未逐格复核图中曲线，也未查看图片本体（只读图注与正文描述）；参考文献仅浏览列表，未读被引论文；论文提到的附件视频（multimedia file）与 GitHub 仓库源码未读，仓库只静态读了 README；v1 之后若有期刊终稿差异，未核对。
- **通读：** 已完成

## 核心摘录

1. **问题与定位（§I）：** 作者指出前作 Scan Context（IROS 2018）"未能实现横向不变性，且用暴力搜索效率低"；新增贡献为横向 / 旋转鲁棒、半度量定位、轻量模块独立（单个 C++ 头文件加源文件）、CPU 实时、DR 曲线评测。
2. **两种描述子（§IV-B / §IV-D）：** Polar Context（PC）以方位角为对齐轴 A-axis、半径为检索轴 R-axis，旋转对应列平移；Cart Context（CC）以横向 y 为 A-axis、纵向 x 为 R-axis，横向平移对应列平移。分箱编码取箱内最大高度（Requirement 1）。
3. **子描述子（§IV-F）：** 检索键为每行的 L1 范数（需置换不变，前作用 L0 范数），对齐键为每列的 L1 范数。
4. **三阶段识别（§V）：** k-d 树按检索键检索（实验 k=1）、对齐键 L2 距离求列平移 n*（§V-B 式 4）、全描述子在 n* 邻域内算距离并与阈值 τ 比较（式 5–6）。
5. **增强（§V-D）：** A-PC 在横向 ±2 m 处生成两个虚拟根位姿；A-CC 为两轴翻转（double flip）；增强描述子共用同一个 k-d 树。
6. **复杂度（§V-E）：** 朴素比较 O(NA·NR·NA)，预对齐后 O(NA·NR·1)；近似 NA~NR~N 即 O(N^3) 降到 O(N^2)。
7. **评测设置（§VI）：** 4 个数据集 8 条序列（KITTI 00/08、MulRan KAIST 03 / Riverside 02 / Sejong、Oxford Radar RobotCar 两条、NAVER LABS Pangyo）；判对阈值 8 m；对比 M2DP、SegMatch（用真值里程计），另与 PointNetVLAD、SegMap 比较（§VIII-E）。
8. **结果（§VII）：** KITTI 08（全是反向重访）上 PC 明显强于 CC 与 M2DP；Riverside 02（多次换道同向）上 CC 大幅领先；Oxford / Pangyo 复合变化下 A-CC 总体最好，A-PC 在纯旋转变化最好。
9. **半度量定位（§VII-E）：** Pangyo 上对齐键给出的 1D 初值平均误差 A-PC 1.03°、A-CC 0.84 m。
10. **耗时：** Fig. 18 平均 7.36 ms（KITTI 00）/ 7.31 ms（Pangyo），Intel i9-9900，含每 10 s 一次 k-d 树重建；Table V（MATLAB）总计 Ours PC 8.3 ms、A-PC 11.5 ms、M2DP 5.8 ms、SegMatch 796.0 ms、PointNetVLAD 34.0 ms；全描述子暴力检索 4500 帧（KITTI 00）"almost 1 second"（§VIII-B）。
11. **SLAM 集成（Table IV）：** LeGO-LOAM 与 SC-LeGO-LOAM 的 ATE（均值 / 最大）：KAIST 03 平移 20.7/42.7 m 对 3.4/8.8 m；Riverside 02 平移 47.7/130.5 m 对 15.2/50.5 m。
12. **局限（§VIII-D / F / H）：** 原生只在一个方向不变；横滚俯仰扰动（±5° 到 ±15° 仿真）下所有方法都退化；走廊类混淆与近处高大遮挡物会失败；不同安装高度或传感器的建图 / 定位互换不在范围内。

## 对 wiki 的映射

- [paper-scan-context-plus-plus](../../wiki/entities/paper-scan-context-plus-plus.md)
