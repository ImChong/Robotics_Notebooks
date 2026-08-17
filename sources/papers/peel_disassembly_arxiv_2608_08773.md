# PEEL: Parallel Extraction for Long-Horizon Disassembly Planning via Scale-Invariant Sampling（arXiv:2608.08773）

> 来源归档（ingest）

- **标题：** PEEL: Parallel Extraction for Long-Horizon Disassembly Planning via Scale-Invariant Sampling
- **缩写 / 框架：** **PEEL** / **MAB-RRT**
- **类型：** paper / motion-planning / disassembly
- **arXiv：** <https://arxiv.org/abs/2608.08773>
- **项目页：** <https://peel-disassembly.surge.sh/>（归档见 [`sources/sites/peel-disassembly-surge.md`](../sites/peel-disassembly-surge.md)）
- **代码（双盲）：** <https://anonymous.4open.science/r/peel-disassembly-meta/>（归档见 [`sources/repos/peel-disassembly.md`](../repos/peel-disassembly.md)）
- **作者：** Servet B. Bayraktar、Andreas Orthey、Zachary Kingston、Marc Toussaint
- **机构：** 普渡大学（Purdue；Kingston）；其余作者单位未在 HTML 页头单列（Toussaint 一线为 TU Berlin 系，本库机构表无对应 alias，暂不注册）
- **入库日期：** 2026-08-17
- **一句话说明：** 尺度不变采样 + 多臂赌博机 RRT 求单件逃逸路径，再并行批次赛跑得到多件拆解顺序，Fetch 执行 10–17 件装配体。

## 开源状态（步骤 2.5）

- **项目页核查（2026-08-17）：** 有方法动画、四套多件装配体视频、76 件 100% 数字。Code 区链到 **anonymous.4open.science** 三仓（meta / MAB-RRT / robot-pipeline），不是 GitHub。项目页 bib 仍写 Anonymous / under double-blind review。
- **实现栈（论文）：** C++ 扩 OMPL；经 Robowflex + MoveIt + DARTSim。对照 [Assemble-Them-All](https://github.com/yunshengtian/Assemble-Them-All)。
- **结论：** **部分开源（双盲匿名仓）**；可按 meta-repo Docker 复现，稳定 GitHub 镜像待审稿后挂。源码运行时序图按匿名仓 README 入口绘制。

## 摘录 1：MAB-RRT

burn-in 自适应球半径估物体尺度；随后 bandit 在均匀采样与两条沿 PCA 逃逸方向的圆柱采样器间切换。单件 76 Automate 装配体 ×10 旋转 = 760 trial，成功率 **100%**，中位 3.2 s（均值 4.3 s）；次优 BFS 仅 53.9%。

## 摘录 2：并行批次协议

物体洗牌入队，每批 B 个并行规划赛跑：先找到无碰路径者写入顺序 \(\sigma\)，失败者回队尾；整批超时则整批重入队。避免指数级 precedence 搜索。四套多件（显微镜 12 / 碟刹 10 / 联轴 14 / 钳子 17）比基线快 2–7×。真机五阶段：侧抓 → 直线抽出 → 目标侧 IK → 桥接位换顶抓 → 跨装配体放置。

**对 wiki 的映射：** [`wiki/entities/paper-peel-disassembly.md`](../../wiki/entities/paper-peel-disassembly.md)；交叉 [MPC](../../wiki/methods/model-predictive-control.md)。

## 当前提炼状态

- [x] 论文摘要填写
- [x] wiki 页面映射确认
- [x] 开源状态核查（anonymous.4open.science）
