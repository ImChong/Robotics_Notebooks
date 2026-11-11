# Motion Graphs

- **类型：** 作者一手论文
- **作者：** Lucas Kovar, Michael Gleicher, Frédéric Pighin
- **会议：** SIGGRAPH 2002
- **机构：** 威斯康星大学麦迪逊分校（University of Wisconsin–Madison）；南加州大学（University of Southern California）
- **论文：** https://graphics.cs.wisc.edu/Papers/2002/KGP02/
- **原始版本：** SIGGRAPH 2002 论文作者存档
- **项目页：** https://research.cs.wisc.edu/graphics/Gallery/kovar.vol/MoGraphs/
- **代码：** 作者主页未列可核实官方代码
- **核对范围：** 原始作者论文完整主体：§1 背景、§2 相关工作、§3 逐帧点云相似度/过渡片段创建/有向图修剪、§4 分支定界图搜索、§5 路径拟合及多动作切换、§6 复杂度与局限；全文图1–8、主要方程、数值例子；参考文献表浏览。并阅读作者实验演示说明，论文无独立补充附录。
- **通读：** 已完成

## 方法与应用

基于动捕片段建立有向连接和可平滑的过渡边；把用户的行进轨迹、风格要求转成图搜索目标。交互方向控制可预计算节点与方向对应的后续片段。

## 实验与结论

原作者演示覆盖沿任意路径行走、混合动作风格与交互改变方向；示例中角色响应转向可能有延迟，不是即时重规划的动力学机器人。

## 局限 / 对 wiki 的映射

不能直接证明重定向到 G1 后的物理稳定性；路径跟踪受动捕库覆盖程度与转场限制。

- [Motion Graphs：动捕片段图与路径控制](../../wiki/entities/paper-motion-graphs.md)

## 原文复核摘录（正文证据）
- **构图步骤：** 时间窗点云距离做平面刚体对齐，取误差局部极小候选，阈值选转场，用 root 插值与关节四元数 slerp 拼接；保留描述标签和足部接触约束；再按标注子图求可用的强连通部分修剪死路。
- **搜索：** 用路径拟合的弧长对齐误差分段进行分支定界；论文使用 80–120 帧搜索视野、每次保留 25–30 帧，不是网络预测未来动作。
- **速度与限制：** Fig.7 走路动作 54.9 秒需要 58.1 秒计算，武术 87.7 秒动作需 15.0 秒；互动方向表通过**预计算**索引，作者明确指出改向存在延迟；构图候选检测 O(F²)。
- **参考：** §3.1–§3.4、§4.2、§5.1–§5.2、§6；[原作者论文 PDF](https://research.cs.wisc.edu/graphics/Papers/Gleicher/Mocap/mograph.pdf)；[原作者演示](https://research.cs.wisc.edu/graphics/Gallery/kovar.vol/MoGraphs/)。
