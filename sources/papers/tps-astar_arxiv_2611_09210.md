# TPS-A*：任意形状车辆的轨迹参数空间认证碰撞检查路径规划（arXiv:2611.09210）

> 来源归档（paper）

- **标题：** TPS-A*: Path Planning for Any-Shape Vehicles with Certified Collision Checking in Trajectory Parameter Space
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2611.09210>
- **PDF：** <https://arxiv.org/pdf/2611.09210v1>（v1，2026-11-06 提交，cs.RO，共 12 页）
- **代码：** https://github.com/MRPT/mrpt_path_planning （论文首页脚注；ROS 2 封装在 https://github.com/mrpt-ros-pkg/mrpt_navigation ；论文称完整基准套件「发表后释放」）
- **作者与机构（论文首页）：** Jose Luis Blanco-Claraco、José Luis Torres-Moreno（阿尔梅里亚大学工程系，CIESOL、CeiA3）；Fernando Cañadas-Aránega、José Carlos Moreno-Úbeda（阿尔梅里亚大学信息学系）；Shravan S. Rai（Teladoc Health，美国）
- **入库日期：** 2026-11-10
- **一句话说明：** 在 SE(2) 格点 A* 中，把每个节点的碰撞检查换成「一次遍历局部障碍点即得所有候选轨迹的经证明保守的可行距离」，成本与脚印顶点数和分支数无关；路径经独立精确检验无碰撞。
- **核对范围：** 通读 arXiv v1 PDF 全部 12 页正文（摘要、第 I-IX 节、定理 1/2 与推论 1、命题 1/2 及其证明、算法 1/2、表 I-V、参考文献表）；该论文无附录或补充材料。图 1-6 仅读了图注与从 PDF 抽出的坐标数值标签，未逐图目视核对；官方仓库只读了 README 首屏（功能列表），未读源码、未运行；论文声称的 BARN 完整基准套件尚未发布，实验数字未能复现。
- **通读：** 已完成

## 开源状态

- **已开源：** 规划器库 `MRPT/mrpt_path_planning`（README 描述 PTG、任意脚印、TPS_Astar 与双向变体、认证碰撞网格、Reeds-Shepp 启发式与解析扩展）；README 与论文一致。论文称完整基准套件待发表后释放（未核实是否已放出）。

## 核心摘录

1. **方法（第 IV 节）：** 离线按（脚印、PTG、净空 m）为每个网格单元 c 与每条轨迹 k 存下 G[c][k]，即该单元内任意点最早可能被脚印「碰到」的轨迹距离（式 2，用相邻采样间的位移上界 r_{k,n}=ℓ+R_B·|Δφ| 保证连续扫掠保守，式 1）；在线对局部障碍点一次遍历取最小，得到所有 k 的自由距离 d_free(k)（式 3，算法 1）。
2. **理论（第 IV-C、VI 节）：** 定理 1（不会高估真实首次接触距离）、推论 1（用边界采样点认证区域障碍）、命题 1（单节点成本 O(N·L̄)，与顶点数 V 和后继数 B 无关）、定理 2（返回路径为 PTG 段拼接且脚印加净空与障碍不相交）、命题 2（C ≤ ε(C*_G + δ)，相对其格点抽象而非原连续问题）。
3. **算法（第 V 节）：** 每个 (x,y,θ) 单元保留一个连续状态代表；加权 A*（实验 ε=1.5）；位置目标用欧氏启发式，位姿目标用 Reeds-Shepp 长度启发式；延迟提交的解析目标扩展，加上由 PTG 极端轨迹组成并逐段认证的 Reeds-Shepp 射击。
4. **BARN 300 世界（第 VII-C 节，表 III）：** 共享目标位姿下 23 个世界的目标脚印本就与占据栅格重叠，最多 277 个查询（92.3%）有无碰撞解；TPS-A* 解出全部 277 个且全部通过独立精确检验（2.09 M 个位姿无重叠），规划时间中位 2.9 ms（均值 24 ms，最大 0.66 s）；SBPL ARA* 中位 169 ms（约 58 倍）；Nav2 Hybrid-A*（Jazzy）对 300 个全部给路径，但 136 条（45%）在返回位姿处与占据格重叠，仅 54.7% 通过检验，时间中位 5.4 ms（正文写 5.1 ms，表为 5.4）。位置目标：TPS-A* 中位 1.9 ms 解 300/300 全部验证；Nav2（Kilted）中位 170 ms，57.3% 验证。
5. **窄门（第 VII-D 节，图 3）：** HouseExpo 40 张图，0.95×0.55 m 差速车，门瓶颈 0.70-1.10 m（中位 0.80 m）。Nav2 外接圆半径只解 20/40，内切圆解 40/40 但 4 条与真实脚印扫掠碰撞；用同一精确多边形加膨胀层的 Nav2 也解 40/40，TPS-A* 更快（中位 1.9 vs 4.8 ms）。作者明说其价值「不是相对配置得当基线的成功率优势」，而是去掉圆形模型的取舍。
6. **泊车、叉车、实机（第 VII-E/F/G 节）：** TPCAP 20 例中 TPS-A* 与 Nav2 各解 19，时间 0.76 vs 1.61 s 无统计差异（p=0.92）；叉形非凸脚印 20 个场景两者全解，Nav2 更快（中位 0.10 vs 0.26 s），Nav2 有 7 条因平滑器与障碍重叠；Husky 校园 15 个目标到达 8 个（210 m，中位 0.49 m/s）。

## 对 wiki 的映射

- [paper-tps-astar](../../wiki/entities/paper-tps-astar.md)
- 相关：[A* 方法页](../../wiki/methods/a-star.md)、[Nav2](../../wiki/entities/navigation2.md)
