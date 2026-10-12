# Same Action, Different Outcome 论文来源归档

## 元数据

- **题目：** Same Action, Different Outcome: Variability in Dynamic Cloth Manipulation
- **作者：** Mahed Dadgostar, Guillem Alenyà, Júlia Borràs
- **单位（据 HTML 版）：** Institut de Robòtica i Informàtica Industrial（IRI），CSIC-UPC，西班牙
- **预印本：** https://arxiv.org/abs/2610.10801
- **HTML：** https://arxiv.org/html/2610.10801v1
- **版本：** v1，2026-10-07，cs.RO
- **项目页：** https://cloth-var.github.io（归档见 [cloth-var-project.md](../sites/cloth-var-project.md)）
- **代码：** 无分析代码仓库；数据集部分发布（仅 1 个样例回合，完整集"将以持久标识符发布"，尚未提供）
- **核查日期：** 2026-10-12

## 论文要解决的问题

布料在重复快速动作下，即使轨迹相同，终态也会不同；这种变化此前没有被系统量化。它直接影响模仿学习策略的可靠性，也决定仿真能否替代真实数据。

## 方法与实验设置

- **四个任务（共 27 个配置、269 个回合，每配置重复 10 次，部分配置 9 次）**：swing place（双臂握短边挥出后放平）、dynamic fold（单臂把角点甩向目标）、linear inward join（两夹爪直线相向，观察上边屈曲方向，新任务）、swing inward join（先挥后相向，新任务）。
- **三种布（均 0.7×0.5 m，19 个标记点）**：棕色羊毛 139 g（400 g/m²，摩擦 0.68，悬垂刚度 9.5%）；绿色缎 35 g（100 g/m²，0.48，5.9%）；红色雪纺 27 g（76 g/m²，0.40，3.1%）。
- **速度**：时间尺度因子 ts（>1 更慢）：任务 1 取 0.8/1/1.2（棕）与 0.65/1/1.35（绿、红）；任务 2 取 1/1.4；任务 3 仅 1；任务 4 取 1/1.5。另有重抓配置。
- **平台**：两台 Franka Research 3 并排；OptiTrack 100 Hz 追踪布标记点与末端标记；ZED 2i 双目 30 Hz；四个 ArUco 标记做 OptiTrack 与相机的标定。轨迹为遥操作演示（任务 1、2、4）与线性关节空间脚本（任务 3）。
- **变化度量**：终态标记点的成对距离 Dc(i,j)；配置离散度 D̄c 为所有回合对的均值；地板变化 εc 取被夹角点标记（反映机器人、抓取、传感噪声）；D̄c>εc 则视为不可重复；速度对比 ΓD = 最快/最慢速度的 D̄c 比。
- **四个仿真器**：Clothilde（不可伸长模型）、Isaac Lab（PhysX 布料）、Genesis（PBD 布料）、Newton（VBD 布料），均开启自碰撞；起始形状由 19 个标记点拟合到 21×29 网格；每种布一组共用材料参数，9 个训练回合做粗网格搜索再逐参数精修。
- **无标记视觉（第 VI 节）**：SAM 2 分割 + FoundationStereo 3D + DINOv3 补丁匹配当虚拟标记。

## 论文报告的结果

- D̄c 范围从 1.2 mm（棕布，慢速挥动相向）到 118 mm（绿布，折叠），相差约 96 倍。
- 所有无重抓配置至少高出地板 2.4 倍，多数高出一到三个数量级（摘要原话为 1–3 个数量级）；重抓会把地板抬到 12–27 mm（原为 2–4 mm），但结果仍高 2.6–4.2 倍，且重抓不显著改变变化量。
- log10 D̄c 的方差解释份额：布料 61%，任务 10%，速度 6%。棕布大多最可重复，两种轻布离散高 3–10 倍。
- ΓD：仅绿布折叠低于 1（0.97）；棕布挥动相向达 21（慢 1.2 mm、快 25.9 mm）。
- swing place：棕 5–9 mm，绿 23–47 mm，红 33–81 mm，地板 ≤2.1 mm，高出 14–1250 倍；dynamic fold：自由角落点距目标 7–41 cm，棕离散 7–12 mm，轻布 55–118 mm。
- 仿真：四个引擎在保留回合上的标记误差：Clothilde 37–48 mm，Isaac Lab 70–78 mm，Genesis 82–90 mm，Newton 79–94 mm；均无法再现变化幅度与排序；Clothilde 与 Genesis 最贴近对角线；Isaac Lab 与 Newton 非确定性且噪声与真实变化不相关。
- 无标记视觉能保持总体趋势顺序，但数值有高估低估，折叠最差；适合定性比较，不适合精确度量。

## 复现与解释边界

- 只有三种同尺寸布、四个任务、有限速度与重抓条件；轨迹为开环，只测量离散度，不测任务成功率。
- 标记标签需人工监督，存在遮挡；地板估计依赖被夹角点标记。
- 仿真标定误差 37–94 mm 本身较大；棕布任务 4 仅 9 个回合。
- 数据：项目页给出 1 个样例回合（含动捕、机器人日志、元数据、仿真孪生和标定参数），CC BY 4.0；完整 269 回合尚未提供。

## Robotics_Notebooks 映射

- 详情节点：wiki/entities/paper-same-action-different-outcome-cloth.md
- 汇总页：wiki/overview/frontier-manipulation-2026-10-05-10-09.md
- 项目页归档：sources/sites/cloth-var-project.md

## 一手来源

- arXiv 摘要：<https://arxiv.org/abs/2610.10801>
- arXiv HTML：<https://arxiv.org/html/2610.10801v1>
- 项目页：<https://cloth-var.github.io>
