# Lodos（arXiv:2611.08872）

> 来源归档（paper）

- **标题：** Lodos: A Lightweight Simulator Designed for Contact-Rich Interactions with Detailed Meshes
- **类型：** paper
- **作者与机构：** Onur Beker、A. René Geist、Georg Martius（University of Tübingen；Georg Martius 另署 Max Planck Institute for Intelligent Systems）（论文首页）
- **arXiv：** <https://arxiv.org/abs/2611.08872>（v1，cs.RO，2026-11-06）
- **PDF：** <https://arxiv.org/pdf/2611.08872v1>
- **代码：** <https://github.com/martius-lab/lodos>
- **入库日期：** 2026-11-09
- **一句话说明：** 面向「精细非凸网格 + 大批量并行」接触仿真从零设计的 GPU 仿真器：球体–SDF 碰撞（三次样条 SDF、分层剔除）加「弹簧阻尼力 + 逐碰撞对缩放因子」的接触模型（BQP + 交错迭代），在所测网格场景下整步吞吐约为 MJWarp 的 6.29–21.7 倍。
- **来源线索：** 具身智能小站 2026-11-09 更新（扫读条目），其说法已逐条回论文核对，见下。

## 核对范围与通读

- **核对范围：** 从 arXiv v1 PDF 逐段通读全文：摘要、I 引言、II 相关工作、III 碰撞检测（样条 SDF、窄相、宽相）、IV 接触动力学（含摩擦修正与集总参数化有效性）、V 实验（Fig.1、4–9、Table I，表和图均看了渲染图）、VI 局限与未来工作、VII 结论、致谢与参考文献。论文 v1 无独立附录；补充材料（supplementary videos）未获得、未看，故视频中的定性现象只依据正文文字。作者代码仓库 README 静态读过首页（未克隆、未运行、未读 papers/ 下 PDF 与 docs/）。
- **通读：** 已完成

## 核心摘录（均来自论文，括号为章节 / 图表）

1. **问题设定（I）：** 常见仿真器以凸多面体碰撞（GJK-EPA）加正则化的 NCP/CCP 接触求解。作者列出三点不足：正则化使摩擦力低于维持静止所需的精确值、所有接触终将滑移；凸多面体只产生单点或稀疏角点接触，扭转摩擦只能靠人为约束模拟；凸多面体碰撞在大并行、非凸复杂网格下扩展性差。
2. **碰撞检测（III）：** 以球体–SDF 碰撞为公理交互；顶点视作半径 0 的球体；每个几何由「表面稠密零半径球 + 体内内切球 + 三次样条 SDF 及法向函数」表示；3D 样条任一查询点仅 64 项非零，求值与几何复杂度无关（III-A）。宽相：体素中心 SDF 做 O(1) 下界剔除（III-C1）、沿 3D Hilbert 曲线排序后分块成「瓦片」的两级层次（III-C2）、容量在线标定以避免主机–设备同步（III-C3）、OBB 预检整对剔除（III-C4）。刻意不做边–SDF 接触，作者承认这在理论上可能出现毫米级穿透，称其仅出现于病态顶点分布与边–边接触（III-B3）。
3. **接触动力学（IV）：** 对每个碰撞对的弹簧阻尼接触力乘一个非负缩放因子 α，使 MuJoCo/Drake 的 SOCP 变成盒约束 QP（BQP，式 4），用投影坐标下降、每个仿真实例单线程求解（IV-C1）；「交错迭代」多次重算弹簧阻尼力分布并累加（IV-C2）；摩擦修正第二个 BQP（式 5）把切向力推到库仑上界（IV-C3）。作者论证 R≠0 时集总参数化与原 SOCP 的差别只在「哪些接触点被耦合」（IV-C4）。
4. **物理忠实度实验（V-A）：** 夹方块实验中无摩擦修正最大摩擦力 8.22 N，有修正为精确 120 N、拾取前 114 ms 内滑移 5.97 mm，随后保持 5 s 平均蠕变速度 -0.0014 mm/s（V-A1）；25 块石拱：Lodos-MJ-0 / 1 不稳，Lodos-MJ-2 稳定（V-A2，Fig.5）；Microduck 双足策略在 MJWarp 训练、零样本放入 Lodos（V-A3，Fig.1）；四指棱柱抓取在 MuJoCo 中用 Dexonomy 优化后直接迁入（V-A4，Fig.6）。
5. **吞吐实验（V-B，Table I）：** 单卡 RTX 4080，2 ms 步长。20 只 armadillo 落地：Lodos-MJ-1 每只 451 顶点 + 100 球，MJWarp 每只 34 个凸多面体；Sharpa Wave 手内重定向：Lodos 手部 3,479 顶点、23 个刚体、无内切球，MJWarp 手部 87 个凸多面体。各自峰值批量下整步加速：手内重定向 9.19 / 7.49 / 6.29 倍（Ns = 0 / 1 / 2），armadillo 落地 21.7 / 14.8 / 10.3 倍；仅接触流水线分别为 14.0 / 11.0 / 9.90 倍与 24.8 / 16.7 / 11.1 倍。
6. **控制实验（V-C）：** iLQR + 零阶随机平滑（两个球夹起 1 m 方块并旋转 60 度）：Lodos 全程 0.75 mm / 0.13 度，Lodos 线性化 + MJWarp 滚动 10.5 mm / 1.24 度（V-C1，Fig.8）；MPPI（64 个并行 rollout、无域随机化，沿用 hydrax 参数）手内跟踪 armadillo 四个目标姿态，Lodos 控制器 + MuJoCo-CPU 被控对象可迁移（V-C2，Fig.9，5 个随机目标序列均值与标准差）。
7. **局限（VI）：** 批量小于约 256 时 MuJoCo-CPU、约 256–1024 时 MJWarp 的 rollout API 仍是最佳选择；几何可用球、胶囊、盒近似时 MJWarp 吞吐仍占优；当前不支持经 Nvidia Warp 的自动微分。

## 与具身智能小站说法的核对

- 「为复杂网格碰撞重做碰撞检测与接触求解」：与论文一致（III、IV 两部分均为新设计）。
- 「保持可用物理表现的同时提高并行吞吐」：论文自述「匹配成熟仿真器的物理忠实度」（摘要），但实际证据是少量定性 / 单例实验（夹方块、拱、Microduck、手抓取），无逐项与真实世界数据对比；吞吐优势限定在「详细网格–网格碰撞」。
- 「实验包含手内操作与控制迁移」：与 V-B2、V-C2 一致；注意迁移是仿真到仿真（Lodos 到 MuJoCo-CPU，MJWarp 到 Lodos），论文无真机实验。
- 「速度优势针对所测精细几何场景」：与 VI 一致，且 MJWarp 一侧用的是凸多面体近似，两边几何表示不同。
- 机构：以论文首页为准为蒂宾根大学与马克斯·普朗克智能系统研究所，小站文本未给出机构，无冲突。

## 代码核对

- 仓库 README（静态读，未运行）：定位为「研究代码库」，基于 NVIDIA Warp，沿用 MJWarp 的约束模型与非光滑 Newton 求解器；README 称当前代码相对 papers/lodos_v1.pdf 有重大扩展（见 papers/lodos_v2.pdf），并说明 bench 对比 MJWarp 与 Kamino，且描述了「原始 Signorini-Coulomb 条件的原始形式精确求解」。这与本文 v1 描述的 BQP + 投影坐标下降不完全相同，说明开源版本不等于本论文方法的逐字实现（推测：仓库为后续版本）。仅依 README 首页，未核实 papers/ 与 docs/ 内容。
- 致谢（论文）：软件开发由 Claude Opus 5 辅助。

## 开源状态

- **已开源**（<https://github.com/martius-lab/lodos>，README 可访问，2026-11-09）

## 对 wiki 的映射

- [paper-lodos](../../wiki/entities/paper-lodos.md)
