# Learning Traversability-Aware Global Planners for Long Horizon Off-Road Navigation（arXiv:2607.23743）

> 来源归档（ingest）

- **标题：** Learning Traversability-Aware Global Planners for Long Horizon Off-Road Navigation
- **缩写 / 系统：** LRP（Long-Range Planner）+ 连续 traversability / path-logits 双头
- **类型：** paper / off-road-navigation / traversability / global-planning / geospatial-dataset
- **arXiv：** <https://arxiv.org/abs/2607.23743>（HTML：<https://arxiv.org/html/2607.23743v1>）
- **DOI：** <https://doi.org/10.48550/arXiv.2607.23743>
- **数据集：** <https://huggingface.co/datasets/anony-008/offroad-global-nav> — 归档见 [`sources/datasets/offroad-global-nav.md`](../datasets/offroad-global-nav.md)
- **项目页：** 无独立项目站（arXiv HTML 仅链 HF 数据集）
- **代码：** 截至入库日 **未发布** 训练 / 推理仓；前作 Trailblazer 代码见 <https://github.com/unmannedlab/Trailblazer>（不可当作本文实现）
- **作者：** Kasi Viswanath、Jason M. Gregory、Shaunak Kolhe、Srikanth Saripalli
- **机构：** 德州农工大学（Texas A&M University）机械系 Unmanned Systems Lab；DEVCOM 陆军研究实验室（Army Research Laboratory）
- **资助：** DARPA + ARL Cooperative Agreement W911NF-21-2-0064
- **入库日期：** 2026-09-21
- **一句话说明：** 从卫星影像 + 航拍 LiDAR + OSM 学习连续可通行图与目标条件路径偏好，用人类 GPS 轨迹做 Positive-Unlabeled 监督；发布约 1,244 km² / 1,130 km 越野全局导航数据集；Warthog 真机长程规划相对纯局部规划干预约少 85%。

## 开源状态（步骤 2.5）

- **项目页：** 无 `*.github.io` / lab 项目页。核查以 arXiv HTML 与 Hugging Face 数据集页为准（2026-09-21）。
- **数据集：** [anony-008/offroad-global-nav](https://huggingface.co/datasets/anony-008/offroad-global-nav) **已发布**（CC BY-NC-4.0；约 29.9 GB）。HF 账号为匿名组织（双盲投稿残留）；论文作者已在 arXiv 署名。卡片写 299 scenes / ~1,244 km² / 1,130 km GPS；HF 元数据截至入库日约 **201 rows**，并注明还会追加。
- **代码：** arXiv / HF 均无 GitHub 训练或部署入口。同组前作 [Trailblazer](https://github.com/unmannedlab/Trailblazer)（arXiv:2505.09739）是 **可微 Neural A\*** 路线，不能当作本文 nnPU + FiLM 双头的复现仓。
- **结论：** **部分开源** — 多模态地理数据集可下载；训练 / 推理代码与权重 **待发布**。

## 摘录 1：问题与贡献（§Abstract–§I）

- **缺口：** 机载传感只能看近场，安全长程路线取决于视野外地形；现有方法要么从局部交互学几何可通行（缺人类意图），要么用可微 planner 跟示范（缺物理约束）。
- **方法：** 统一模型吃卫星 / 航拍 LiDAR / OSM，同时预测连续 traversability 与 goal-conditioned path likelihood；人类 GPS 用 **nnPU** 监督，LiDAR 几何作自监督 prior，**不需要** 可微 A\* 或稠密人工标注。
- **数据：** 299 场景、约 **1,244 km²**、**1,130 km** 人类驾驶轨迹（作者称目前越野导航地理覆盖最大）。
- **真机：** Clearpath Warthog；7 条路线、两处场地；路径长度相对人类 **+3.66%**（SRP-only **+9.7%**）；相对纯局部规划 **干预约 −85%**。

**对 wiki 的映射：** 升格 [`wiki/entities/paper-offroad-global-nav.md`](../../wiki/entities/paper-offroad-global-nav.md)；全局搜索层见 [`wiki/methods/a-star.md`](../../wiki/methods/a-star.md)；局部分层见 [`wiki/methods/mppi.md`](../../wiki/methods/mppi.md)、[`wiki/comparisons/mobile-robot-navigation-planning-methods.md`](../../wiki/comparisons/mobile-robot-navigation-planning-methods.md)。

## 摘录 2：数据与网络（§III–§IV）

- **采集：** USGS 航拍 LiDAR（5–27 pts/m²，垂向约 10 cm）定界 → ArcGIS 30 cm/px 卫星 → OSM 公路/步道/水系 → OSM 公开 GPS。季节跨度覆盖植被变化。
- **预处理：** LiDAR 栅格化 height / slope / intensity（intensity 取补 \(1-I\)，使高值=更难通行）；OSM 栅格对齐；GPS 轨迹扩成 200 m–1.5 km 走廊；融合分辨率 256×256，损失/推理上采样到 1024×1024。
- **架构：** 冻结 **DINOv3-SAT ViT-L/16**（SAT-493M）+ FPN；LiDAR+OSM 三阶 CNN；**空间 FiLM** 用卫星特征调制 LiDAR；起终点热图拼接后双头（path logits / traversability）。
- **损失：** nnPU path corridor（\(r=5\) px 正样本带）+ TV + mass；OSM track / 历史轨迹弱正（\(T^*=0.75\)）；LiDAR intensity / slope（\(\sigma_s=0.445\)≈30°）/ height-gradient prior；辅助项 3 epoch 线性 warmup。\(\lambda_p=1.0\)，其余 \(\lambda\le 0.2\)。

**对 wiki 的映射：** 实体页画 flowchart；数据集细节见 [`sources/datasets/offroad-global-nav.md`](../datasets/offroad-global-nav.md)。

## 摘录 3：离线基准、消融与真机（§V–§VII）

- **训练：** RTX 4090 24 GB；7533 / 1716 训练/验证（场景级 80/20）；train loss 0.065 / val 0.076。
- **Costmap 基准（Table I，n=356，含 TartanDrive 2.0 持出）：** A\* \(K=4\) 后取最小离散 Fréchet。本文 \(d_F=61.3\pm14.3\) m、\(r_c=0.93\pm0.05\)，优于 Height / Trailblazer / OVerSeeC；OVerSeeC 的 \(r_d\) 更好但 \(r_c=0.23\) 说明抄近路穿过人类拒绝的地形。
- **消融（Table II）：** 仅 path corridor 时稠密走廊 vs 稀疏备选 \(\Delta=0.019\)；加 OSM/轨迹弱正 + LiDAR prior 后 \(\Delta=0.200\)。
- **部署：** LRP 把 \(C=1-T\) 交给加权 A\* + Yen \(K\)-shortest；路点进 ARL Phoenix：DLIO 定位 + TerrainNet 局部 costmap 上的 **MPPI**（SRP）。速度 ≤2 m/s。
- **场地：** Levee（250 m 堤+湖+跑道）与 TAMU RELLIS 越野（>9 km²）。Table III：7 条路线人类 vs LRP vs SRP；最长约 1.7–2.0 km；Route 6 遇到地图外沙堆后注入高代价并重规划。
- **局限：** LRP–SRP lookahead/handoff 敏感；静态 costmap 不覆盖季节变化；PU 缺显式负例导致植被过渡带偏保守（Route 4 比人类长约 17%）。

**对 wiki 的映射：** 与室内可通行规划 [TravExplorer](../../wiki/entities/paper-travexplorer.md) 对照：本文是 **越野轮式 UGV × 开销地理先验 × 全局 costmap**，不是室内 ObjectNav。

## 建议 wiki 动作

- 新建 **`wiki/entities/paper-offroad-global-nav.md`**（`## 源码运行时序图` 标不适用）。
- 新建 `sources/datasets/offroad-global-nav.md`。
- 交叉 [`a-star`](../../wiki/methods/a-star.md)、[`mppi`](../../wiki/methods/mppi.md)、[`mobile-robot-navigation-planning-methods`](../../wiki/comparisons/mobile-robot-navigation-planning-methods.md)、[`paper-travexplorer`](../../wiki/entities/paper-travexplorer.md)。
- 注册机构 alias `arl`（DEVCOM ARL）；`texas-am` 已在注册表。
