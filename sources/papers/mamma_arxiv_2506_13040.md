# MAMMA: Markerless & Automatic Multi-Person Motion Action Capture（arXiv:2506.13040）

> 来源归档（ingest）

- **标题：** MAMMA: Markerless & Automatic Multi-Person Motion Action Capture
- **类型：** paper / markerless motion capture / dense landmarks / multi-view / SMPL-X
- **arXiv abs：** <https://arxiv.org/abs/2506.13040>
- **会议：** CVPR 2026 Oral
- **作者：** Hanz Cuevas-Velasquez, Anastasios Yiannakidis, Soyong Shin, Giorgio Becherini, Markus Höschle, Joachim Tesch, Taylor Obersat, Tsvetelina Alexiadis, Michael J. Black（MPI-IS Tübingen / CMU）
- **项目页：** <https://mamma.is.tue.mpg.de/>
- **代码：** <https://github.com/cuevhv/mamma>
- **入库日期：** 2026-06-09
- **一句话说明：** 从同步多视角视频 **全自动** 恢复 **双人近距离交互** 的 **SMPL-X** 参数；核心为 **MammaNet**（512 稠密体表 landmark queries + 可见性/不确定性/接触概率）+ 跨视角几何匹配 + 多阶段 SMPL-X 优化，精度接近商业 marker-based Vicon。

## 摘要级要点

- **问题：** 传统光学 marker mocap 需专用硬件、贴标与大量后处理；现有学习型 markerless 方法多面向单人、稀疏关键点，或在遮挡与双人交互下失效。
- **核心思路：** 两阶段——(1) 各视角预测 **512 稠密 2D 体表 landmark**（含不确定性 σ 与可见性 p）；(2) 跨视角匹配后 **拟合 SMPL-X**（β, θ, t），无需回归式 pose 初始化。
- **MammaNet：** ViT-Base 图像特征 + mask CNN；Transformer decoder 为 **每个 landmark 学习独立 query embedding**（相对 CameraHMR 单 token 解码整组 landmark）；可选 **SAM 2 分割 mask 条件化**，显著改善双人交互歧义。
- **MAMMASyn 合成训练集：** 扩展 BEDLAM 至实验室 **32 相机**数字孪生；2.5M crops / 955k 图像；子集 **S**（单人 BEDLAM+MOYO）、**I**（双人交互 Hi4D/Harmony4D/Inter-X + 自采拉丁舞/情侣交互）、**H**（Interhand2.6M + SignAvatars 手部）。
- **跨视角匹配（项目页补充）：** 对称极线距离 + Hungarian 一对一分配 + 环一致对应图；SAM2 跨帧传播维持身份。
- **SMPL-X 优化：** 三阶段 L-BFGS——(1) 仅平移/旋转重投影；(2) pose+shape+translation，Geman-McClure 鲁棒项；(3) Huber 精修；损失含 shape/pose 正则与帧间平滑；项目页增加 **人–人/人–地接触** 能量（排斥+吸引）。
- **评测：** 单人 RICH/MOYO；双人 Harmony4D/CHI3D；新建 **MAMMAEval-Singles / MAMMAEval-Dance**；相对 Look-Ma* / CameraHMR / SMPLify 在 MPJPE/PVE 全面更优。
- **与 Vicon 对比：** 37 个 held-out marker（不参与 MoSh++ 拟合）上，markerless 与 marker-based 误差差 **~1.6 mm**，动画视觉难区分——支持将 markerless 输出作 SMPL-X 采集 ground truth。
- **工程扩展：** 除 32 路棚拍 RGB，演示 **4 台 iPhone** 室内外采集；代码管线 `ma_cap → ma_masks → ma_2d → ma_3d → ma_vis`。
- **发布：** 数据集、benchmark、训练代码与预训练权重（学术用途；非商业科研许可）。

## 核心摘录（面向 wiki 编译）

### 方法栈

| 模块 | 输入 | 输出 / 作用 |
|------|------|-------------|
| `ma_masks` | 多视角视频 | SAM + YOLO 每人分割 mask（跨帧跟踪） |
| MammaNet (`ma_2d`) | 图像 + mask | 512×(μ, σ, p, 接触概率) 每视角 |
| Multiview matching | 各视角 landmark 集 | 极线距离亲和 + Hungarian → 跨视角对应 |
| SMPL-X fit (`ma_3d`) | 匹配 landmark + 相机标定 | 每帧每人 β, θ, t |
| `ma_vis` | 优化结果 | 叠加可视化 / Rerun 场景 |

### MammaNet 损失（论文式 1–2）

- \(\mathcal{L}_{\text{ldmk}}\)：高斯 NLL，逐 landmark 加权
- \(\mathcal{L}_{\text{vis}}\)：可见性二值交叉熵（遮挡交互关键）

### 与相关方法对比（归纳）

| 方法 | 输入 | 输出 | 双人交互 | 手部/稠密表面 |
|------|------|------|----------|---------------|
| 商业 markerless（Captury 等） | 多相机 | 骨架/点云 | 部分支持 | 通常无 SMPL-X |
| CameraHMR | 单/多视图 | 稠密 landmark | 弱 | 单 token 解码 |
| **MAMMA** | 同步多视角（1–2 人） | SMPL-X 序列 | **mask 条件 + 可见性** | 512 FPS 采样顶点 |

### 代码与复现要点（GitHub README）

- 环境：`micromamba`/`conda`，详见 `docs/INSTALL.md`；`python -m inference doctor` 校验权重
- 快速 demo：`configs/examples/presets/quick.yaml` + 4-cam iPhone 样例 `data/mamma_example/`
- GUI：`gui/scripts/dev.sh`（Flask + React）
- 数据集：项目页注册账号下载 dance / multi-person / iPhone / eval / synthetic 五类
- 许可：非商业科研 [LICENSE](https://github.com/cuevhv/mamma/blob/main/LICENSE)

## 对 wiki 的映射

- 沉淀实体页：[MAMMA（CVPR 2026）](../../wiki/entities/paper-mamma-markerless-motion-capture.md)
- 项目页归档：[sources/sites/mamma-tue-mpg-de.md](../sites/mamma-tue-mpg-de.md)
- 代码归档：[sources/repos/mamma.md](../repos/mamma.md)

## 参考来源（原始）

- 论文：<https://arxiv.org/abs/2506.13040>
- 项目页：<https://mamma.is.tue.mpg.de/>
- 代码：<https://github.com/cuevhv/mamma>
