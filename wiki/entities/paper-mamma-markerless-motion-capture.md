---

type: entity
tags: [paper, motion-capture, markerless, smpl-x, multi-view, dense-landmarks, cvpr-2026, human-motion, eth, max-planck]
status: complete
updated: 2026-06-09
arxiv: "2506.13040"
venue: "2026 · CVPR Oral"
code: https://github.com/cuevhv/mamma
related:
  - ../concepts/motion-retargeting-pipeline.md
  - ../concepts/motion-retargeting.md
  - ../concepts/whole-body-tracking-pipeline.md
  - ./freemocap.md
  - ./gvhmr.md
  - ./paper-opencap-monocular.md
  - ./amass.md
  - ../methods/motion-retargeting-gmr.md
  - ../overview/paper-notebook-category-14-human-motion.md
sources:
  - ../../sources/papers/mamma_arxiv_2506_13040.md
  - ../../sources/sites/mamma-tue-mpg-de.md
  - ../../sources/repos/mamma.md
summary: "MAMMA（CVPR 2026 Oral）：多视角 markerless 管线，用 MammaNet 预测 512 稠密体表 landmark（含可见性/不确定性/接触）并优化 SMPL-X，专攻双人近距离交互；精度对标商业 Vicon，开源推理与训练。"
---

# MAMMA：无标记多视角多人 SMPL-X 动作捕捉

**MAMMA**（*Markerless Accurate Multi-person Motion Acquisition*，arXiv:2506.13040，CVPR 2026 Oral，[项目页](https://mamma.is.tue.mpg.de/)，[代码](https://github.com/cuevhv/mamma)）提出一套 **全自动 markerless motion capture 管线**：从 **同步多视角视频**（棚拍 32 相机或 **4 台 iPhone**）恢复 **一至两人** 每帧的 **SMPL-X** 姿态与体型，在 **近距离交互、手部细节与重度遮挡** 场景下仍保持与 **商业 marker-based Vicon** 可比的重建质量。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| MoCap | Motion Capture | 动作捕捉，本方法输出 SMPL-X 时序供动画/重定向消费 |
| SMPL-X | Skinned Multi-Person Linear Model with eXpressions | 带手/脸的参数化人体网格与姿态参数 |
| HMR | Human Mesh Recovery | 从图像恢复人体网格/姿态；MAMMA 走多视角稠密 landmark 路线 |
| MPJPE | Mean Per-Joint Position Error | 关节位置平均误差（mm），常用 3D 评测指标 |
| PVE | Per-Vertex Error | 网格顶点平均误差（mm） |
| SAM | Segment Anything Model | Meta 分割模型；MAMMA 用 SAM 2 做每人 mask 与跨帧跟踪 |
| IK | Inverse Kinematics | 重定向下游将人体轨迹映射到机器人关节角 |
| Retargeting | Motion Retargeting | 人体参考动作→机器人可执行参考的映射 |

## 为什么重要

- **降低高质量 MoCap 门槛：** 相对光学 marker 系统（贴标、清 swapped marker、MoSh++ 后处理），MAMMA 强调 **像素级全自动**；论文报告 held-out marker 误差与 Vicon 管线仅差 **~1.6 mm**，动画视觉难区分。
- **填补双人交互空白：** 多数学术 HMR / markerless 方法面向单人稀疏关键点；MAMMA 用 **mask 条件化 + per-landmark query + 可见性估计** 处理拥抱、舞蹈、搬运等 **人–人遮挡**。
- **机器人上游数据源：** 输出 **SMPL-X 序列** 可直接进入 [Motion Retargeting Pipeline](../concepts/motion-retargeting-pipeline.md)（与 [AMASS](./amass.md) 棚拍 SMPL 库、[GVHMR](./gvhmr.md) 单目估计形成 **多视角高质量 / 单目便捷 / 大库离线** 三角）；相对 [FreeMoCap](./freemocap.md) 更偏 **研究级精度与 SMPL-X 参数**，而非教学向低成本 GUI。
- **可复现工程栈：** 开源 `ma_cap → ma_masks → ma_2d → ma_3d → ma_vis` CLI + 浏览器 GUI，附带 MAMMASyn 训练数据与 MAMMAEval 基准。

## 流程总览

```mermaid
flowchart TB
  subgraph in [输入]
    mv["同步多视角视频<br/>32-cam 棚拍 或 4×iPhone"]
    cal["相机标定<br/>内参/外参"]
  end
  subgraph seg [ma_masks]
    det["YOLO 人体检测"]
    sam["SAM 2 mask 跟踪<br/>每人实例分割"]
    det --> sam
  end
  subgraph net [ma_2d · MammaNet]
    vit["ViT-Base 图像特征"]
    msk["Mask CNN 条件"]
    dec["512 landmark queries<br/>Transformer decoder"]
    out2d["μ, σ, p_vis, p_contact"]
    vit --> dec
    msk --> dec
    dec --> out2d
  end
  subgraph match [跨视角匹配]
    epi["对称极线距离亲和"]
    hung["Hungarian 一对一分配"]
    graph["环一致对应图"]
    epi --> hung --> graph
  end
  subgraph opt [ma_3d · SMPL-X 优化]
    s1["Stage 1: 平移/旋转重投影"]
    s2["Stage 2: pose+shape<br/>Geman-McClure"]
    s3["Stage 3: 接触能量 + Huber 精修"]
    smpl["SMPL-X β, θ, t 每帧每人"]
    s1 --> s2 --> s3 --> smpl
  end
  mv --> det
  sam --> net
  cal --> match
  out2d --> match
  graph --> opt
  smpl --> vis["ma_vis 叠加 / Rerun"]
```

## 核心机制（归纳）

### 1）MammaNet 稠密 landmark 估计

- **512 顶点** 经 FPS 采样自 SMPL-X 表面（手/脚/头加权），作为 **虚拟 marker**。
- **架构：** ViT-Base + mask 特征逐元素相加；Transformer decoder 为 **每个 landmark 学习独立 query**（相对 [CameraHMR](https://camerahmr.is.tue.mpg.de/) 单 token 解码全组 landmark 更利于部位注意力）。
- **输出：** 2D 位置 μ、不确定性 σ、**可见性概率 p**（遮挡交互关键）、项目页补充 **人–人/人–地接触概率**。
- **训练：** 纯 **MAMMASyn 合成数据**（扩展 BEDLAM 32 相机数字孪生）；损失为高斯 NLL + 可见性 BCE。

### 2）MAMMASyn 合成数据

| 子集 | 内容 | 目的 |
|------|------|------|
| MAMMASyn-S | BEDLAM + MOYO | 单人极端姿态 |
| MAMMASyn-I | Hi4D / Harmony4D / Inter-X + 自采拉丁舞/情侣交互 | 双人接触与遮挡 |
| MAMMASyn-H | Interhand2.6M + SignAvatars | 手部高保真 |

约 **2.5M crops**、955k 图像；6 fps、2056×1504 渲染，附带 per-subject mask、深度与顶点可见性 GT。

### 3）跨视角匹配与身份

- 两视角 landmark 集以 **对称极线距离** 度量几何亲和；**Hungarian** 求一对一匹配；高亲和边构成 **环一致对应图**。
- SAM 2 **跨帧传播** mask，分离重叠个体——vision-based 系统相对 marker 的结构性优势（可融合更丰富的像素监督）。

### 4）SMPL-X 拟合（无回归初始化）

- 初始：多射线最近点 + 「L」形手臂；**不用** HMR 回归 pose/shape。
- **三阶段 L-BFGS：** 重投影 → 全参数 + 鲁棒 Geman-McClure → 接触排斥/吸引 + Huber；含 shape/pose 正则与帧间关节平滑。
- 重投影项按 **p_vis 加权、σ 归一化**，自动降权不可见或高不确定 landmark。

## 实验与评测（索引级）

- **2D landmark：** MammaNet（+mask）在 RICH、Harmony4D、CHI3D、MAMMAEval-S/D、MOYO 上优于 Look-Ma* / CameraHMR。
- **3D 拟合（MPJPE/PVE mm）：** 全面领先学术基线；双人 Harmony4D/CHI3D 增益明显。
- **新基准：** MAMMAEval-Singles（22 单人序列）、MAMMAEval-Dance（17 西海岸摇摆舞双人）；填补 dense-landmark 与 markerless mocap 评测空白。
- **Vicon 对标：** 额外 37 marker/人作独立 GT；markerless vs MoSh++ held-out 误差差 ~1.6 mm。
- **局限：** 当前聚焦 **最多两人**；需 **标定多相机**；非商业科研许可；真机部署仍需 [GMR](../methods/motion-retargeting-gmr.md) 等重定向与物理筛选。

## 常见误区

1. **MAMMA = 单目手机 HMR：** 核心是 **同步多视角**（虽可用 4 iPhone，但仍需标定与几何一致）；与 [GVHMR](./gvhmr.md) 单目深度歧义问题不同。
2. **输出可直接驱动人形：** SMPL-X 是 **人体运动学模型**；上机器人需 [Motion Retargeting](../concepts/motion-retargeting.md) 与接触/平衡约束，不能直接当关节指令。
3. **与 FreeMoCap 重复：** FreeMoCap 偏 **开源低成本教学 GUI**；MAMMA 偏 **MPI-IS 研究级 SMPL-X 精度与双人交互**，许可与硬件假设也不同。
4. **合成训练 = 不能泛化真机：** 论文强调对 MOYO 等 **极端 OOD 姿态** 与真实 MAMMAEval 序列的强泛化；但野外仍受标定与遮挡限制。

## 与其他页面的关系

- **重定向流水线：** [Motion Retargeting Pipeline](../concepts/motion-retargeting-pipeline.md) — MAMMA 属于上游 **「干净多视角 → SMPL-X」** 源，与棚拍 BVH、AMASS 同级，优于单目估计噪声源。
- **全身跟踪下游：** [Whole-Body Tracking Pipeline](../concepts/whole-body-tracking-pipeline.md) — 精炼 SMPL-X 可作为 WBT/模仿学习参考轨迹输入。
- **开源 MoCap 对照：** [FreeMoCap](./freemocap.md) — 低成本多相机平台；[GVHMR](./gvhmr.md) — 单目视频上游；[OpenCap Monocular](./paper-opencap-monocular.md) — 单手机 OpenSim 运动学/动力学。
- **动捕数据生态：** [AMASS](./amass.md) — 统一 SMPL 大库；MAMMA 提供 **现场采集** 到 SMPL-X 的替代 Vicon 路径。
- **论文笔记索引：** [Human Motion 分类](../overview/paper-notebook-category-14-human-motion.md)

## 参考来源

- [MAMMA 论文归档（arXiv:2506.13040）](../../sources/papers/mamma_arxiv_2506_13040.md)
- [MAMMA 项目页](../../sources/sites/mamma-tue-mpg-de.md)
- [cuevhv/mamma 代码索引](../../sources/repos/mamma.md)

## 推荐继续阅读

- [MAMMA 项目页](https://mamma.is.tue.mpg.de/) — 演示视频、Vicon 对比与 iPhone demo
- [GitHub 仓库](https://github.com/cuevhv/mamma) — 安装、`quick.yaml` demo 与 GUI
- [arXiv:2506.13040](https://arxiv.org/abs/2506.13040) — 完整方法与消融
- [BEDLAM](https://bedlam.is.tue.mpg.de/) — MAMMASyn 合成数据基座
- [Motion Retargeting Pipeline](../concepts/motion-retargeting-pipeline.md) — SMPL-X 输出如何接入机器人栈
