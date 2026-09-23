# Room-Mediated Co-occurrence for Zero-Shot Object-Centric Semantic Navigation via Frontier Scoring（arXiv:2607.25448）

> 来源归档（ingest）

- **标题：** Room-Mediated Co-occurrence for Zero-Shot Object-Centric Semantic Navigation via Frontier Scoring
- **类型：** paper / objectnav / zero-shot / semantic-navigation / habitat / training-free
- **arXiv abs：** <https://arxiv.org/abs/2607.25448>
- **PDF：** <https://arxiv.org/pdf/2607.25448>
- **项目页：** <https://uts-ri.github.io/RPV-SemNav/> — 归档见 [`sources/sites/rpv-semnav-uts-ri.md`](../sites/rpv-semnav-uts-ri.md)
- **代码：** <https://github.com/UTS-RI/RPV-SemNav> — 归档见 [`sources/repos/rpv-semnav.md`](../repos/rpv-semnav.md)
- **机构：** 悉尼科技大学机器人研究所（University of Technology Sydney, UTS-RI）— Adam Scicluna、Gavin Paul、Alen Alempijevic
- **venue：** IROS 2026（accepted，README 标注）
- **入库日期：** 2026-09-23
- **一句话说明：** **训练-free、object-centric** 零样本 ObjectNav：用 CLIP 将物体标签映射为 **Room Probability Vector（RPV）**，以房间共现重叠估计目标关联，经 **Fast Marching geodesic flood-fill** 注入 value map 并排 frontier；HM3D val 上 SR **+3%**、SPL **+1.3%**（相对 image-holistic baseline）。

## 相关资料（策展）

| 类型 | 链接 | 说明 |
|------|------|------|
| 代码 | <https://github.com/UTS-RI/RPV-SemNav> | Habitat 0.3.3 + VLFM  fork；README 详述安装 |
| 对照 VLFM | [VLFM](https://arxiv.org/abs/2306.14846) | image-holistic BLIP-2 value map |
| 对照 SemUtil / SEEK | 物体/房间 embedding 导航 | 直接语言相似 ≠ 空间共现 |
| 仿真 | [Habitat](../repos/habitat-sim.md)（若已归档） | HM3D ObjectNav 评测 |

## 摘要级要点

- **问题：** 零样本 ObjectNav 常用 VLM **物体–物体 latent 相似度**，但 linguistic/visual 相似 **≠ 空间共现**（如 microwave 与 bed）。
- **RPV：** 紧凑 **room lexicon**；每个 object label → CLIP text embedding → **房间概率分布**；目标–检测物共现 = RPV overlap。
- **地图：** open-vocab 检测（YOLO + SAM）→ 物体位置 → geodesic **flood-fill**（FMM）传播 semantic signal（自适应衰减）→ **frontier scoring** → pointNav。
- **感知管线：** 无关区域 mask/blur → YOLO 框 → SAM 实例 mask → 语义打分（Fig. 3）。
- **任务：** Habitat ObjectNav；initialize 12×30° 环视；动作 {forward 0.25m, turn 30°, stop}；500 步内距目标 1m 成功。
- **结果：** HM3D validation — SR +3%、SPL +1.3% vs image-holistic baselines；强调 **可解释性** 与 **开放词汇**。

## 核心摘录（面向 wiki 编译）

### 1) 与 image-holistic 方法差异

- VLFM 等：整帧 embedding vs 目标文本 → visibility-cone 投影，FOV 外 frontier 更新慢。
- 本文：**检测级 anchor** + **room-mediated** 共现，信号与几何可达性通过 geodesic 绑定。

### 2) 开源状态（GitHub + README，2026-09-23）

| 组件 | 状态 |
|------|------|
| 评测管线 / 安装文档 | **已开源**（仓库 +  lengthy README） |
| V1 代码整理 | README TODO: *Add V1 code* |
| Checkpoint 下载 | SAM3 / YOLOE / Mask2Former 链接 **WIP** |

## 对 wiki 的映射

- 新建：[paper-rpv-semnav](../../wiki/entities/paper-rpv-semnav.md)
- 交叉：[zero-shot-object-navigation](../../wiki/tasks/zero-shot-object-navigation.md)、[embodied-semantic-cognitive-map](../../wiki/concepts/embodied-semantic-cognitive-map.md)

## 当前提炼状态

- [x] arXiv + 项目页 + GitHub 核查
- [x] 开源：已开源（checkpoint 链 WIP）
- [x] 可运行入口：`python -m vlfm.run` + `scripts/launch_dl_servers.sh`
