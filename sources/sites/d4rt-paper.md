# D4RT — Efficiently Reconstructing Dynamic Scenes One D4RT at a Time

> 来源归档（ingest · 项目页）

- **标题：** D4RT: Efficiently Reconstructing Dynamic Scenes One D4RT at a Time
- **类型：** site / project-page
- **项目页：** <https://d4rt-paper.github.io/>
- **论文：** <https://arxiv.org/abs/2512.08924>
- **机构：** 谷歌 DeepMind（Google DeepMind）；牛津大学 VGG 等
- **会议（项目页 BibTeX）：** CVPR 2026
- **入库日期：** 2026-09-10
- **代码：** 截至 2026-09-10 **项目页未列 GitHub / Hugging Face / Zenodo**（确认未开源）
- **DeepMind 通稿：** <https://deepmind.google/blog/d4rt-teaching-ai-to-see-the-world-in-four-dimensions/>
- **一句话说明：** 统一 transformer 前馈 4D 重建：全局自注意力编码器 → **Global Scene Representation**，轻量查询解码器对任意 $(u,v,t_{\text{src}},t_{\text{tgt}},t_{\text{cam}})$ 独立输出 3D 点位置，统一深度 / 相机 / 稀疏–稠密 3D 跟踪 / 全像素重建。

## 项目页要点（2026-09-10 核查）

### 能力（Capabilities）

| 模式 | 查询设定 | 输出 |
|------|----------|------|
| **3D Tracking** | 固定源像素 + 变 $t_{\text{tgt}}=t_{\text{cam}}$ | 局部相机坐标稀疏 3D 轨迹 |
| **3D Reconstruction** | 深度 + 相机位姿投影 | 无需显式对应；动态物去重 |
| **All pixels tracking** | 全像素查询 | 世界坐标 holistic 4D 重建 |

### 方法（Method）

- **Encoder：** 全局自注意力，将视频映射为 latent **Global Scene Representation** $F$。
- **Decoder：** 轻量 cross-attention；查询含 **Fourier 2D 坐标 + 时间嵌入 + 源点周围 local RGB patch**（论文为 $9{\times}9$）。
- **查询解耦：** 空间索引 $(u,v,t_{\text{src}})$ 与时间/相机索引 $(t_{\text{tgt}},t_{\text{cam}})$ **可不一致**，支持任意时空探针。

### 开源核查（步骤 2.5）

| 项 | 结论 |
|----|------|
| 项目页 Code / GitHub | **无链接** |
| arXiv PDF code availability | 未声明公开仓库 |
| DeepMind 博客 | 无官方代码链接 |
| **判定** | **确认未开源**（截至 2026-09-10） |

## 对 wiki 的映射

- [paper-d4rt](../../wiki/entities/paper-d4rt.md) — 本次升格主实体页
- [GenCeption](../../wiki/entities/genception.md) — 深度任务对照专家之一
- [VGG-T³](../../wiki/entities/paper-vgg-ttt.md) — VGGT 系离线前馈几何对照
- [State Estimation](../../wiki/concepts/state-estimation.md) — 相机 / 稠密几何上游

## 当前提炼状态

- [x] 项目页与 DeepMind 博客核查
- [x] 开源状态：确认未开源
- [x] 升格 wiki 实体页
