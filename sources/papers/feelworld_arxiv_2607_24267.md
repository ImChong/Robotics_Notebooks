# FeelWorld: Visuo-Tactile World Model for Hierarchical Contact Prediction and Planning（arXiv:2607.24267v1）

> 来源归档（ingest）

- **标题：** FeelWorld: Visuo-Tactile World Model for Hierarchical Contact Prediction and Planning
- **类型：** paper / world-model / visuo-tactile / contact-rich / model-based-planning / manipulation
- **arXiv abs：** <https://arxiv.org/abs/2607.24267>
- **PDF：** <https://arxiv.org/pdf/2607.24267v1>
- **HTML：** <https://arxiv.org/html/2607.24267v1>
- **机构：** 中国科学院自动化研究所（CASIA）；ImprintX Robotics；北京智源人工智能研究院（BAAI）
- **作者：** Wenxuan Ma、Chaofan Zhang、Chao Xue、Yinghao Cai、Guocai Yao、Shaowei Cui*、Shuo Wang（* 通讯：shaowei.cui@ia.ac.cn）
- **入库日期：** 2026-09-30
- **页数 / 图：** 9 pages, 7 figures（arXiv 元数据）
- **一句话说明：** 分层预测 **contact / 3D tactile latent / slip** 的视触觉世界模型 + **contact-gated asymmetric attention**；真机 chip/fruit/USB 上 **contact-aware CEM** 平均成功率 **81.7%**（较 vision-only +32.5 pp）。

## 开源 / 项目页核查（2026-09-30）

- **项目页：** arXiv 与正文 **未给出** 独立 project page URL。
- **代码：** 无官方 GitHub / HF 模型仓链接；勿与无关账号 [`github.com/feelworld`](https://github.com/feelworld)（旧 ROS/工具仓库）或视频切换台仓库混淆。
- **相关生态：** ImprintX 提供 [`imprintx` PyPI SDK](https://pypi.org/project/imprintx/)（传感器/手套采集），**不含** FeelWorld 训练/规划代码。
- **判定：** **截至入库日未开源** FeelWorld 实现。

## 摘要级要点

- **动机：** 纯视觉 WM 在接触丰富任务上「看起来合理」但可能违背接触物理；触觉在 **非接触段** 多为噪声，需 **分层 + 门控** 融合。
- **三层触觉状态：** (1) **contact** 二值；(2) **3D tactile latent**（FG-CLTP 编码点云，力相关几何）；(3) **slip**（时序 1D conv，focal loss，仅 contact 帧监督）。
- **Contact-gated asymmetric attention：** 视觉 token 保留 **visual-only** 自注意力路径；触觉 attend 视觉；预测 contact 概率 **门控** 视觉→触觉 cross-attention。
- **训练：** 冻结 **V-JEPA 2** 视觉编码 + 冻结 **FG-CLTP** 触觉编码；联合 teacher forcing + **自回归 rollout** + **context noise injection**（\(R=4\), \(\sigma=0.007\)）。
- **规划：** **Contact-aware CEM** — contact 前仅视觉 goal distance；contact 后加 tactile + slip penalty（\(H_p=6\), \(N=400\), 8 elites, 8 iters, 执行前 2 步 action）。

## 核心摘录（面向 wiki 编译）

### 1) 10-step 视觉预测（Table I 节选）

| Method | LPIPS ↓ | Contact F1 | Slip F1 |
|--------|---------|------------|---------|
| V-JEPA 2 (visual-only) | 0.084 | – | – |
| FeelWorld (Ours) | **0.058** | **0.981** | **0.834** |

### 2) 真机 zero-shot CEM（Table IV，40 trials/task）

| Planner | Chip | Fruit | USB | Avg |
|---------|------|-------|-----|-----|
| Visual-only CEM | 40.0 | 70.0 | 37.5 | 49.2 |
| Contact-aware CEM | **82.5** | **87.5** | **75.0** | **81.7** |

### 3) 平台与数据

- **机器人：** Imeta-Y1；三 RGB + **DM tactile**（3D 点云 + contact API）。
- **数据：** 每任务 200 train / 40 test traj；30 Hz 采集，训练 **6 fps**；含约 **30%** 失败轨迹（滑移、插装失败等）。

### 4) 对 wiki 的映射

| 主题 | 目标页 |
|------|--------|
| 论文实体 | `wiki/entities/paper-feelworld.md` |
| 生成式世界模型 | `wiki/methods/generative-world-models.md` |
| 视触觉融合概念 | `wiki/concepts/visuo-tactile-fusion.md` |
| 同族 WM | `wiki/entities/paper-sa-2603-19201-omnivta-visuo-tactile-world-modeling-for-contact.md` |

## 推荐继续阅读

- [arXiv:2607.24267v1 PDF](https://arxiv.org/pdf/2607.24267v1) — 门控注意力与 CEM 目标（Eq. 15）
- [V-JEPA 2](https://arxiv.org/abs/2506.09885) — 视觉骨干
- [OmniVTA（arXiv:2603.19201）](../../wiki/entities/paper-sa-2603-19201-omnivta-visuo-tactile-world-modeling-for-contact.md) — 视触觉 WM 对照
