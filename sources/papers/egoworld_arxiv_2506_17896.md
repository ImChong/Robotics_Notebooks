# EgoWorld：Translating Exocentric View to Egocentric View using Rich Exocentric Observations

> 来源归档（ingest）

- **标题：** EgoWorld: Translating Exocentric View to Egocentric View using Rich Exocentric Observations
- **类型：** paper
- **机构：** LG Electronics AI Lab；韩国科学技术院（KAIST）；牛津大学 Visual Geometry Group（VGG）
- **venue：** ICLR 2026
- **原始链接：**
  - <https://arxiv.org/abs/2506.17896>
  - PDF：<https://arxiv.org/pdf/2506.17896>
  - OpenReview：<https://openreview.net/forum?id=wcTuZG9P2o>
  - 项目页：<https://redorangeyellowy.github.io/EgoWorld/>
  - 代码：<https://github.com/redorangeyellowy/EgoWorld>
- **入库日期：** 2026-07-24
- **一句话说明：** 从**单张**第三人称（exocentric）图，经深度点云重投影稀疏 egocentric RGB、3D 第一人称手姿与 VLM 文本描述，再条件化潜扩散 inpainting，生成稠密第一人称视图；在 H2O / TACO / Assembly101 / Ego-Exo4D 上 SOTA，并展示野外泛化。
- **同名消歧：** 与 StellarNex 的 [EgoWorld-100W](../blogs/stellarnex_egoworld_100w.md) **百万级自中心操作数据集无关**（仅共享 “EgoWorld” 品牌词）。

## 核心论文摘录（MVP）

### 1) 问题：单视角 exo→ego 仍被 2D 先验与推理假设卡住

- **链接：** <https://arxiv.org/abs/2506.17896>
- **摘录要点：** 现有 exo→ego 方法常依赖 2D hand layout（如 Exo2Ego）、已知相对相机位姿（如 4Diff）、多视角或初始 egocentric 帧；单张第三人称图下视角差、遮挡与外观变化使问题高度欠定。EgoWorld 主张用 **点云几何 + 3D 手姿 + 文本语义** 的多模态观测，把问题改写成「稀疏 egocentric 观测上的条件生成」。
- **对 wiki 的映射：**
  - [EgoWorld（论文）](../../wiki/entities/paper-egoworld.md) — 写清相对 Exo2Ego / 4Diff 的设定差异与可部署边界。

### 2) 两阶段管线：\(\Phi_{exo}\) 观测抽取 → \(\Phi_{ego}\) 扩散重建

- **链接：** 论文 §3.1–3.3
- **摘录要点：**
  - \(\Phi_{exo}(I_{exo})\to(S_{ego},P_{ego},T_{exo})\)：深度估计 → MANO 手姿标定尺度 → 点云；ViT+MLP 直接从 \(I_{exo}\) 回归 egocentric 3D 手姿 \(P_{ego}\)；Umeyama 求 exo↔ego 变换后投影得稀疏 RGB \(S_{ego}\)；VLM 产出场景/交互文本 \(T_{exo}\)。
  - \(\Phi_{ego}\)：VAE 编码 \(S_{ego}\) 与 2D 手姿图，与噪声潜变量拼接成 9 通道输入 U-Net；CLIP 文本作 cross-attention；CFG 强化文本引导，解码得 \(\hat{I}_{ego}\)。
- **对 wiki 的映射：**
  - [EgoWorld（论文）](../../wiki/entities/paper-egoworld.md) — 流程总览 + 源码运行时序图对齐 `train.py` / `test.py`。

### 3) 评测：四数据集 + 未见物体/动作/场景/主体 + 野外样例

- **链接：** 论文 §4 / 项目页 Tables
- **摘录要点：** H2O 四类 unseen 设定上相对 pix2pixHD / pixelNeRF / CFLD 全面领先（FID、PSNR、SSIM、LPIPS、PA-MPJPE、CLIPScore）；TACO、Assembly101、Ego-Exo4D 未见动作设定同向；消融显示 **pose+text 同开最优**，缺文本时未见物体语义易错，缺 pose 时手构型偏离；backbone 上 LDM 优于 MAE/MAT；失败仍集中在细手指、重遮挡与 VLM 错描述传播。
- **对 wiki 的映射：**
  - [EgoWorld（论文）](../../wiki/entities/paper-egoworld.md) — 评测表与「结论」可操作读法。

### 4) 局限与开源边界

- **摘录要点：** 依赖深度与 3D 手姿；遮挡/噪声下退化；罕见物体与歧义姿态仍难；隐私与同意式处理需注意。官方仓库 **MIT**，提供 `train.py` / `test.py`、H2O 预处理包与 SD-inpainting / H2O checkpoint 下载；深度估计、exo 手姿估计与 VLM 仍为 off-the-shelf 依赖。
- **对 wiki 的映射：**
  - [EgoWorld（论文）](../../wiki/entities/paper-egoworld.md) — 「开源状态：已开源（训练/测试入口 + 权重）」；链到 [repos](../repos/egoworld.md) 与 [项目页](../sites/egoworld-github-io.md)。

## 当前提炼状态

- [x] 摘要、两阶段方法、主结果与局限已摘录到可维护粒度
- [x] 与 `sources/sites/egoworld-github-io.md`、`sources/repos/egoworld.md` 交叉互指
- [x] 与 EgoWorld-100W 数据集消歧说明已写入

## BibTeX

```bibtex
@inproceedings{park2026egoworld,
  author    = {Park, Junho and Ye, Andrew Sangwoo and Kwon, Taein},
  title     = {EgoWorld: Translating Exocentric View to Egocentric View using Rich Exocentric Observations},
  booktitle = {International Conference on Learning Representations},
  year      = {2026},
}
```
