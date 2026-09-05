# PointDiT（arXiv:2607.02515）

> 来源归档（ingest）

- **标题：** PointDiT: Pixel-Space Diffusion for Monocular Geometry Estimation
- **类型：** paper / monocular-geometry / point-map / pixel-space-diffusion / icml2026
- **arXiv abs：** <https://arxiv.org/abs/2607.02515>
- **PDF：** <https://arxiv.org/pdf/2607.02515>
- **HTML：** <https://arxiv.org/html/2607.02515>
- **项目页：** <https://haofeixu.github.io/pointdit/> — 归档见 [`sources/sites/pointdit-github-io.md`](../sites/pointdit-github-io.md)
- **代码：** <https://github.com/google-research/pointdit> — 归档见 [`sources/repos/pointdit.md`](../repos/pointdit.md)（**已开源**，Apache-2.0）
- **权重：** <https://huggingface.co/haofeixu/pointdit>（B/L/H × 256/512；不含门控 DINOv3）
- **机构：** 谷歌（Google）；苏黎世联邦理工（ETH Zürich）；图宾根大学 / 图宾根 AI 中心（University of Tübingen）；微软（Microsoft）；可扩展自主智能（KE:SAI）；慕尼黑工业大学（TUM）
- **作者：** Haofei Xu、Rundi Wu、Philipp Henzler、Nikolai Kalischek、Michael Oechsle、Fabian Manhardt、Marc Pollefeys、Andreas Geiger、Federico Tombari、Michael Niemeyer
- **发表 / 上传：** ICML 2026；arXiv 2026-07-02（2607.02515）
- **入库日期：** 2026-09-05
- **一句话说明：** 用 plain ViT 在原始点图空间做 flow matching（x-prediction + DINOv3 条件），去掉 VAE / 混合回归头；单步即可出仿射不变点图，多步再锐化边界。

## 相关资料（策展）

| 类型 | 链接 | 说明 |
|------|------|------|
| 项目页 | [haofeixu.github.io/pointdit](https://haofeixu.github.io/pointdit/) | 3D 对照、1–4 步细化、回归 vs 生成受控实验 |
| 官方仓 | [google-research/pointdit](https://github.com/google-research/pointdit) | `main.py` + `scripts/demo_*.sh` / `eval_*.sh` / `train_stage*.sh` |
| 权重 | [haofeixu/pointdit](https://huggingface.co/haofeixu/pointdit) | 六个 checkpoint（约 25 GB 全量） |

## 开源状态（步骤 2.5，截至 2026-09-05 复核）

- **已开源：** 项目页链到 [google-research/pointdit](https://github.com/google-research/pointdit)（Apache-2.0）与 [Hugging Face 权重](https://huggingface.co/haofeixu/pointdit)。仓内有 `main.py`、`denoiser.py`、`engine.py`、`model.py`、`loss.py`、`dataloader/`、`scripts/`、`DATASETS.md`、`MODELS.md`。
- **门控依赖：** 冻结 DINOv3 权重不随 checkpoint 发布，需向 [facebookresearch/dinov3](https://github.com/facebookresearch/dinov3) 申请后放入 `pretrained/dinov3/`。
- **处理：** wiki 写「已开源可运行」；源码运行时序图对齐 `scripts/demo_l_512.sh` / `eval_*.sh` / `train_stage*.sh`。

## 摘要级要点

- **问题：** 单目点图估计的两条主流各付代价——确定性回归（MoGe 族）在歧义处平均出过平滑几何；潜空间扩散（GeometryCrafter）先被有损 VAE 抹掉高频结构。
- **主张：** 不需要混合架构、复杂损失或点图 tokenizer。plain ViT 直接在原始 \(H{\times}W{\times}3\) 点图上做 flow matching，条件来自冻结 DINOv3。
- **方法：** \(z_t=tx+(1-t)\epsilon\)；网络做 **x-prediction**（预测干净点图再转速度）；logit-normal + 10% 强制 \(t{=}0\)；点图按质心/均距标准化（仿射不变）；天空投影到半径 3 的虚球并降权。
- **结果要点（512²，7 个真实集 3,444 张，4 步除非另注）：**
  - PointDiT-H：Rel\(_p\) **4.40** / \(\delta^p_1\) **98.02** / Rel\(_d\) **2.75** / \(\delta^d_1\) **98.54** / BF1 **10.49**（1 步 72 ms vs GeometryCrafter 1,178 ms）
  - PointDiT-L 4 步 BF1 **10.50**（全表最高边界）
  - 受控同架构：生成式 BF1 **13.92** vs 确定性回归 **10.90**
  - 室外 KITTI / DIODE / ETH3D 仍落后 MoGe / UniDepthV2（合成室外覆盖不足）
- **局限：** 固定分辨率 256/512；输出仿射不变、不是 metric；室外弱；DINOv3 门控。

## 核心摘录（面向 wiki 编译）

### 1) 两条范式各自模糊世界

回归预测分布均值，椅背细杆/透明物被抹平。潜扩散即便只做 VAE encode→decode（不生成）也已丢掉边缘。PointDiT 把扩散直接放在原始点图上。

### 2) x-prediction 对几何是硬门槛

同设定下 v-prediction Rel\(_p\) **35.44**、BF1 **0.46**（崩）；x-prediction Rel\(_p\) **9.29**、BF1 **13.47**。损失仍在速度空间算（v-loss），再加 \(\lambda{=}0.1\) 的相对点损失。

### 3) 单步已经可用，多步只锐化边界

PointDiT-H 1 步已超先前方法的 BF1；2–4 步 Rel/\(\delta_1\) 几乎不动，BF1 从 9.79 → 10.49。全零初始化与随机噪声几乎一样（Table 2）。

### 4) 复现入口

`pip` + 本地 clone DINOv3 → HF 拉 `*-nodinov3-*.pth` → `bash scripts/demo_l_512.sh`（任意 jpg/png，无需内参）。评测 `scripts/eval_{b,l,h}_{256,512}.sh`；训练两阶段 `train_stage1_256_*.sh` → `train_stage2_512_*.sh`。点云不落盘：loader 读 RGB-D + 内参，在线反投影。

## 对 wiki 的映射

- 实体页：[wiki/entities/paper-pointdit.md](../../wiki/entities/paper-pointdit.md)
- 交叉：[单目深度综述](../../wiki/entities/paper-monocular-depth-estimation-survey.md)、[2D→3D 语义提升 Gap](../../wiki/concepts/2d-to-3d-semantic-lifting-gap.md)、[感知栈选型闭环](../../wiki/queries/robot-perception-stack-selection-loop.md)、[Flex-π](../../wiki/entities/paper-flex-pi.md)、[ADM-BA](../../wiki/entities/paper-adm-ba.md)
