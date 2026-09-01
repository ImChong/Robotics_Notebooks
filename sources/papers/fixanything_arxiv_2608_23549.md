# FixAnything: 3D-Consistent Rendering Refinement via Video Generative Priors

> 来源归档（ingest）

- **标题：** FixAnything: 3D-Consistent Rendering Refinement via Video Generative Priors
- **短名：** FixAnything
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2608.23549>
- **PDF：** <https://arxiv.org/pdf/2608.23549>
- **项目页：** <https://fix-anything.github.io/>
- **代码：** <https://github.com/kvuong2711/fix-anything>
- **权重：** <https://huggingface.co/kvuong2711/fix-anything>
- **机构：** 卡内基梅隆大学（CMU）
- **会议：** ECCV 2026
- **入库日期：** 2026-09-01
- **一句话说明：** 单一通用视频模型，用 Wan2.1-I2V-14B + rank-64 LoRA 修复 3DGS / NeRF / mesh / 稀疏点云渲染伪影；mask 锚定干净训练视角；Flow-DPO 以 COLMAP 位姿 AUC@5° 作奖励提升几何一致性。

## 开源状态（步骤 2.5）

- **已开源**：推理代码 + Apache-2.0 权重（HF `kvuong2711/fix-anything`）；`scripts/run_inference.py` / `run_mapanything.py` / `download_models.py` 可跑通；**未发布** SFT / Flow-DPO 训练脚本。

## 核心摘录（面向 wiki 编译）

### 摘录 1：表示无关的渲染清理 + 轻量适配

- 将退化渲染序列视为 **video-to-video translation**：退化视频 latent 与噪声 latent、逐帧二值 mask 在通道维拼接，冻结 Wan2.1-I2V-14B VAE 与骨干，仅训 **rank-64 LoRA**（<1% 参数）。
- 同一模型联合处理 **NeRF / 3DGS / mesh / 稀疏 COLMAP 点云** 四类渲染输入；稀疏点云渲染即可暴露相机轨迹，无需 Plücker 等显式相机坐标教学。
- 数据：DL3DV-10K 每场景 3–12 训练视角、61 帧轨迹；**20 条配对视频** 已有效，500 条用于主实验。

**对 wiki 的映射：** [paper-fixanything](../../wiki/entities/paper-fixanything.md)、[Wan 视频基础模型](../../wiki/entities/paper-wan-video.md)

### 摘录 2：Mask 条件 + Flow-DPO 几何偏好

- **Mask-aware conditioning**：`m_i=1` 标记训练视角（信任/锚定），`m_i=0` 为待修复帧；无 mask 时 PSNR **降 1.3 dB**（17.65 vs 16.37，6-view）。
- **Flow-DPO**：每场景 5 个随机种子 rollout，以 COLMAP 恢复位姿的 **AUC@5°** 排序构造偏好对（gap≥0.2）；SFT → +DPO 使 AUC@5° **61.12→68.32**（+7.2%），推理无额外代价。

**对 wiki 的映射：** [paper-fixanything](../../wiki/entities/paper-fixanything.md)

### 摘录 3：DL3DV 主表（6 training views，3DGS 输入）

- FixAnything（3DGS 输入）：PSNR **17.65** / SSIM **0.561** / LPIPS **0.289**。
- 对比 post-hoc：3DGS-Enhancer **16.94** / 0.565 / 0.356；Xu et al. **17.35** / 0.566 / 0.396；Difix3D+ **14.41** / 0.424 / 0.400。
- 同模型换 **mesh** 输入：PSNR **17.95** / SSIM **0.583** / LPIPS **0.269**；**sparse SfM points**：**17.74** / 0.568 / 0.271。

**对 wiki 的映射：** [paper-fixanything](../../wiki/entities/paper-fixanything.md)、[R2S-EGO](../../wiki/entities/paper-r2s-ego.md)（同轴 Difix3D+ 基线语境）

## 对 wiki 的映射

- 升格 [`wiki/entities/paper-fixanything.md`](../../wiki/entities/paper-fixanything.md)

## 当前提炼状态

- [x] 方法要点与开源核查
- [x] wiki 实体与交叉引用
