# mint-project-page（MINT 官方项目页）

- **标题：** MINT — World-Space Camera and Hand Motion Estimation
- **类型：** site / project-page
- **URL：** <https://1847540790.github.io/mint-project-page/>
- **配套论文：** [arXiv:2609.04958](https://arxiv.org/abs/2609.04958v1) — [`sources/papers/mint_arxiv_2609_04958.md`](../papers/mint_arxiv_2609_04958.md)
- **代码：** <https://github.com/wuji-technology/wuji-ego-mint> — [`sources/repos/wuji-ego-mint.md`](../repos/wuji-ego-mint.md)
- **模型：** <https://huggingface.co/ZZJAsher/mint_v1>
- **数据集：** <https://huggingface.co/datasets/ZZJAsher/wuji_ego_mint>
- **入库 / 复核日期：** 2026-09-07

## 一句话摘要

舞肌科技 / 上海科大等联合发布的 MINT 官方站点：展示挑战性 egocentric 样例、EgoPipeline 阶段图、模型架构、**1,021 h** 数据集组成，以及 HOT3D / ARCTIC 上相机系双手与世界系相机轨迹完整 benchmark 表。

## 公开信息要点（截至复核日）

- **页首资源链：** Paper · GitHub · Model weights · Dataset 四按钮齐全。
- **挑战性视频：** 常见动作、快速相机运动、近距离手部交互、低照度、双手协同、运动模糊；含与 RGB / EgoPipeline / MINT 同帧对比。
- **EgoPipeline：** GeoCalib → MoGe-2 → MegaSaM → HaWoR → 后处理；从公开 egocentric 视频生成世界系监督。
- **模型规模：** **1,021 h** 监督、**1.139 B** 可训练参数、**4** 预测头（camera · FOV · MANO · presence）、**32** 帧训练窗。
- **定量（节选）：**
  - HOT3D 相机系：MINT **FAcc 0.940**、**PA-MPJPE-p 10.70**、**Jitter 11.52**（+UKF **2.39**）。
  - HOT3D 世界相机：MINT **ATE 181.7 mm**、**arc-length ratio 1.094**。
- **失败样例：** 面团塑形、极端运动模糊。

## 开源状态（步骤 2.5）

- **已开源：** 项目页列出的 GitHub、HF 模型与数据集在复核日均可访问；仓库含 Web Viewer 与训练入口。
- **边界：** MANO 与部分 HaWoR 适配件不可再分发；数据集相机轨迹为 scale-enlarged 预训练版。

## 为何值得保留

- 步骤 2.5 主入口：核对四链开放程度与 benchmark 数字。
- 可视化样例与失败案例补 PDF 叙述。

## 关联资料

- 论文：[`sources/papers/mint_arxiv_2609_04958.md`](../papers/mint_arxiv_2609_04958.md)
- 代码：[`sources/repos/wuji-ego-mint.md`](../repos/wuji-ego-mint.md)
- Wiki：[wiki/entities/paper-mint-ego-world-space-camera-hand-motion.md](../../wiki/entities/paper-mint-ego-world-space-camera-hand-motion.md)
