# KINO（arXiv 归档 — 无独立项目页）

> 来源归档（site / arXiv 直核查）

- **标题：** KINO: A Keyframe Interface for VLM Planning and Whole-Body Control in Humanoid Loco-Manipulation
- **类型：** site（arXiv 作为唯一官方入口）
- **链接：** <https://arxiv.org/abs/2609.18869>
- **PDF：** <https://arxiv.org/pdf/2609.18869>
- **机构：** 苏黎世联邦理工学院（ETH Zürich）
- **作者：** Sitong Chen, Fatemeh Zargarbashi, Jin Cheng, Tianxu An, Stelian Coros
- **入库日期：** 2026-09-21（步骤 2.5 再核）
- **一句话说明：** VLM 选 whole-body keyframe → 场景重定向 → keyframe-conditioned RL；G1 真机 loco-manipulation。
- **沉淀到 wiki：** [`wiki/entities/paper-kino.md`](../../wiki/entities/paper-kino.md)

## 开源状态（步骤 2.5，2026-09-21）

- **项目页：** **无** — arXiv 与 PDF 为唯一官方链接；未检索到 `*.github.io` 或 ETH lab 专页。
- **GitHub：** **未发布** — arXiv v1（2026-09-16）无 code/data 链接；第三方同名仓库均无关。
- **论文 PDF：** **已公开**。

## 论文核心摘录（再核）

1. **VLM：** Qwen3.6-27B，单卡 RTX 4090；平均 keyframe 输出 **<0.2s**，重定向 **~5ms**；ROS service 接口。
2. **数据：** 52 AMASS locomotion + 105 OmniRetarget 双手 box + 120 in-house 单手 bucket；场景增强改物体初态/朝向。
3. **低层：** PPO + multi-critic（track / goal / reg）；29-DoF G1 目标关节 + PD；10 步 actor 历史、3 步 critic 特权历史。
4. **路径规划：** Dubins / 切向圆弧，曲率上限 **3 m⁻¹**（~0.33m 最小转弯）；中间 goal 贪心选取（heading ≤45°、位移 ≤2m）。
5. **消融（100 trials）：** uniform 端到端 **44%** vs saliency **92%**；Pick 63%→95%，Place 63%→93%。
6. **局限：** keyframe 库手工构建；规划用第三人称视觉 + 外部 mocap 物体位姿。

## 对 wiki 的映射

- [`wiki/entities/paper-kino.md`](../../wiki/entities/paper-kino.md)
