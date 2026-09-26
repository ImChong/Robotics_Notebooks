# bear-ty.github.io/Beyondretarget_page（BeyondRetarget 项目页）

- **标题：** BeyondRetarget: Learning Executable Humanoid Motions Directly from Monocular Video
- **类型：** site / project-page
- **URL：** <https://bear-ty.github.io/Beyondretarget_page/>
- **配套论文：** [BeyondRetarget（arXiv:2609.29850）](https://arxiv.org/abs/2609.29850) — 归档见 [`sources/papers/beyondretarget_arxiv_2609_29850.md`](../papers/beyondretarget_arxiv_2609_29850.md)
- **代码：** <https://github.com/bear-ty/BeyondRetarget> — 归档见 [`sources/repos/beyondretarget.md`](../repos/beyondretarget.md)
- **Demo：** <https://huggingface.co/spaces/bear-ty/BeyondRetarget>（ZeroGPU 社区空间；页内在线 demo 为 **base 版**）
- **入库日期：** 2026-09-26

## 一句话摘要

南京大学等提出的 **端到端单目视频→多机人形可执行 motion** 官方页：对比两阶段 GVHMR/WHAM→GMR/NMR 的 **collapse 与延迟**，展示 **八款人形**、仿真 SR 与 **~192 ms 流式推理**；声明当前开源与 demo 均为 **base 版**（固定相机、无大尺度全局轨迹），**performance 版** 计划后续发布。

## 公开信息要点（截至入库日）

- **机构：** 南大电子学院 / 智能学院、江苏移动、中国移动紫金创新研究院。
- **卖点：** End-to-end · Multi-robot · Contact-aware · Streaming-ready · Real-time visual teleoperation（单目 1080p@30Hz，背景 mocap **未启用**）。
- **Method 区块：** 两阶段 vs 端到端对比图；HMR2 特征 + TCAM + robot decoder + contact refine。
- **表格：** 与 GVHMR/WHAM→NMR/GMR、GT→NMR/GMR 的 RAMPJPE / SR / foot slide；八机 RAMPJPE；流式 latency/VRAM/throughput。
- **Notice：** base 版不支持 moving camera 与大范围 global human trajectory。

## 为何值得保留

- **非 PDF 证据：** 双手抱腰/背后/大画圆/深蹲等 **motion collapse** 视频对比是理解「为何跳过 SMPL 中间态」的关键。
- **开源三角互证：** 页链 GitHub + HF Demo；README 安装与 `infer_video.py` 与页上 latency 表一致。

## 关联资料

- 论文归档：[`sources/papers/beyondretarget_arxiv_2609_29850.md`](../papers/beyondretarget_arxiv_2609_29850.md)
- 代码仓库：[`sources/repos/beyondretarget.md`](../repos/beyondretarget.md)
