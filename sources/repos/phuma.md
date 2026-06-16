# PHUMA（Physically Reliable Humanoid Locomotion Dataset）

- **标题**: PHUMA
- **类型**: dataset / repo
- **仓库**: https://github.com/davian-robotics/PHUMA
- **项目页**: https://davian-robotics.github.io/PHUMA/
- **论文**: Lee et al., *PHUMA: Physically Reliable Humanoid Locomotion Dataset*, arXiv:[2510.26236](https://arxiv.org/abs/2510.26236) (2025)
- **数据**: Hugging Face [DAVIAN-Robotics/PHUMA](https://huggingface.co/datasets/DAVIAN-Robotics/PHUMA)；`bash download_phuma.sh` 一键获取 **G1 / H1-2 预重定向** 轨迹
- **机构**: DAVIAN Robotics, KAIST AI
- **收录日期**: 2026-06-16

## 一句话摘要

面向 **人形 locomotion imitation** 的 **物理可信、已重定向** 数据集：两阶段管线（physics-aware curation + **PhySINK** 物理约束重定向）把 Motion-X / Humanoid-X 等大规模人体动作清洗并映射到 **Unitree G1** 与 **H1-2**，公开约 **76k clips · 73 h**；G1 轨迹可直接用于 MaskedMimic / ProtoMotions 等 tracking 训练。

## 为何值得保留

- **「已重定向好」的 G1 数据**：用户明确提到宇树相关、重定向已完成——与 AMASS（仅 SMPL 人体）形成鲜明对照，降低从零跑 GMR/PhySINK 的工程摩擦。
- **物理可信策展**：针对 floating、穿透、脚滑等互联网视频/大规模 MoCap 常见伪影做过滤；默认阈值保留跳跃等腾空相位，可按 `--foot_contact_threshold` 收紧。
- **生态接入**：NVIDIA [ProtoMotions](https://github.com/NVlabs/ProtoMotions) 已原生支持 PHUMA 数据准备；LIMMT、Humanoid-GPT 等将其与 AMASS 并列评测。

## 管线要点（编译自 README / 项目页）

1. **Curation**：SMPL-X → (N,69) 体姿；脚接触、质心、jerk 等物理感知过滤。
2. **PhySINK Retargeting**：形状适配（`betas.npy` 一次性）+ 运动适配（`root_trans` / `root_ori` / `dof_pos`）；相对 GMR 线性身高缩放，对高矮被试伪影更稳。
3. **输出格式**：`data/humanoid_pose/<robot>/` 下 `.npy` 字典；官方 split：`phuma_train.txt` / `phuma_test.txt` / `unseen_video.txt`。
4. **真机**：论文报告 **Unitree G1** 零样本 sim2real tracking 优于 AMASS 同管线。

## 对 Wiki 的映射

- **wiki/entities/dataset-bfm-phuma.md**：升格为完整 PHUMA 实体归纳（保留 BFM 索引互链）。
- **wiki/comparisons/humanoid-reference-motion-datasets.md**：与 AMASS / LAFAN1 / OMOMO / Humanoid Everyday 对照。
- **wiki/entities/unitree-g1.md**、**wiki/entities/protomotions.md**：G1 预重定向数据与训练入口互链。
