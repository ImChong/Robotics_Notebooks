# GENMO / GEM（统一人体运动估计与生成）

- **标题**: GENMO: A GENeralist Model for Human MOtion（仓库与后续权重常以 **GEM** 名义发布）
- **论文**: https://arxiv.org/abs/2505.01425 （ICCV 2025 Highlight）
- **机构页**: https://research.nvidia.com/labs/dair/publication/genmo2025/
- **代码**: https://github.com/NVlabs/GENMO
- **类型**: paper / code-release
- **机构**: NVIDIA Research（DAIR 等）
- **收录日期**: 2026-05-07

## 一句话摘要

把人体运动 **估计** 重新表述为带观测约束的 **生成** 问题，用扩散式建模统一多条件（单目视频、2D 关键点、文本、音频、3D 关键帧/SMPL 等）下的运动恢复与合成，使生成先验改善遮挡等困难条件下的估计质量。

## 为何值得保留

- **管线枢纽**：与视频驱动的人形控制（如把生成视频转为 SMPL 轨迹再交给跟踪控制器）直接衔接。
- **方法论**：强调回归与扩散的协同（约束生成），便于和纯回归 MoCap、纯无条件扩散生成对照。
- **开源**：论文配套 GitHub 与机构说明页，便于复现与权重获取。

## 技术要点（来自论文公开描述）

1. **约束运动生成**：输出轨迹需同时满足多种时变条件（关键点、文本片段、音频节拍等），可在不同时间段混合不同模态条件。
2. **估计引导的训练**：利用野外视频与 2D 标注、文本等弱监督，增强生成分布多样性；生成侧先验反过来提升困难帧（遮挡、截断）上的估计鲁棒性。
3. **输出形态**：以 SMPL 族参数化运动为主干接口之一，便于下游重定向或 motion tracking（具体接口以代码与文档为准）。

## 对 Wiki 的映射

- **wiki/methods/genmo.md**：人体运动估计/生成方法页。
- **wiki/methods/exoactor.md**：作为「生成视频 → SMPL 全身轨迹」环节的参照实现选型。
- **wiki/methods/diffusion-motion-generation.md**：扩散范式在运动领域的实例交叉引用。
