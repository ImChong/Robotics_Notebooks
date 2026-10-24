# PhysibleMDM（arXiv:2610.30500）

> 来源归档（paper）

- **标题：** PhysibleMDM: Enforcing Inference-Time Physical Feasibility Constraints on Motion Diffusion Models via Noise Optimization
- **类型：** paper
- **作者：** Maximilian Stölzle, David Müller, Agon Serifi, Ruben Grandia, Espen Knoop, Moritz Bächer
- **单位：** Disney Research；arXiv 页面另列 Maximilian Stölzle 与 MIT CSAIL
- **arXiv：** <https://arxiv.org/abs/2610.30500>
- **HTML：** <https://arxiv.org/html/2610.30500>
- **PDF：** <https://arxiv.org/pdf/2610.30500>
- **项目页：** 截至 2026-10-24，arXiv 页面未列独立项目页
- **代码：** 截至 2026-10-24，未找到作者提供的代码仓库链接
- **提交日期：** 2026-10-21（v1）
- **入库日期：** 2026-10-24
- **一句话说明：** 不重新训练动作扩散生成器，而在推理时搜索初始及后验扩散噪声，使生成动作满足地面穿透、脚滑、自碰撞等运动学约束，并通过约束逆动力学评估双足机器人的动态可行性。

## 核心摘录（据论文原文整理）

- **问题：** Motion Diffusion Model（MDM）生成的动作可能视觉自然，却存在穿地、脚滑、自碰撞、超出关节扭矩能力或接触动力学不一致等问题，难以直接用于仿真与控制。
- **方法：** 冻结生成模型，在推理时联合参数化初始噪声与后验噪声；用 Sep-CMA-ES 对噪声做无梯度搜索。候选动作由轻量运动学约束和以内层锥规划求解的逆动力学约束打分，无需每个候选都跑完整轨迹仿真。
- **长序列：** 将优化后的动作片段自回归串接，沿用已提交片段作为后续历史条件。
- **SMPL 结果：** 在 100 种动作风格上，相对未优化生成，平均地面穿透、自碰撞穿透和地面滑动分别下降 75%、92%、74%。
- **双足机器人结果：** 在 11 段自定义双足机器人动作上，平均加速度重建误差的逐序列中位降幅为 89.5%，平均逆动力学残差的中位降幅为 79.5%。
- **边界：** 评测集中于双足形体，并假设地面接触为粘着接触；反复评估与逆动力学求解可能不适合实时控制。优化噪声也可能偏离生成模型训练分布，造成动作风格退化。
- **开源核查：** 本次核查的 arXiv 摘要与 HTML 未列项目页或代码链接，未将第三方论文索引页当作作者实现入口。

## 对 wiki 的映射

- [PhysibleMDM 论文实体页](../../wiki/entities/paper-physiblemdm.md)
- [PhysDiff 论文实体页](../../wiki/entities/paper-notebook-physdiff-physics-guided-human-motion-diffusion-m.md)

## 参考来源（原始）

- 论文摘要与元数据：<https://arxiv.org/abs/2610.30500>
- 论文全文：<https://arxiv.org/html/2610.30500>
- PDF：<https://arxiv.org/pdf/2610.30500>
