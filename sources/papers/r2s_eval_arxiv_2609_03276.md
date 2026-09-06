# R2S-Eval: Robot Evaluation with Real-to-Sim Calibration via Vision-Language Models（arXiv:2609.03276）

> 来源归档（ingest）

- **标题：** R2S-Eval: Robot Evaluation with Real-to-Sim Calibration via Vision-Language Models
- **简称：** R2S-Eval
- **类型：** paper / robot-evaluation / real-to-sim / vlm / vla
- **arXiv：** <https://arxiv.org/abs/2609.03276>
- **项目页：** <https://r2s-eval.github.io/> — 归档见 [`sources/sites/r2s-eval.md`](../sites/r2s-eval.md)
- **作者：** Yidi Wang、Feixiang Ruan、Ruoqu Chen、Jie Yin、Yang Yu、Mengdi Xu、Kaifeng Zhang 等
- **机构：** 南京大学（NJU）；同济大学（Tongji）；清华大学（Tsinghua）；Sharpa 等
- **入库日期：** 2026-09-06
- **索引来源：** [具身智能小站 9 篇资源汇总](../blogs/wechat_embodied_station_9_papers_resources_2026-09-06.md)
- **一句话说明：** 真实环境校准到仿真生成 rollout 视频，VLM 成对偏好 + Bradley–Terry 聚合策略排名；减少重复硬件试验并揭示二元成功标签遗漏的行为质量差异。

## 开源状态（步骤 2.5，2026-09-06）

| 组件 | 状态 |
|------|------|
| 项目页 | 已上线（方法、7 任务 real/sim 对比、VLM 评测） |
| 官方 GitHub | **未见** 独立实现仓 |
| 依赖栈 | 论文引用 **Isaac Lab-Arena**、评测 VLA（π₀/π₀.₅、OpenVLA 等） |

**结论：待发布** — 可复现管线依赖 Isaac Lab-Arena 与第三方 VLA 仓，非 R2S-Eval 官方一体包。

## 核心摘录

### 摘录 1：评测范式

- 将策略评测从「数成功次数」转为 **rollout 行为偏好估计**。
- Real-to-sim 校准：机器人几何/运动学/控制接口/物体/相机/初始化对齐真机，无需照片级数字孪生。
- VLM 描述进度、连续性、控制质量与完成度，成对比较后 Bradley–Terry 排名。

**对 wiki 的映射：** [paper-r2s-eval](../../wiki/entities/paper-r2s-eval.md)

### 摘录 2：实验规模

- **LIBERO** 40 任务 × 6 VLA × 8 VLM judges。
- **7 个真机桌面任务**（盖杯、捡球入盒等）校准仿真对比。
- 真机设置：偏好排序与人类标注 **91.9%** 一致；Spearman 与成功率排序 **0.957**。

**对 wiki 的映射：** [paper-r2s-eval](../../wiki/entities/paper-r2s-eval.md)

## 当前提炼状态

- [x] 项目页核查（2026-09-06）
- [x] wiki 映射：`wiki/entities/paper-r2s-eval.md`
