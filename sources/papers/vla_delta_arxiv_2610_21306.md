# VLA-Δ：学习物理状态转移的视觉语言动作模型 / Learning Physical State Transition for Vision-Language-Action Models

> 一手论文摘录，不是论文全文镜像。

- **标题：** VLA-Δ: Learning Physical State Transition for Vision-Language-Action Models
- **作者：** Qi Lu、Jiarun Li、Ce Zhang、Miao Pan、Yi Pan、Xu Wang、Wenqi Zhang、Xuhong Zhang
- **机构：** 浙江大学；中南大学（Ce Zhang）
- **原文：** <https://arxiv.org/html/2610.21306v1>；[PDF](https://arxiv.org/pdf/2610.21306)
- **版本：** arXiv:2610.21306v1，2026-10-15
- **项目主页：** <https://zju-omniai.github.io/VLA-Delta/>
- **代码：** <https://github.com/ZJU-OmniAI/VLA-Delta>
- **入库日期：** 2026-11-16
- **核对范围：** 论文 v1 摘要、§1–4（方法、实验和消融、结论）、附录 A–D（数据、八步标注流水线与质量抽查、评测协议和结构化 slot 训练目标）；对照官方项目页及开源 README。
- **通读：** 已完成
- **一句话说明：** 未来状态差异文本、结构化转移槽和光流在预训练期塑造物理状态查询表示，测试时只抽取查询表示并生成连续动作。

## 核心方法（§2）

- 骨干沿用 GR00T N1 设计，使用 Qwen3-VL-2B-Instruct 与九层 DiT action expert；基于 ZR-0 base 初始化视觉语言模块。
- 输入当前多视角图像、指令与机器人 proprioceptive state；Physical State Difference Queries 把视觉语言上下文压为任务相关转移表示 Z。
- **九类 Structured Transition Slots：** 进度、阶段、目标位置、夹爪位置、相对运动、可交互位置、夹爪状态切换、接触转移、障碍风险。
- **Dense Optical Flow：** 同视角当前与后续帧之间的像素位移，补足局部运动细节。
- **Future-State Difference Text：** 任务进度、空间变化、接触与约束的连贯文本；文本解码位置仅可读取 Queries 与之前文本位置，使状态信息必须通过 Query 传递。
- 联合预训练损失含 text + slots + optical flow + flow-matching action；下游微调只使用 action 目标；部署不解码中间监督标签，仍保留 Queries。

## 数据与自动标注（§2.3、附录 B）

- DROID、MolmoAct、RH20T 共约 **484 小时、2400 万帧**的示教数据。
- 8 个步骤：样本筛选、LeRobot v3 格式统一、数据清洗、Qwen3.8-27B 阶段/关键帧与语义、SAM 3/UniDepth V2/MegaFlow 几何运动标注、当前–未来对齐生成样本、规则与模型审查及人工审核、标注审计统计。
- 论文按三来源各抽 100 对，审核总数 300 个当前-未来配对；各字段有效率不可视为像素框 IoU。附录明确目标框、夹爪框参考由 GPT-5.6 生成，并非独立人工真值。
- 未来观测只作为预训练标签，不进入动作推理输入。

## 实验（§3、附录 C–D）

| 项目 | VLA-Δ | 对照与协议 |
|---|---:|---|
| LIBERO 平均成功率 | 98.4% | π0.5 96.9%、ZR-0 97.8%；40 任务 × 50 回合 |
| LIBERO-Plus Language 扰动 | 89.0% | π0.5 82.6%；四扰动共计 8000 回合 |
| RoboTwin 2.0 Clean | 90.06% | ZR-0 88.70%；50 任务 × 100 回合 |
| RoboTwin 2.0 Random | 89.74% | ZR-0 87.98%；50 任务 × 100 回合 |
| AgileX PiPER 双臂真机 | 68.0% | 5 任务 × 40 回合；论文称比 π0.5 高 5.5 个百分点 |
| LIBERO 消融：Query + action 预训练 | 96.45% | + 文本 97.20%；+ slots 96.90%；+ 光流 97.35%；三者齐备 98.40% |

RoboTwin 预测 16 步、执行前 8 步后再观察重规划。真实机器人是三相机、双臂 PiPER；受控五任务不能证明复杂未知场景的稳定性。

## 局限与开源验证

- 指标是作者报告，**未独立复现实测**；消融表支持三类监督的联合收益，不代表模型掌握了所有接触动力学。
- 自动标注的深度、光流、位置和语言描述可能有误；抽样通过不代表逐帧真实标签全对。
- 官方仓库提供三阶段训练、LIBERO/RoboTwin 微调与仿真评测入口，以及公开 base、LIBERO、RoboTwin 权重。尚未核实完整 484h 物理差分标注集及从原始视频复做全部标注的端到端管线公开。训练准备脚本依赖已有标注。

## 对 wiki 的映射

- [VLA-Δ 论文独立节点](../../wiki/entities/paper-vla-delta.md)
- [项目主页归档](../sites/vla-delta.md)、[源码运行入口归档](../repos/vla-delta.md)
- [VLA 纵深路线 Stage 2](../../roadmap/depth-vla.md)
