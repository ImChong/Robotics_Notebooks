# Accelerating VLA Inference with Perturbation Informed Execution（PICES，arXiv:2610.19656）

> 来源归档（ingest）

- **标题：** Accelerating VLA Inference with Perturbation Informed Execution
- **缩写 / 框架：** **PICES**（Perturbation-Informed Chunk and Execution Horizon Selection）
- **类型：** paper / vla / inference-acceleration / action-chunking / flow-matching
- **arXiv：** <https://arxiv.org/abs/2610.19656>（Submitted 2026-10-14；PDF：<https://arxiv.org/pdf/2610.19656>）
- **代码（论文声明）：** <https://anonymous.4open.science/r/PICES_VLA-B1BF>（**匿名仓库**，论文 Reproducibility 节与摘要同时给出）
- **核查结果（2026-10-15）：** 通过 `curl` 与 WebFetch 访问该链接均返回 **401**（`{"error":"not_connected"}`），仓库内容、许可证与可运行入口**未核实**。
- **项目页：** 未检索到独立项目页。
- **作者：** Jehun Lee、Hajun Lee、Jimin Ryu、Youhan Lee、Beomseok Kang、Jonghyun Choi、Jae-Joon Kim
- **机构：** Seoul National University（SNU）
- **入库日期：** 2026-10-15
- **文章中的转述：** 见 [微信图文归档](../raw/wechat_embodied_station_robot_31_papers_2026-10-15.md)（第 14 篇，标注「代码（anonymous 4open）」）
- **一句话说明：** 训练免费方法：单次策略调用内同时生成名义动作块与局部扰动样本，据其逐步敏感度联合选择动作块和执行长度，以减少策略调用次数。

## 开源状态（步骤 2.5，核查日期 2026-10-15）

- 唯一代码链接为匿名 4open 仓库，本次访问返回 401，无法核对内容。
- 论文承诺代码“available”，但未给出可公开核验的 GitHub / 权重地址。
- **结论：未核实**（论文声明匿名代码；待作者公开后复核）。

## 摘录 1：问题与主张（摘要 / §1）

- **痛点：** 动作分块 VLA 采用固定执行长度时，短 horizon 需要更多策略调用；长 horizon 摊薄推理开销，却会减少episode内的精细修正机会。
- **主张：** 在统一的一次调用中，比较名义 chunk 与扰动 counterpart 的对应动作差异，为每个候选确定执行长度并选出名义 chunk。

## 摘录 2：结果（摘要 / §4）

- **摘要数字：** 在 Jetson Orin 与 Thor 上，相对固定 horizon 基线，策略推理的 wall-clock 时间最高加速 **2.3×**，同时成功率提升。
- **实验设定（正文）：** 仿真包括 SimplerEnv–Fractal（六个 Google Robot 任务，每任务 100 episodes）与 LIBERO（四个 suite，每 suite 10 任务、每任务 50 episodes）；另有真实机械臂三任务。
- **对比：** AAC、AutoHorizon、A3 等方法及其 sample-matched 变体。

**对 wiki 的映射：** 升格 [`wiki/entities/paper-pices.md`](../../wiki/entities/paper-pices.md)；与动作分块、执行 horizon、推理频率解耦页互链。
