# ECHO：用于人形全身控制的上下文行为基础模型

> ECHO: In-Context Behavior foundation model for Humanoid Whole-Body Control（arXiv:2611.25778v1）

- **类型：** paper / humanoid / behavior foundation model / whole-body tracking / in-context adaptation
- **英文标题：** ECHO: In-Context Behavior foundation model for Humanoid Whole-Body Control
- **中文标题：** ECHO：用于人形全身控制的上下文行为基础模型
- **论文：** https://arxiv.org/abs/2611.25778
- **HTML 全文：** https://arxiv.org/html/2611.25778v1
- **PDF：** https://arxiv.org/pdf/2611.25778
- **版本：** arXiv v1，2026-11-19
- **项目：** https://mwondering.github.io/echo-bfm/
- **网站源码：** https://github.com/mwondering/echo-bfm （不是策略训练/推理代码）
- **机构：** 浙江大学、中国电信人工智能研究院（TeleAI）、上海科技大学、上海交通大学、Gamma Robotics
- **作者：** Xingyi Wang、Bin Li、Jiyuan Shi、Chenyun Zhang、Weiji Xie、Xiangyu Miao、Junsong Wu、Wei Song、Shiqiang Zhu、Chenjia Bai
- **平台：** Unitree G1；mjlab + PPO 训练、MuJoCo CPU 仿真评测
- **通读：** 已完成
- **核对范围：** 核对 arXiv:2611.25778v1 HTML 第 1–5 节正文以及第 6–11 节方法、训练、评测与限制附录（公式、表 2–9），对照作者官方项目页 Method / Results / Demo 及其 GitHub 仓库 README、目录；未独立复现模型。
- **入库日期：** 2026-11-23

## 核心摘录

### 论文在解决什么

同一个全身参考运动，面对不同负载、摩擦和执行器动力学，需要不同的关节控制响应。通用策略虽有广覆盖运动能力，仍可能在多种物性间取保守折中。ECHO 将「会做什么」与「当前物性下如何做准」分开。

### 三阶段方法（`3、附录 `6）

1. **MIAS / Mixed and Annealed Sampling**：在同一预训练期对失败片段回退、均匀选段和按失败率采样进行混合；退火放宽覆盖以防只记住困难动作。附录 Eq.16：失败后 rewind / uniform / adaptive 各占 1/3；正常完成后 uniform / adaptive 各占 1/2。
2. **CRA 的物理上下文表征**：冻结 ECHO-PRE rollout 后，以短期状态—动作历史加同一物理 session 的长期记忆喂给编码器。多步未来状态/接触预测和跨运动 physical-neighborhood soft alignment 联合训练。环境真值用于训练监督，推理只读交互历史。
3. **残差适配**：冻结先验和上下文编码器，训练残差 actor，将 `a_t = a_t^θ + a_t^Δ` 作为最终控制指令。无需测试时参数更新；预测器只在上下文训练期使用，不在部署端执行。

### 实验结果（原文表 2、3、4、8、9）

| 评测 | 数值与解释 |
|------|------------|
| AMASS-2000-Random（训练未见） | ECHO-PRE SR 0.9832、ECHO-N 0.9827；CRA 提升误差精度但不能声称每个 SR 都提升 |
| BONES-SEED-100-HighDynamic（训练分布内） | SR：去 MIAS 0.4500 → MIAS 先验 0.7200 → ECHO-N 0.7633 |
| HDR 物性变化 | ECHO-H SR 0.8922、RMA-Teacher 0.8827、RMA-Student 0.8700；global MPJPE 分别 210.351、216.826、224.568 mm |
| LAFAN-40 表征 | 环境识别 97.478%、跨运动 mAP 93.872%；50→450 步历史，识别 91.967%→98.403% |
| G1 真机 | 四种参考动作，每 checkpoint 每动作 10 次；ECHO-H 在每小腿附加 2 kg 时，四项 MPJPE-L 均优于 Vanilla 和 RMA-Student |

训练动作库过滤后共 **220,560** 条、**222.46M** 帧、约 **1,235.88 h**（MotionDecode 953.17 h、BONES-SEED 260.85 h、100STYLE 18.63 h、LAFAN1 2.52 h）。HDR 在 NDR 基础上额外采样四肢负载，每手 0–2.5 kg、每小腿 0–4 kg，总计最多 13 kg。真机演示中有四肢合计 10 kg 负载，与表 9 的双小腿 2+2 kg 评测条件不同。

### 局限与开源核查

- 上下文训练依赖仿真 rollout 和特权监督，超出随机化范围的物性仍需评测。
- 长期记忆假设每个物理 session 内参数固定；session 中途突变后历史可能过时。
- 真机以 G1 的参考条件全身跟踪为主，跨本体与自主任务规划未验证。
- **2026-11-23 核查**：项目页面 GitHub 入口是论文展示站 `mwondering/echo-bfm`，没有可运行的 ECHO-PRE/CRA 策略训练、推理、真机部署源码或公开模型 checkpoint；不能按网站构建脚本编造策略运行时序图。

## 对 wiki 的映射

- [ECHO 独立论文实体](../../wiki/entities/paper-echo-bfm.md)
- [BFM 概念](../../wiki/concepts/behavior-foundation-model.md)
- [BFM 纵深路线](../../roadmap/depth-bfm.md)
- [项目页开放核查](../sites/echo-bfm-project.md)
- [网站仓库性质核查](../repos/echo-bfm-website.md)

## 参考原文

- [ECHO 论文 v1（HTML）](https://arxiv.org/html/2611.25778v1)
- [ECHO 项目主页](https://mwondering.github.io/echo-bfm/)
