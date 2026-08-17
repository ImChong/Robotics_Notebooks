# Reflex: Enabling Fast and Predictive Vision-Language-Action Models for Reaction-Critical Manipulation（arXiv:2608.14379）

> 来源归档（ingest）

- **标题：** Reflex: Enabling Fast and Predictive Vision-Language-Action Models for Reaction-Critical Manipulation
- **缩写 / 框架：** **ReflexVLA** / **ReflexBench**
- **类型：** paper / vla / latency / dynamic-manipulation
- **arXiv：** <https://arxiv.org/abs/2608.14379>（PDF：<https://arxiv.org/pdf/2608.14379>）
- **项目页：** <https://reflexvla.github.io/> — 归档见 [`sources/sites/reflexvla-github-io.md`](../sites/reflexvla-github-io.md)
- **作者：** Yuxuan Chen、Wanruo Zhang、Xiao Li
- **机构：** 上海交通大学（SJTU）
- **入库日期：** 2026-08-17
- **一句话说明：** 为反应关键操纵建延迟感知基准 ReflexBench，并用未来隐特征预测 + 视觉骨干时序融合 + CUDA Graph 做低延迟 1B VLA。

## 开源状态（步骤 2.5）

- **项目页核查（2026-08-17）：** Hero 按钮为 **Code After acceptance**（未给 GitHub URL）；Paper 链 arXiv。
- **结论：** **宣称录用后开源 / 截至入库日无可运行实现。**

## 摘录 1：问题与基准（§I、§III）

- **痛点：** 现有 VLA 榜多为静态操纵，且仿真常在推理时暂停环境，忽略感知–执行延迟。
- **ReflexBench 六任务：** 传送带拣放、接球、打地鼠、斜坡截球、投球、旋转插销。
- **延迟协议：** 仿真与控制解耦；同步（推理期间机器人空转）与异步（执行上一 chunk 同时推下一 chunk）；可用墙钟延迟经 RTF 注入仿真。
- **采数：** 分阶段规划 + 未来轨迹预测；复杂交互再补任务 RL 专家。

**对 wiki 的映射：** 升格 [`wiki/entities/paper-reflexvla.md`](../../wiki/entities/paper-reflexvla.md)；回链 [VLA](../../wiki/methods/vla.md)、[Action Chunking](../../wiki/methods/action-chunking.md)、[实时性↔泛化](../../wiki/concepts/embodied-fm-latency-generalization-tradeoff.md)。

## 摘录 2：方法（§IV）

- **骨干：** VLA-Adapter 族；DINOv2+SigLIP 224、Qwen2.5-0.5B、连续回归动作头；约 **1B**。
- **未来预测：** 冻结 DINOv3 作目标；\(H\) 个 future token 对齐 action chunk；masked cosine loss，\(\lambda_{\mathrm{future}}=0.05\)。可训练目标会崩（SR 36.8→4.9）。
- **时序融合：** 在视觉中间层对同 patch 做因果 MHA，只把当前帧融合特征喂给 LM，不增加语言侧 token。
- **推理加速：** 多视角×多帧 batched 编码 + 整图 CUDA Graph replay。

**对 wiki 的映射：** 强调「预测进表示、历史进视觉、延迟进系统」三件套，而不是再堆参数。

## 摘录 3：评测（§V）

- **协议：** 每任务 200 demo 共训；异步 chunk=8、horizon=2；RTX 5880 Ada；每任务 150 ep × 3 seed。
- **ReflexBench 均值 SR：** ReflexVLA **50.4%** vs VLA-Adapter 30.3%、PUMA 50.2%（4B）、\(\pi_{0.5}\) 36.9%、OpenVLA-OFT 36.0%。
- **LIBERO：** **97.2%**，与骨干持平，说明动态模块未牺牲静态榜。
- **消融（传送带）：** 冻结未来预测 62.8%；中间层 MHA 71.7%；加速后 **73.8% / 65.0 ms**（相对融合后 125 ms）。
- **真机 AgileX Piper：** Conveyor 16/20、PressButtons 22.5、CatchBalls 6.7，优于 SmolVLA / PUMA。
- **局限：** 模块只在微调阶段、未大规模预训练；未试 RTC 等更先进异步。

**对 wiki 的映射：** 用「1B 打平 4B 动态专精、静态榜不掉」写选型读法。

## 建议 wiki 动作

- 新建 **`wiki/entities/paper-reflexvla.md`**、**`sources/sites/reflexvla-github-io.md`**。
- 交叉更新 VLA、Action Chunking、实时性取舍、操作任务页。
