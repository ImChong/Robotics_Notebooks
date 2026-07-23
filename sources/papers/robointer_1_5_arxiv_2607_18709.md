# RoboInter1.5: A Holistic Intermediate Representation Suite for Embodied World Modeling and Robotic Manipulation（arXiv:2607.18709）

> 来源归档（ingest）

- **标题：** RoboInter1.5: A Holistic Intermediate Representation Suite for Embodied World Modeling and Robotic Manipulation
- **类型：** paper / intermediate representation / VLA / VLM / world model / manipulation dataset
- **arXiv：** <https://arxiv.org/abs/2607.18709>（PDF：<https://arxiv.org/pdf/2607.18709.pdf>）
- **前序：** RoboInter1.0（[arXiv:2602.09973](https://arxiv.org/abs/2602.09973)，ICLR 2026；公开仓 README 仍主要引用此编号）
- **项目页：** <https://lihaohn.github.io/RoboInter.github.io/>
- **代码：** <https://github.com/InternRobotics/RoboInter>（MIT）
- **数据：** <https://huggingface.co/datasets/InternRobotics/RoboInter-Data>；VQA：<https://huggingface.co/datasets/InternRobotics/RoboInter-VQA>
- **权重：** <https://huggingface.co/InternRobotics/RoboInter-VLM>（及 Qwen2.5-VL-3B / LLaVA-OV 变体）
- **作者团队：** RoboInter1.5 Team（核心贡献单位见附录：北航 / 中科大 / 上海人工智能实验室）
- **机构：** 北京航空航天大学（BUAA）、中国科学技术大学（USTC）、上海人工智能实验室（Shanghai AI Lab）
- **入库日期：** 2026-07-23
- **一句话说明：** 在 RoboInter1.0 的 **中间表示套件**（Data / Tool / VQA / VLM / VLA）之上，1.5 补齐 **RoboInter-CV** 控制视频基准与 **RoboInter-World**（以中间表示作结构化条件的可控世界模型），把中间表示写成连接 **语义规划、低层动作与长程像素推演** 的双向接口。

## 开源状态（项目页 + 仓库核查，2026-07-23）

- **部分开源：**
  - **已发布：** `RoboInterData/`（转换 / LMDB / LeRobot dataloader）、`RoboInterTools/`（SAM2 半自动标注）、`RoboInterData-Demo/`、`RoboInterVLM/`（Qwen2.5-VL / LLaVA-OV 训练评测）、HF **RoboInter-Data / VQA / VLM**。
  - **待发布：** 根 README TODO：`RoboInterVLA` 模型权重；更多预训练 checkpoint；LeRobot v3.0 全量数据支持。`RoboInterVLA/README.md` 现为 “We will release the code as soon as possible”。
  - **1.5 增量：** 论文新增 **RoboInter-World / RoboInter-CV**；截至入库日公开仓 **无 `RoboInterWorld/` 目录**，World 训练/权重入口以后续更新为准。
- **License：** 仓库 **MIT**；HF Data 可能有 gated 访问表单（以 HF 卡片为准）。

## 摘要级要点

- **动机：** 现有机器人数据缺细粒度中间结构，VLA 与具身世界模型难共享同一套时空脚手架。
- **RoboInter-Data：** **>230k** episode、**571** 场景、**6** 类机械臂、**10+** 类逐帧标注（子任务/技能、分割、物体/夹爪框、affordance、抓取位姿、接触点、运动轨迹等），人机协作校验。
- **下游：** RoboInter-VQA（空间+时间理解/生成）→ RoboInter-VLM Planner；RoboInter-VLA 三种 plan-then-execute（IC-E2E / EC-E2E / Modular + F-CoT）；RoboInter-World 用渲染控制视频 \(u\) 约束未来 latent。
- **1.5 相对 1.0：** 新建 IR 条件长程世界建模基准 RoboInter-CV（约 **65k** clip / **16.9k** episode）；World 变体与「World 指导 VLA」系统评测。

## 核心论文摘录（MVP）

### 1) 中间表示作为双向接口

- **链接：** Abstract；§1
- **摘录要点：** 中间表示不仅可解释，更应同时 **正则化低层动作空间** 与 **约束开放世界物理仿真器的 latent rollout**。
- **对 wiki 的映射：**
  - [RoboInter1.5](../../wiki/entities/paper-robointer-1-5.md) — 总定位。
  - [VLA](../../wiki/methods/vla.md) / [Generative World Models](../../wiki/methods/generative-world-models.md)

### 2) RoboInter-Data / Tool / VQA

- **链接：** §3；Tab. 1；Fig. 2
- **摘录要点：** DROID / RH20T / OXE 等源；RoboInter-Tool 做人机半自动标注；VQA 约 1M 空间生成 + 大量理解条目；相对 LLARVA / ECoT / ShareRobot 等在规模与稠密 IR 覆盖上更完整。
- **对 wiki 的映射：**
  - [RoboInter1.5](../../wiki/entities/paper-robointer-1-5.md) — 数据工程。
  - [`sources/repos/robointer.md`](../repos/robointer.md) — 下载与 dataloader。

### 3) RoboInter-VLA：三种 plan-then-execute

- **链接：** §4.2；Fig. 3
- **摘录要点：** Planner（VLM）+ Executor（Qwen2.5-VL + DiT action head，对齐 InternVLA-M1 / CogACT）；IC-E2E / EC-E2E / Modular；F-CoT 可组合文本与视觉中间表示。
- **对 wiki 的映射：**
  - [InternVLA-A1.5](../../wiki/entities/paper-internvla-a15-unified-vla.md) — 同组织生态对照。
  - [VLA 开源复现景观](../../wiki/overview/vla-open-source-repro-landscape-2025.md)

### 4) RoboInter-World + RoboInter-CV

- **链接：** §3.3；§4.3；Eq. (1)–(2)
- **摘录要点：** 从分割/轨迹渲染黑底控制视频 \(u\)；条件历史 latent、语言、可选动作与 \(u\) 做未来观测去噪；CV 过滤无效控制帧。报告 World 预测可提升 VLA 动作精度。
- **对 wiki 的映射：**
  - [Masked Visual Actions](../../wiki/entities/paper-masked-visual-actions.md) — 同属「结构化视觉条件」世界模型，条件形态不同（掩码实体 vs IR 控制视频）。
  - [video-as-simulation](../../wiki/concepts/video-as-simulation.md)

## BibTeX（1.5；仓内仍常引 1.0）

```bibtex
@article{robointer15_2026,
  title   = {RoboInter1.5: A Holistic Intermediate Representation Suite for Embodied World Modeling and Robotic Manipulation},
  author  = {{Team of RoboInter1.5}},
  journal = {arXiv preprint arXiv:2607.18709},
  year    = {2026}
}
```

1.0（ICLR 2026 / 公开仓默认引用）：

```bibtex
@article{li2026robointer,
  title   = {RoboInter: A Holistic Intermediate Representation Suite Towards Robotic Manipulation},
  author  = {Li, Hao and Wang, Ziqin and Ding, Zi-han and Yang, Shuai and Chen, Yilun and
             Tian, Yang and Hu, Xiaolin and Wang, Tai and Lin, Dahua and Zhao, Feng and
             Liu, Si and Pang, Jiangmiao},
  journal = {arXiv preprint arXiv:2602.09973},
  year    = {2026}
}
```

## 对 wiki 的映射

- 主实体页：[`wiki/entities/paper-robointer-1-5.md`](../../wiki/entities/paper-robointer-1-5.md)
- 代码归档：[`sources/repos/robointer.md`](../repos/robointer.md)
- 项目页：[`sources/sites/lihaohn-robointer-github-io.md`](../sites/lihaohn-robointer-github-io.md)
