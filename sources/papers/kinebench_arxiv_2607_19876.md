# KineBench: Benchmarking Embodied World Models via IDM-Free Kinematic Grounding（arXiv:2607.19876）

> 来源归档（ingest）

- **标题：** KineBench: Benchmarking Embodied World Models via IDM-Free Kinematic Grounding
- **类型：** paper / embodied world model / benchmark / IDM-free / ManiSkill3 / kinematic grounding
- **arXiv：** <https://arxiv.org/abs/2607.19876>（PDF：<https://arxiv.org/pdf/2607.19876.pdf>）
- **会场：** ECCV 2026（Accept）
- **作者：** Zeyu Liu、Zhangzhe Zhu、Yang Zhang、Chenyou Fan、Chenjia Bai、Xuelong Li
- **机构：** 中国电信人工智能研究院（TeleAI）；新加坡国立大学（NUS）；复旦大学；清华大学；西北工业大学深圳研究院
- **代码：** <https://github.com/minecraft-zzz/KineBench>（**MIT**）
- **入库日期：** 2026-07-27
- **一句话说明：** 用级联视觉基础模型从生成视频逐帧抽 **6D 末端位姿**，在 **ManiSkill3** 闭环执行，避开脆弱 IDM；指标含任务成功、**SPARC** 平滑度、**Maruyama Manipulability**；20 任务 × 四套件评测具身视频 WM。

## 开源状态（核查，2026-07-27）

- **已开源：** 官方仓 [minecraft-zzz/KineBench](https://github.com/minecraft-zzz/KineBench) · **MIT License**（`LICENSE`）。
- **可运行入口：** `kinebench/` 包（perception / planning / eval / generation）、`configs/eval/*.yaml`、`scripts/run_eval.py`、`scripts/prepare_third_party.py`；感知依赖 FoundationPose / MoGe / YOLO，规划依赖 **pyroki**；仿真路径为 ManiSkill3。
- **边界：** 顶层 README 极简；完整评测需自备 CAD / YOLO / MoGe checkpoint 与 third_party；`local_smoke.yaml` 可用合成 `local_video.npy` 通管道。

## 摘要级要点

- **问题：** 闭环评测几乎都靠 IDM 抽动作 → OOD 视频上 IDM 失败与 WM 失败 **归因混淆**。
- **管线：** YOLO 末端掩码 → MoGeV2 度量深度 → FoundationPose 6D 位姿 → pyroki / EE pose 控制 → ManiSkill3 rollout。
- **指标：** 执行成功率；**SPARC**（频谱弧长平滑度）；**Maruyama Manipulability Index**（运动学可行性）。
- **套件：** Suite0 基本执行 / Suite1 任务迁移 / Suite2 视觉 OOD / Suite3 复杂度条件缩放；20 个 ManiSkill3 操纵任务。
- **发现：** 接触丰富任务（StackCube、PickFruits）掉点严重；Wan 2.2 在 OpenBoxHard 未见资产上 **60%→30%**；任务复杂度约束下数据/算力边际收益非线性。

## 核心论文摘录（MVP）

### 1) IDM-free 运动学落地

- **链接：** §1；§3.1
- **摘录要点：** 显式 6D EEF 接地直接测物理可执行性，并对 gripper 消失等幻觉敏感，同时吸收高频像素抖动。
- **对 wiki 的映射：**
  - [KineBench](../../wiki/entities/paper-kinebench.md)
  - [Masked Visual Actions](../../wiki/entities/paper-masked-visual-actions.md) — 对照仍用 IDM 的逆向抽动作

### 2) 机器人中心运动学指标

- **链接：** §3.2
- **摘录要点：** SPARC + Maruyama 与执行成功呈任务/模型依赖关联，提供互补诊断。
- **对 wiki 的映射：**
  - [KineBench](../../wiki/entities/paper-kinebench.md)
  - [运动学 vs 动力学可行](../../wiki/concepts/kinematic-vs-dynamic-feasibility.md)

### 3) 四套件与缩放

- **链接：** §3.3；§4
- **摘录要点：** 视觉 OOD 与接触动力学仍是瓶颈；复杂度升高后 scaling 变非线性。
- **对 wiki 的映射：**
  - [物理保真度输出轴](../../wiki/overview/world-model-physics-fidelity-outputs.md)
  - [EWMBench](../../wiki/entities/ewmbench.md)

## BibTeX

```bibtex
@inproceedings{liu2026kinebench,
  title     = {KineBench: Benchmarking Embodied World Models via IDM-Free Kinematic Grounding},
  author    = {Liu, Zeyu and Zhu, Zhangzhe and Zhang, Yang and Fan, Chenyou and
               Bai, Chenjia and Li, Xuelong},
  booktitle = {ECCV},
  year      = {2026},
  note      = {arXiv:2607.19876}
}
```

## 对 wiki 的映射

- 主实体页：[`wiki/entities/paper-kinebench.md`](../../wiki/entities/paper-kinebench.md)
- 代码归档：[`sources/repos/kinebench.md`](../repos/kinebench.md)
- 互链：[EWMBench](../../wiki/entities/ewmbench.md)、[物理保真度输出轴](../../wiki/overview/world-model-physics-fidelity-outputs.md)、[Imagined Rollouts…](../../wiki/entities/paper-imagined-rollouts-kinematic-not-dynamic.md)、[Generative World Models](../../wiki/methods/generative-world-models.md)
- 策展入口：[微信 · 世界模型物理保真度](../blogs/wechat_embodied_ai_lab_world_model_physics_fidelity.md)
