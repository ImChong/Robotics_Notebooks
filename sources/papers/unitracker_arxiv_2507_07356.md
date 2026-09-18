# UniTracker: Learning Universal Whole-Body Motion Tracker for Humanoid Robots（arXiv:2507.07356）

> 来源归档（ingest）

- **标题：** UniTracker: Learning Universal Whole-Body Motion Tracker for Humanoid Robots
- **类型：** paper / humanoid / whole-body-tracking / teacher-student / cvae / sim2real
- **arXiv：** <https://arxiv.org/abs/2507.07356>
- **PDF：** <https://arxiv.org/pdf/2507.07356>
- **HTML：** <https://arxiv.org/html/2507.07356v1>
- **项目页：** <https://yinkangning0124.github.io/Humanoid-UniTracker/>
- **GitHub（项目页镜像）：** <https://github.com/yinkangning0124/Humanoid-UniTracker>（**非**训练代码仓，见开源核查）
- **作者：** Kangning Yin、Weishuai Zeng、Ke Fan、Minyue Dai、Zirui Wang、Qiang Zhang、Zheng Tian、Jingbo Wang、Jiangmiao Pang、Weinan Zhang 等
- **机构：** 上海交通大学（SJTU）；上海人工智能实验室（Shanghai AI Lab）；上海创智学院（Shanghai Innovation Institute）；北京大学（PKU）；浙江大学（ZJU）；复旦大学（Fudan）；香港科技大学（广州）（HKUST-GZ）；上海科技大学（ShanghaiTech）
- **版本：** arXiv:2507.07356（v1，2025-07-10）
- **入库日期：** 2026-09-18
- **一句话说明：** 两阶段全身跟踪：仿真特权 Oracle（PPO）→ 在线蒸馏为带 CVAE 的 deployable 策略；用 full-observation encoder 对齐 partial-observation prior，缓解 MLP+DAgger 在部分观测下的朝向漂移与 OOD 退化；G1 单策略跟 8k+ 动作，并接 MDM 文本生成与 GVHMR 视频估计。

## 开源状态（核查，2026-09-18）

- **训练/部署代码：确认未开源。** 项目页 `index.html` 中 **Code 按钮被 HTML 注释掉**；GitHub 仓仅含 Nerfies 项目页模板（`index.html` + `static/`），README 21 字节占位，**无 IsaacGym 训练脚本、权重或部署入口**。
- **可复现边界：** 论文 §II–III 给出观测空间、奖励分项、CVAE 蒸馏目标与 IsaacGym 8192 并行设定；真机/仿真视频在项目页；**截至入库日无官方 checkpoint**。
- **源码运行时序图：** wiki 实体页标 **不适用**。

> **项目页 vs 论文：** 项目页摘要写「三阶段 + adaptation module」；**arXiv v1 正文为两阶段**（Oracle + CVAE 在线蒸馏）。本归档以 arXiv v1 为准；若后续版本或代码发布 third stage，lint 时再更新。

## 摘要级要点

- **问题：** 通才全身 motion tracking 需在部分观测、传感器噪声与动力学失配下仍保持参考对齐与动作多样性；纯 teacher–student / MLP+DAgger 蒸馏后常 **motion diversity 下降**，全局朝向等属性 **漂移**，OOD 参考退化。
- **数据：** AMASS 过滤交互与短序列 → **11,313** SMPL 动作；H2O 风格 **两阶段 retarget**（16 link 形状优化 + 序列梯度下降）到 G1；训练/评测另经 **PHC** 过滤过激动作以提升真机可部署性。
- **Stage 1 — Oracle：** IsaacGym + PPO，特权状态含刚体位姿/速度、关节量、goal 一帧差分；29 DoF G1 **锁 6 腕关节 → 23D 动作**（PD 目标）；课程学习 + early termination + reference state initialization。
- **Stage 2 — CVAE 蒸馏：** Deploy 本体为 **25 步历史**（关节 pos/vel、根角速度、重力、前动作）+ 稀疏 goal（参考高度、根朝向/速度差、相对根位移等）；CVAE 用 **full-obs encoder ε** 与 **partial-obs prior ρ** 对齐，把 global intent 注入 latent；actor 解码 \((s^{p-deploy}, s^{g-deploy}, z)\)。
- **仿真：** 8192 并行 env + domain randomization；指标 **SR / MPKPE / Vel-Dist / Acc-Dist**；MuJoCo sim-to-sim 与消融。
- **真机：** Unitree G1（1.3 m，控 23 DoF）；拉伸、武术、舞蹈、高踢、踢球、深蹲等 **单策略** 演示。
- **下游：** **MDM** 文本→SMPL→retarget→跟踪；**GVHMR** 单目视频→SMPL→retarget→跟踪（训练外参考源）。

## 核心论文摘录（MVP）

### 1) Oracle 特权训练 + deploy 观测设计

- **链接：** §II-A–C；Table III（奖励）
- **摘录要点：** Goal-conditioned RL；Oracle 用仿真全状态；Deploy 仅本体历史 + 稀疏 goal，刻意 **不** 依赖相机或多视角。
- **对 wiki 的映射：**
  - [UniTracker 实体页](../../wiki/entities/paper-loco-manip-161-024-unitracker.md)
  - [Privileged Training](../../wiki/concepts/privileged-training.md)
  - [Whole-Body Tracking Pipeline](../../wiki/concepts/whole-body-tracking-pipeline.md)

### 2) CVAE 在线蒸馏 vs MLP+DAgger

- **链接：** §II-D；Table I(a)；Fig. 3–4
- **摘录要点：** 显式把 reference 再喂给 actor 时 latent **被忽略**（退化为 DAgger）；纯 MLP 第二段在 OOD 明显变差。Ours：**SR 91.83 / MPKPE 82.62** vs DAgger w/o CVAE **88.21 / 84.79** vs scratch **58.32 / 145.59**（All AMASS Train）。
- **对 wiki 的映射：**
  - [UniTracker 实体页](../../wiki/entities/paper-loco-manip-161-024-unitracker.md)
  - [TWIST](../../wiki/entities/paper-twist.md)（同 teacher–student 线）
  - [Humanoid-GPT](../../wiki/entities/paper-humanoid-gpt.md)（对照：Transformer+scaling vs CVAE+distillation）

### 3) PHC 过滤与下游应用

- **链接：** §III-C；§III-D
- **摘录要点：** 未过滤 AMASS 训练 → 关键点误差与动作率上升、过激行为不适合真机；PHC 过滤后更平滑稳定。文本/视频外部参考可零样本跟踪。
- **对 wiki 的映射：**
  - [UniTracker 实体页](../../wiki/entities/paper-loco-manip-161-024-unitracker.md)
  - [Loco-Manipulation](../../wiki/tasks/loco-manipulation.md)
  - [Motion Retargeting (GMR)](../../wiki/methods/motion-retargeting-gmr.md)

## BibTeX

```bibtex
@misc{yin2025unitrackerlearninguniversalwholebody,
  title         = {UniTracker: Learning Universal Whole-Body Motion Tracker for Humanoid Robots},
  author        = {Kangning Yin and Weishuai Zeng and Ke Fan and Minyue Dai and Zirui Wang and Qiang Zhang and Zheng Tian and Jingbo Wang and Jiangmiao Pang and Weinan Zhang},
  year          = {2025},
  eprint        = {2507.07356},
  archivePrefix = {arXiv},
  primaryClass  = {cs.RO},
  url           = {https://arxiv.org/abs/2507.07356}
}
```

## 对 wiki 的映射

- 主实体页：[`wiki/entities/paper-loco-manip-161-024-unitracker.md`](../../wiki/entities/paper-loco-manip-161-024-unitracker.md)
- 项目页：[`sources/sites/humanoid-unitracker-github-io.md`](../sites/humanoid-unitracker-github-io.md)
- 项目页镜像仓：[`sources/repos/humanoid-unitracker.md`](../repos/humanoid-unitracker.md)
- 161 策展摘录：[`sources/papers/loco_manip_161_survey_024_unitracker.md`](loco_manip_161_survey_024_unitracker.md)
- 互链：[Loco-Manip 161 · 01 运控基座](../overview/loco-manip-161-category-01-motion-base-wbt.md)、[Humanoid Motion Tracking 选型](../../wiki/queries/humanoid-motion-tracking-method-selection.md)
