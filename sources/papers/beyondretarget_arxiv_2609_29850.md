# BeyondRetarget（arXiv:2609.29850）

> 来源归档（ingest）

- **标题：** BeyondRetarget: Learning Executable Humanoid Motions Directly from Monocular Video
- **缩写：** **BeyondRetarget**
- **类型：** paper / monocular-video / humanoid motion / end-to-end retargeting / teleoperation
- **arXiv：** <https://arxiv.org/abs/2609.29850>
- **PDF：** <https://arxiv.org/pdf/2609.29850>
- **项目页：** <https://bear-ty.github.io/Beyondretarget_page/> — 归档见 [`sources/sites/beyondretarget-github-io.md`](../sites/beyondretarget-github-io.md)
- **代码：** <https://github.com/bear-ty/BeyondRetarget> — 归档见 [`sources/repos/beyondretarget.md`](../repos/beyondretarget.md)
- **Demo：** <https://huggingface.co/spaces/bear-ty/BeyondRetarget>
- **作者：** Tianyu Xiong*、Yi Lu*、Jinrui Wang、Ziqi Liang、Dandan Lei、Xiaoyang Zhou、Xiao-xiao Long、Qiu Shen†、Xun Cao（* 共同一作；† 通讯）
- **机构：** 南京大学电子科学与工程学院；南京大学智能科学与技术学院；江苏移动信息系统集成有限公司；中国移动紫金（江苏）创新研究院
- **入库日期：** 2026-09-26
- **开源状态（步骤 2.5，2026-09-26）：** GitHub **已开源**（`scripts/setup_rgb2robo.sh`、推理脚本、HMR2 预处理；checkpoint 经 Google Drive）；当前发布为 **base 版**（满足遥操作实时性；**不支持** 浮动相机与大范围全局轨迹；项目页承诺后续 **performance 版**）。

## 核心论文摘录（MVP）

### 1) 问题：两阶段 human→robot 的信息瓶颈与误差累积

- **链接：** <https://arxiv.org/abs/2609.29850>
- **核心贡献：** 主流管线先估计 **显式人体运动（SMPL 等）** 再 **重定向**；人体与人形在 DoF、动力学与比例上差异大，且 **第一阶段误差无法与重定向联合优化**。BeyondRetarget **丢弃推理期人体中间表示**，从单目 RGB 直接学 **面向机器人的隐式共享 motion 表征**，经 **robot-specific decoder** 输出可执行轨迹。
- **对 wiki 的映射：**
  - [BeyondRetarget 论文实体](../../wiki/entities/paper-beyondretarget-monocular-humanoid.md)
  - [Motion Retargeting Pipeline](../../wiki/concepts/motion-retargeting-pipeline.md)
  - [Teleoperation](../../wiki/tasks/teleoperation.md)

### 2) 架构：视觉编码 → 时序聚合 → 多机解码 → 接触感知 refine

- **核心贡献：** 冻结 **person detector + HMR2.0** 提帧级特征；**BiGRU + RoPE self-attention + cross-window attention**（TCAM）得共享 $\mathbf{u}_t$；各机器人 **残差解码** root 平移、连续 root 旋转与关节；**contact-aware** 模块做时序滤波、root 修正与约束腿 IK。支持 **Unitree G1/R1/H1、Booster T1、Tienkung、Fourier GR1-T1/GR2-V3、Atlas** 等八款人形。
- **对 wiki 的映射：**
  - [GMR](../../methods/motion-retargeting-gmr.md)（对照：优化式两阶段）
  - [NMR](../../methods/neural-motion-retargeting-nmr.md)（对照：学习式重定向）

### 3) 定量与实时性（项目页表，2026-09-26）

- **相对 GVHMR/WHAM → GMR/NMR：** 报告 **RAMPJPE 26.36 mm**、**Simulation SR 95.86%**；两阶段基线 **motion collapse** 与更高 failure rate。
- **流式推理（同 GPU）：** **Latency ~192.8 ms**、**~50 FPS throughput** vs GVHMR→GMR **~1369 ms**。
- **对 wiki 的映射：** [SONIC / 全身遥操作栈](../../methods/sonic-motion-tracking.md)（真机 teleop 可经 SONIC + 本项目 GR00T ZMQ 接口对接）
