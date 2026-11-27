# SPARC: Reliable Spatial Annotations from Robot Demonstrations at Scale

> 原文核对归档（ingest）

- **作者：** Nils Blank, Paul Mattes, Maximilian Xiling Li, Jakub Suliga, Thomas Roth, Moritz Reuss, Pankhuri Vanjani, Rudolf Lioutikov
- **机构：** Karlsruhe Institute of Technology (KIT); NVIDIA; Robotics Institute Germany (RIG)
- **版本 / 载体：** arXiv:2606.13497v2 [cs.RO]（2026-10-14；v1 提交于 2026-06-11；CoRL 2026 录用）
- **原文：** <https://arxiv.org/abs/2606.13497>（HTML 全文：<https://arxiv.org/html/2606.13497>）
- **项目页 / 代码：** **已开源** — 项目页 <http://intuitive-robots.github.io/sparc-labeling>（镜像 <https://sparc-labeling.github.io>）；官方标注流水线源码 <https://github.com/intuitive-robots/sparc>；Hugging Face 集合 <https://huggingface.co/collections/irl-kit/sparc>（已公开 `SPARC-Qwen3.5-4B`、`SPARC-VQA`、`IA-Bench` 及 `sparc-droid-annotations`；BridgeData / AgiBotWorld / Open X-Embodiment 标注显示 Coming soon）
- **核对范围：** 通读 `arXiv:2606.13497v2` HTML 全文（Abstract, Sec. 1–5, Appendix A–F；含 Tables 1–21、Eq. 1–6、Figs. 1–7 文本说明）及官方项目页与 GitHub README。
- **通读：** 已完成

## 原文核对笔记

1. **核心问题与选择性预测建模：** 现有自动化空间标注管线直接将目标检测器置信度作为物体身份与质量过滤依据，但在杂乱、遮挡场景下，检测器置信度只反映外观识别置信度而非"正在被机器人操纵的对象"，导致高置信度错锁干扰物并在跟踪中传播误差。SPARC 将自动化时空标注建模为**选择性预测（selective prediction）**问题，为每条候选标注输出基于物理交互证据的连续可靠性分数 $R_i$，从而在无需人工审核的情况下显式调控质量—覆盖率权衡。
2. **三阶段管线与混合信号评分（Sec. 3 & Appendix B）：**
   - **Stage 1（子任务分解与夹爪相位检测）：** 对归一化夹爪信号 $a_t \in [0, 1]$ 进行双遍滞回状态机分割（$\tau_c = 0.6, \delta = 0.05$，自适应最小时长 $d_{\min}$ 过滤 `grasp_failure`，双臂独立处理），随后用 `Qwen3.6-30B-MOE` 将语言指令与检测到的 grasp/interact/release 相位对齐为结构化子任务 JSON。
   - **Stage 2（候选提议、分割与 3D 跟踪）：** 在首次抓取中点关键帧 $t^\star$ 用低阈值（0.05）`LLMDet-Base` 提议 5–25 个候选框，结合早期/晚期帧生成目标位置候选框；`SAM2.1 Hiera-Small` 生成实例掩码，`RobotSeg` 分割机械臂与夹爪，`AllTracker`（6 fps 原始分辨率）提取稠密像素轨迹，并通过 `MoGe-2` 单帧几何预测将其提升至 3D 轨迹 $\Gamma_i$。
   - **Stage 3（可靠性评分）：** 组合四类互补信号：相位感知运动信噪比 $\widehat{A}_i$（Selector，区分交互期与非交互期运动，$\alpha=0.6, \beta=0.2$）、检测器置信度 $D_i$（Anchor，跨样本标定尺度）、3D 夹爪近邻比例 $\widehat{P}_i$（Conditional，自适应球半径 $k_r=8$）、以及经 3D 深度差衰减的机器人重叠二次惩罚 $\Pi_i$（Rejector，抑制误选夹爪同时保留接触物体）。总分 $R_i = w_A \widehat{A}_i + w_D(\widehat{P}_i) D_i + w_P \widehat{P}_i - \Pi_i$（$w_A=0.5, w_P=0.3, w_D(\widehat{P}_i)=0.75-0.15\widehat{P}_i$）。
3. **基准测试、下游 VLM 与真机策略（Sec. 4 & Appendix C–E）：**
   - **IA-Bench：** 1,748 条人工标注交互起始/目标框（AgiBotWorld 451、DROID 405、BridgeData 454、OXE 438 零样本测试），覆盖 12 种本体（含双臂）。测试集（$n=1,256$）上 SPARC 达到 **80.2%** 准确率、**77.6%** Cov@90、**0.056** AURC，显著优于检测器基线（LLMDet 58.1% Acc / 26.6% Cov@90）与 FSD（38.6% Acc / 33.1% Cov@90），且在低检测置信度边际（low-margin）子集上将准确率从 35.8% 提升至 73.5%。规则无需再标定即可零样本迁移至 OXE（83.5% Acc）并可适配人类第一视角视频 EgoLive（75.4% Acc, 57.5% Cov@90）。
   - **下游 VLM（SPARC-VQA）：** 从 28 万条轨迹在 95% 精度阈值下筛出 16.7 万条高质量轨迹（检测基线仅 3 万条），构建约 51.1 万条 VQA 样本微调 `Qwen3.5-4B`。空间指向基准平均分 **69.6**，优于人类标注的 `EO-1.5M`（62.6）及 7B 开源具身大脑模型（RoboBrain 2.0: 53.8, MiMo-Embodied: 60.5）。
   - **真机策略（Franka-Panda, 100 rollouts）：** 在 10 个视觉歧义杂乱桌面任务（250 条演示）上，SPARC 轨迹推理 VLA（Qwen3.5-0.8B + DiT-B）实现 **64/100** 正确物体定位与 **31/100** 任务成功率，显著高于检测推理策略（21/100 定位, 12/100 成功, $p=.0017$）与无推理基线（21/100 定位, 9/100 成功, $p=.00015$）。
4. **局限核对：** 受限于上游预训练感知模型（如目标部件误检、相似物跟踪 ID 切换、透明物体 3D 重建误差）；依赖夹爪开合进行相位切分，无夹爪开合（fallback 74.1% Acc）或视觉静默接触任务（按按钮 Press/Switch 仅 38.5% Acc）性能下降；假设相机基本静止。

- 对 wiki 的映射：[SPARC](../../wiki/entities/paper-agentic-2606-13497.md)
