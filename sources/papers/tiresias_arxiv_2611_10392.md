# Tiresias / LCWM-Bench（arXiv:2611.10392）

> 来源归档（paper · 原文通读）

- **标题：** Tiresias: Towards Language-Conditioned World Models
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2611.10392>
- **PDF：** <https://arxiv.org/pdf/2611.10392v1>（v1，首页标注 7 Nov 2026，cs.CV；论文内 Date 写 October 17, 2026）
- **代码：** https://github.com/UWMILab/Tiresias
- **机构（论文首页）：** 华盛顿大学（University of Washington）；新加坡国立大学；Microsoft Research；卡内基梅隆大学。通讯作者 Zhi-Qi Cheng。
- **入库日期：** 2026-11-10
- **一句话说明：** 提出 LCWM 任务与 LCWM-Bench（15 个数据集、4 个域），并给出 Tiresias：扩散策略生成 128 条候选，轨迹排序器留 8 条，视频世界模型想象后再用视觉排序器融合打分选 1 条。
- **核对范围：** 通读正文 §1–§6（含表 1–7、图 1–6）与附录 A（基准构建与统计）、B（Tiresias 模块细节、系统提示、伪代码）、C（指标定义、实现细节、表 A4 漏斗消融、表 A5 延迟）、D（局限）；数据集逐个介绍（A.1 各数据源长段说明）与附录定性图 A4–A8 仅浏览，参考文献未逐条核对。已读 GitHub README 的安装 / 数据 / 训练部分（只静态读，未运行、未下载权重或数据）。
- **通读：** 已完成

## 开源状态

- 代码仓库 `UWMILab/Tiresias` 的 README 可读（2026-11-10 核对）：含环境安装、Wan2.2-TI2V-5B 与 DINOv3 权重下载说明、`download_data.sh` 数据脚本、LDP / 世界模型 / 两个排序器的训练入口。README 数据目录只列 train / val_seen / val_unseen。权重与数据是否实际可下载：**未核实**。

## 核心摘录

1. **任务（§3.1）：** 输入初始第一视角 RGB 与语言指令，输出动作序列；四个域共享基础动作 (u_t, φ_t)（平面位移与偏航），3D 游戏加垂直位移，机器人加垂直位移、姿态与夹爪（7 维）。
2. **基准（§3.2–3.3、摘要）：** 15 个数据集、142,987 条 episode（约 4.4M 帧）、428,961 条指令，三种风格（Concise / Landmark / Intricate）；指令由 Qwen3.6-27B 粗标，再人工核对（§3.2、A.2）；1,200 条抽样人评均分 5.37/6（A.3）。
3. **方法（§4）：** LDP（仅语言条件的 DiT 扩散策略，从零训练）→ Tr-Ranker（仅语言与动作，预测 SR、nDTW、ATE、RPE）→ 世界模型（Wan2.2 初始化，flow matching，DINOv3 + umT5 交叉注意力）→ V-Ranker；四个模块独立训练、推理时无联合微调（§4.4）。
4. **主结果（§5.2，表 1）：** Tiresias 总分 val seen 63、val unseen 51；Tiresias (MPC) 为 59 / 46；最强对照 LCVN-WM 54（seen）、LCVN-Uni 42（unseen）。总分为 SR 与 nDTW 在四域上的宏平均。
5. **消融（§5.3）：** 随机取 8 条候选时，Tr-Ranker SR 0.30、V-Ranker 0.28、融合 0.39（表 5）；N 约 128 后 SR 饱和（图 6a）；128→8 保留约 85% 的全池 oracle SR（表 6）。
6. **成本（C.3，表 A5）：** 默认 128→8→1 的 SR 0.60、延迟 36 s；128→2→1 为 SR 0.55、9 s；NoMaD (lang) 0.32 / 0.7 s；LCVN-Uni 0.47 / 85 s。
7. **自述局限（附录 D）：** 未在实体机器人上部署，真机有效性待验证。

## 对 wiki 的映射

- [paper-tiresias](../../wiki/entities/paper-tiresias.md)
