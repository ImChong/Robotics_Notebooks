# KungFuAthleteBot

- **标题：** KungFuAthleteBot
- **类型：** repo / dataset
- **仓库：** <https://github.com/NPCLEI/KungFuAthleteBot>
- **项目页：** <https://kungfuathletebot.github.io/>
- **论文：** Lei et al., *A Kung Fu Athlete Bot That Can Do It All Day*, arXiv:[2602.13656](https://arxiv.org/abs/2602.13656) (2026)
- **机构：** 北京理工大学（BIT）、启元实验室（QIYUAN Lab）
- **收录日期：** 2026-07-02

## 一句话摘要

国家级武术运动员训练视频 → **KungFuAthlete** 高动态参考数据集（848 样本，Ground/Jump 子集）+ GVHMR/GMR 后处理与 **FastSAC 单策略 tracking+recovery** 训练栈；Ground 子集已 largely ready，Jump 与完整模型仍在 active development。

## 为何值得保留

- **动力学上界数据：** Jump 子集关节/线/角速度统计显著高于 LAFAN1、PHUMA、AMASS，适合 push humanoid WBT 极限。
- **视频→机器人完整管线：** GVHMR 重建 + GMR 重定向 + **根高度抛物线校正** + SG 平滑——对 noisy monocular 参考库有复用价值。
- **tracking∪recovery 单策略：** GRSI 跌倒初态 + LKE 采样 + 混合奖励，与 SafeFall/FIRM 分段策略、HoST 纯起身形成对照。
- **开源生态：** GitHub 227+ stars（2026-07）；与 Unitree G1 + Isaac Sim 5.0 栈对齐。

## 管线要点（编译自 README / 项目页 / 论文）

1. **采集：** 197 训练视频（谢远航等公开示范授权）→ 自动切分 1,726 子片段。
2. **重建：** GVHMR 单目人体网格恢复 → GMR 重定向到人形。
3. **校正：** 根高度漂移（地面接触 + 跳跃抛物线）+ Savitzky–Golay 时序平滑。
4. **训练 LoRA：** 848 最终样本；Daily Training / 拳术 / 器械 / 技巧（空翻、旋子）分类。
5. **训练（开发中）：** FastSAC + 混合 $r_{\mathrm{mt}}/r_{\mathrm{rc}}$ + GRSI + LKE；Isaac Sim 5.0 训练，MuMuJoCo 评测，G1 真机部署。

## 对 Wiki 的映射

- **wiki/entities/paper-kungfuathlete-humanoid-martial-arts-tracking.md**：论文+数据集+训练范式归纳。
- **wiki/comparisons/humanoid-reference-motion-datasets.md**：与 AMASS / PHUMA / LAFAN1 动力学对照。
- **wiki/tasks/balance-recovery.md**：单策略 tracking+recovery 真机案例。
