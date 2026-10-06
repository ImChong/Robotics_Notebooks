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


## 版本核查更新（2026-10-06）

- 新版论文：*KungfuAthleteBot: Learning High-Dynamic Humanoid Motion from Video with Unified Robust Recovery*, arXiv:2610.03388。它延续同一项目，故更新既有 Wiki 实体，不另建同名节点。
- README 公开 `retarget/` 高度修正脚本及 `unitree_rl_mjlab/` 训练/回放入口；README 勾选数据、height-adjusted code、training code、FastSAC、1307 recovery checkpoint 与 real deployment。代码仓库仍有 848 样本旧版概述。
- 当前 HF 数据卡和 README 后续统计为 992（Ground 822 / Jump 170）；新版论文附录 C 同为 992，但附录 E 与官网旧文本仍写 848。使用具体 release 文件时应检查版本。
- 新论文称项目页给出三阶段配置、恢复 checkpoint 和 30 fps qpos；论文称资产接收后按 MIT 发布，而 HF 卡当前 license 字段为 Apache-2.0。不同资产应逐一核对授权；运动员原始视频不分发。
- 来源互链：[2610.03388 论文](../papers/kungfuathletebot_arxiv_2610_03388.md)、[项目页](../sites/kungfuathletebot.md)、[HF 数据集](../datasets/kungfuathletebot-hf.md)、[Wiki 实体](../../wiki/entities/paper-kungfuathlete-humanoid-martial-arts-tracking.md)。
