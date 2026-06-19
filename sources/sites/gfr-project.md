# GfR（Generalizing from References）项目页

- **类型：** 项目站 / RSS 2026 论文主页
- **主链接：** <https://jiashunwang.github.io/GfR/>
- **论文 PDF（镜像）：** <https://jiashunwang.github.io/GfR/static/mat/gfr_paper.pdf>
- **收录日期：** 2026-06-19
- **关联论文：** [mtrg_reference_goal_driven_rl_arxiv_2602_20375.md](../papers/mtrg_reference_goal_driven_rl_arxiv_2602_20375.md)（arXiv:2602.20375；官方项目简称 **GfR**）

## 一句话

**GfR** 为 RAI Institute / CMU 论文 *Generalizing from References using a Multi-Task Reference and Goal-Driven RL Framework*（**RSS 2026**）的官方主页，展示箱式跑酷泛化、长程技能组合、MuJoCo sim-to-sim 与感知扩展实验视频。

## 为什么值得保留

- **会议定稿证据**：主页标注 **Robotics: Science and Systems (RSS) 2026**，与 arXiv 预印本互补。
- **交互证据**：页面视频对应论文 Figure 1–6——单技能泛化、规则状态机长程组合、MuJoCo 跨引擎串联、真机多箱跑酷、以及 elevation map + one-hot 技能嵌入扩展。
- **BibTeX 与 PDF 镜像**：主页给出 `wang2026generalizing` 引用块；PDF 托管于项目站 `static/mat/gfr_paper.pdf`。

## 主页要点（相对 arXiv 摘要的补充）

- **长程技能组合**：基于箱体布局的 **rule-based state machine** 提供任务级 goal，将 walk-climb / walk-jump / climb-down 策略串联为多箱跑酷序列；真机与 MuJoCo 均展示「走—攀—下—走—跳—再攀」等组合。
- **MuJoCo sim-to-sim**：在 Isaac Lab 训练的策略不经重训即可在 **MuJoCo** 中组合执行，验证跨物理引擎鲁棒性（论文 Fig. 4）。
- **方法扩展（Fig. 6）**：同一框架可并入 **one-hot 技能嵌入**（单策略多技能）与 **elevation map** 外感受输入，无需改动核心目标或 \(\lambda\) 课程。

## 对 wiki 的映射

- [wiki/methods/mtrg-reference-goal-driven-rl.md](../../wiki/methods/mtrg-reference-goal-driven-rl.md)（本库以 **MTRG** 作方法导航标签；**GfR** 为官方项目名）
- [wiki/comparisons/hil-vs-mtrg-vs-zest-parkour-imitation.md](../../wiki/comparisons/hil-vs-mtrg-vs-zest-parkour-imitation.md)

## 参考链接

- 项目页：<https://jiashunwang.github.io/GfR/>
- arXiv：<https://arxiv.org/abs/2602.20375>
- 演示视频：<https://youtu.be/9NamvWhtFPM>
