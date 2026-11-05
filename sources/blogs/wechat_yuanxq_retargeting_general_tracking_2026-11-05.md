# 具身智能研究室：重定向能不能借通用运控再往前走一步

> 来源归档（blog / 微信公众号）

- **标题：** 这三年，机器人训练越来越快，数据重定向能不能借通用运控再往前走一步？
- **账号：** 具身智能研究室（Yuanxq，微信公众号）
- **原始链接：** https://mp.weixin.qq.com/s/LAjf3Tuk4LqLWlRU955Yhg
- **发表日期：** 2026-11-05
- **入库日期：** 2026-11-05
- **一句话说明：** 作者认为训练已大幅加速而重定向仍留有穿模、自碰撞和物理不可行问题，主张让通用运控模型的执行结果反过来参与参考动作的生成与修正，并按四类列出 16 个相关方法。
- **说明：** 全文是**作者表述与判断**，文中没有给出表格、实验或引用链接；下文「文章要点」是转述，「核对表」是入库时对各方法一手资料（arXiv 摘要页、官方仓库）的核对结果，不是文章内容。文末的赞赏、开源知识库推广与 Previous/Next 导航不收录。
- **核对范围：** 16 个方法的 arXiv 摘要页（标题 / 作者 / 日期）与官方仓库（git ls-remote、README、入口源码）；论文全文未逐篇通读。
- **沉淀到 wiki：** [retargeting-vs-general-tracking-speedup-2026-11](../../wiki/overview/retargeting-vs-general-tracking-speedup-2026-11.md)

## 文章要点（作者表述）

1. 训练侧：并行仿真、现成训练流程、可微世界模型让通用跟踪策略越来越便宜；作者举 SuperMimic、BlinkMimic，称 BlinkMimic 覆盖 40 段 G1 重定向 LAFAN1 动作的策略训练约三分钟。
2. 训练变便宜后，可以更频繁比较不同参考、更早在数据处理阶段发现问题，因此重定向可以「生成参考后立即让机器人执行，用实际运动修正参考」。
3. 重定向现状分三类：身体对应（GMR 的比例适配、UMR 的表面稠密对应）、接触保持（OmniRetarget 的交互网格，CoRe、VICAR）、动力学可行性（KDMR、SPARK、Shooting for Contact 的 DSMS，DynaRetarget）。作者称目前没有公认的、同时处理几何、碰撞、接触和动力学的通用开源框架。
4. 作者的核心矛盾：「几何上可定义」与「动力学上可实现」之间的差距；关键点对齐不等于网格不相交；逐帧映射难以处理速度、加速度、力矩与接触切换。
5. 用通用运控模型修动作：多候选 + 同一控制器执行择优（S³）；让参考参与优化（ReActor 联合优化参考参数与策略，MimicX 用执行误差调整监督与课程）；把物理修正后的动作与人体动作配对训练重定向模型（NMR）。
6. 期待的系统：几何求解给初值，再用控制优化或通用跟踪策略执行，把碰撞、接触丢失、力矩饱和、失稳反馈给参考生成（作者举 FRDV）；评价应看穿透深度与持续时间、自碰撞、脚滑、接触丢失、力矩裕度和任务完成，而不只是平均姿态误差。
7. 文末分类：训练与跟踪（SuperMimic、BlinkMimic）；身体对应与接触保持（GMR、UMR、OmniRetarget、CoRe、VICAR）；动力学重定向（KDMR、SPARK、DynaRetarget、Shooting for Contact / DSMS）；执行反馈与数据学习（S³、ReActor、MimicX、NMR、FRDV）。

## 核对表（入库时一手资料核对）

| 方法 | arXiv / 来源 | 作者（摘要页） | 日期 | 官方代码（2026-11-05） | wiki 节点 |
|---|---|---|---|---|---|
| SuperMimic | 2610.28480 | Liao, Shao, Long, Zakka, Horowitz, Abbeel, Sreenath | 2026-10-20 | 项目页只链论文，无仓库 | [paper-supermimic](../../wiki/entities/paper-supermimic.md) |
| BlinkMimic | 无论文；社区仓库 fuyichen0725/blinkmimic | — | — | MIT，社区实现，非 SuperMimic 官方 | [blinkmimic](../../wiki/entities/blinkmimic.md) |
| GMR | 2510.02252 | Araujo, Ze, Xu, Wu, Liu | 2025-10-02 | YanjieZe/GMR，MIT | [motion-retargeting-gmr](../../wiki/methods/motion-retargeting-gmr.md) |
| UMR | 2609.02134 | Cao, Fang, Kwon 等 13 位 | 2026-09-02 | hanyang9/UMR 可达 | [paper-umr-unified-motion-retargeting](../../wiki/entities/paper-umr-unified-motion-retargeting.md) |
| OmniRetarget | 2509.26633 | Yang, Huang, Wu, Kanazawa, Abbeel, Sferrazza, Liu, Duan, Shi | 2025-09-30 | amazon-far/holosoma | [paper-hrl-stack-03-omniretarget](../../wiki/entities/paper-hrl-stack-03-omniretarget.md) |
| CoRe | 无 arXiv；Humanoids 2025，DOI 10.1109/Humanoids65713.2025.11203055（Crossref 核对标题） | Jeong, Chai, Choi, Bak, Kim 等 | 2025-09-30 | tmjeong1103/CoRe 默认分支为项目页模板 | [paper-core](../../wiki/entities/paper-core.md) |
| VICAR | 2610.13079 | Kalaria, Zhang, Sastry | 2026-10-09 | 官方仓库仅项目网站与回放 | [paper-vicar-contact-aware-video-imitation](../../wiki/entities/paper-vicar-contact-aware-video-imitation.md) |
| KDMR | 2603.09956 | Zhang, Haener, Madabushi, Tucker | 2026-03-10 | 宣称发表时开源，暂无 | [paper-kdmr](../../wiki/entities/paper-kdmr.md) |
| SPARK | 2603.11480 | Wang, Liao, Zhang, Ren, Sreenath, Xiong | 2026-03-12 | 项目页无代码 | [paper-spark-skeleton-aligned-retargeting](../../wiki/entities/paper-spark-skeleton-aligned-retargeting.md) |
| DynaRetarget | 2602.06827 | Dhedin, Taouil, Omar, Yu, Tao, Dai, Khadiv | 2026-02-06 | Atarilab/sbto（仅 SBTO 阶段） | [paper-notebook-dynaretarget-dynamically-feasible-retargeting-us](../../wiki/entities/paper-notebook-dynaretarget-dynamically-feasible-retargeting-us.md) |
| Shooting for Contact | 2608.03116 | Esteban, Siu, Mach, Li, Kurtz, Burdick, Ames | 2026-08-04 | sesteban951/shooting-for-contact | [paper-shooting-for-contact](../../wiki/entities/paper-shooting-for-contact.md) |
| S³ | 2609.26420 | Memmesheimer, Behnke | 2026-09-22 | 项目页 Code 禁用，未开源 | [paper-sample-simulate-select](../../wiki/entities/paper-sample-simulate-select.md) |
| ReActor | 2605.06593 | Müller, Serifi, Christen, Grandia, Knoop, Bächer | 2026-05-07 | 未找到官方代码 | [reactor-physics-aware-motion-retargeting](../../wiki/methods/reactor-physics-aware-motion-retargeting.md) |
| MimicX | 2610.09055 | Liu, Zhang, Liu, You 等 10 位 | 2026-10-06 | NEBULIS-Lab/MimicX，Apache-2.0 | [paper-mimicx](../../wiki/entities/paper-mimicx.md) |
| NMR | 2603.22201 | Zhao, Yang, Wang 等 9 位 | 2026-03-23 | 3DV-HumanoidGroup/MakeTrackingEasy（仅推理） | [neural-motion-retargeting-nmr](../../wiki/methods/neural-motion-retargeting-nmr.md) |
| FRDV | 2610.28470 | Marew, Park, Perera, Kim | 2026-10-20 | DARoSLab/FRDV，MIT（仅运动学部分） | [paper-fast-motion-retargeting-pipeline-humanoid](../../wiki/entities/paper-fast-motion-retargeting-pipeline-humanoid.md) |

## 未核实项

- 文章「约三分钟」对应 BlinkMimic README 的作者自报结果，未复现；见概览页。
- 文章对各方法归类与效果的措辞（例如 CoRe 处理「脚—地接触和动作修正」、UMR「表面对应」）按各页已通读内容对照，未逐篇重读全文。
- 文中「目前没有公认的通用开源框架」是作者判断，无法核实。
