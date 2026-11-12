# ReSYNC：从机器人失败中学习技能与概念（arXiv:2606.18328）

> 来源归档（paper）

- **标题：** Recover, Discover, Plan: Learning Skills and Concepts from Robot Failures
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2606.18328>（v1，2026-06-16）
- **PDF：** <https://arxiv.org/pdf/2606.18328v1>
- **项目页：** <https://jaraxxus-me.github.io/ReSYNC/>
- **代码：** https://github.com/Jaraxxus-Me/ReSYNC （项目页「Code」按钮所指；本环境无法访问该仓库，未核对其内容与完整度）
- **作者机构：** CMU、Princeton、AI2、MIT、Centaur AI、Bosch Center for AI（论文首页）
- **入库日期：** 2026-11-12
- **一句话说明：** ReSYNC 在失败驱动的课程中交替做两件事：用 PPO 学恢复技能，再用「做梦」生成的数据发现关系谓词并更新 PDDL 算子，使恢复技能能在未见任务里被抽象规划主动调用。

- **核对范围：** 读了 arXiv v1 PDF 的正文第 1–6 节（含 Table 1–3、图 1–6）、参考文献，以及附录 C（基线细节）、E（方法细节：PDDL 形式化、谓词选择算法、失败物体识别、终止函数学习）、F、G、H、I、J；附录 A（符号表）、B（各场景详细成功率）、D（域细节）仅浏览标题与零星内容，附录 E 后半段的谓词池学习与算子学习公式只读了开头；未读补充视频；未运行代码，也未访问代码仓库。
- **通读：** 已完成

## 核心摘录

1. **问题设定（§3）：** 关系分解 MDP（RF-MDP），稀疏奖励；训练时可重置的动力学模型，测试时单次有限时域 rollout；初始技能 C0、谓词 Ψ0、算子 Op_C0 只能解简单任务；用户给出分阶段课程（实验中每阶段 N_i=10 个任务），假设能检测失败状态（如仿真碰撞）。
2. **技能学习（§4.1）：** 失败挖掘（对失败物体 o^F 相对相关物体 o^R 拟合成对高斯并重采样扩充失败集 D_F）；恢复 MDP 的成功判据是到达一个「继续规划能成功」的状态；用 PPO 学恢复策略，监督学习终止函数；策略与终止函数都是关系式的（只看 o^R 与 o^F 的特征）。
3. **概念学习（§4.2）：** 以 IVNTR（RSS 2025）为离线谓词发现骨架；因无演示，用「做梦」造数据：在失败前/后两套成对高斯里重采样 o^F，覆盖恢复技能被复用的情形（例：开着但空的抽屉，用来区分 IsOpen 与 HammerInOpenDrawer）。
4. **渐进学习（§4.3）：** 组合式做梦（把前面阶段的失败物体与分布组合起来）与概念微调（旧谓词随新技能改变的状态分布而微调，避免语义漂移）。
5. **实验（§5）：** 4 个仿真域（Icy Transport、Blocked Stacking、Cluttered Drawer、Rearrange）与 2 个真机域（Cornered Insertion-S/-L，UR7e）；5 个随机种子；基线 GNN/TF 策略、GPT-5.4 单样例 VLM 规划、DSG-S/M、Recovery Chaining（RC）、RC+IVNTR。引言称在组合新任务上平均成功率 70%；§5.2 称在 T_i^test 上比基线高 20–70%。
6. **真机（§5.2，表 2）：** 每个场景 10 个随机初始化任务 x 3 个种子；T1test 上 ReSYNC 约 0.77（S）与 0.63（L），RC 为 0.00；T1train 上 RC 与 ReSYNC 相当（S：0.77 对 0.73；L：0.63 对 0.67）；附「人类指定下一技能」参照 0.80/0.73/0.80/0.77。
7. **消融（§5.3）：** 无做梦的 RC+IVNTR 在第一阶段掉 10–36%；无微调（表 3，Blocked Stacking）测试成功率最多降 39%；与测试时恢复学习（TTRL）相比，Icy Transport 阶段 1 训练回合约少 22 倍（图 6，约 25 对 544）。
8. **局限（§6、附录 I、J）：** 依赖用户给定的诱发失败课程；做梦用高斯分布，难以扩展到高维观测；继承 IVNTR 的表达力有限与技能到算子一一对应；真机最常见失败是插入（脚本式高斯滤波控制器、无力反馈），RL 恢复技能亦受 sim-to-real 差距影响；Rearrange（6 物体）的组合做梦在单张 A100 上约 2 小时。

## 对 wiki 的映射

- [paper-resync-recovery-driven-skill-concept-discovery](../../wiki/entities/paper-resync-recovery-driven-skill-concept-discovery.md)
