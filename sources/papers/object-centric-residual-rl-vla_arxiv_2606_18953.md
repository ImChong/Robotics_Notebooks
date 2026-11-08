# Object-Centric Residual RL for Zero-Shot Sim-to-Real VLA Enhancement（arXiv:2606.18953）

> 来源归档（paper）

- **标题：** Object-Centric Residual RL for Zero-Shot Sim-to-Real VLA Enhancement
- **类型：** paper
- **作者与机构：** Kinam Kim（KAIST / Microsoft Research Asia - Tokyo，实习期间完成）、Namiko Saito、Heecheol Kim、Katsushi Ikeuchi、Jaegul Choo、Yasuyuki Matsushita（首页署名：KAIST；Microsoft Research Asia - Tokyo；东京大学）
- **arXiv：** <https://arxiv.org/abs/2606.18953>（v1，cs.RO，2026-06-17）
- **PDF：** <https://arxiv.org/pdf/2606.18953>
- **项目页：** <https://www.microsoft.com/en-us/research/articles/object-centric-residual-rl/>（论文首页给出；本次未读）
- **代码：** 论文正文未给出代码仓库链接（未核实是否另有发布）
- **入库日期：** 2026-11-08
- **核对范围：** 下载 arXiv PDF（8 页正文 + 附录至第 19 页，含参考文献之后的附录 A.1–A.11），pdftotext 逐段通读摘要、引言、相关工作、方法 3.1–3.4、实验设置、结果 5.1–5.4、结论、局限（第 7 节）与全部附录（奖励、仿真搭建、超参、伪代码、行为一致性、强基座、涌现行为、失败案例）；表 1/2/3/4/5/6/7 与图 6/图 7 的柱状数值已核对（图 6、7 另渲染页面图像读数）；图 9–12 的照片类关键帧未逐帧分析；参考文献 [1]–[41] 只读了被正文引用处的叙述，未逐篇阅读被引论文；未读项目页与视频。
- **通读：** 已完成

## 一句话说明

在仿真中只用 6-DoF 物体位姿 + 本体感觉 + 基座 VLA 当前动作训练一个小型残差策略（TD3），叠加到冻结 VLA 上零样本迁移到真机；GR00T-N1.5 基座在 FR3 五任务上真机平均成功率 42%→76%（表 1）。

## 核心摘录（均出自论文原文，章节 / 表号随文标注）

1. **问题设定（摘要、第 1 节）：** 作者指出 VLA 基于模仿学习，小误差累积导致失败；扩散 / 流匹配动作头难以直接做策略梯度，因此用残差 RL；既有残差 RL 三条路各有问题：特权状态蒸馏（ResiP [16]）、图像式仿真残差（视觉域差）、真机 RL（ResFiT [17]、PLD [18]，成本与安全）。
2. **方法三阶段（第 3 节、图 2）：** (1) 同一批遥操作动作在仿真中回放，渲染出仿真数据集，训练仿真 VLA（π_VLA^sim），与真实 VLA（π_VLA^real）并行（3.1）；(2) 仿真中训练残差 π_res^sim，观测 s_t=[s_obj, s_prop, a_base]（式 2），动作合成 a_t=a_base ⊕ π_res(s_t)，位置与夹爪相加、旋转用四元数乘法（式 1、7）；(3) 部署时用真实 VLA 取代仿真 VLA，位姿由 FoundationPose + SAM2 提供（3.2–3.3）。
3. **位姿鲁棒化（3.2、A.6）：** 每步对位置加 U(−σ̃p,σ̃p)、σ̃p~U(0,σp_max) 的噪声，σp_max=5 mm；朝向噪声 σq_max=0.1 rad；以 ρ_drop=0.1 的概率整体置零位姿；部署时当跟踪置信度 c_t 低于 τc=0.5 置零位姿（A.6、算法 2）。
4. **训练设置（第 4 节、A.5、A.6）：** GR00T-N1.5 为基座，每任务每域（仿真 / 真实）30 条遥操作示范微调；残差为 2 层 MLP（actor 512、critic 1024 隐层）、TD3、MuJoCo；动作块长 H=16；单次前向约 0.06 ms，VLA 推理约 140 ms；FoundationPose 跟踪约 18 ms/帧，异步运行（3.3）。奖励为分阶段稠密奖励，截断到 [0,1]（A.1、表 3）。
5. **主结果（表 1，每格 20 次试验）：** 真机 Base→+Res：Cube Lift 7→17，Pick-and-Place 9→16，Stack Cube 7→15，Close Drawer 14→20，Stand Cup Up 5→8，平均 8.4/20→15.2/20，即正文所称 42%→76%；仿真（3 seed 均值）平均 7.6/20→17.2/20。
6. **消融（表 2，真机 x/20）：** 去噪声注入、去位姿 dropout、两者都去均下降（Ours 17/16/15/20/8；w/o both 12/10/9/16/5）；观测空间对比：蒸馏式 9/4/8/20/5，图像式 8/10/9/14/6；图 6(b) 平均真机成功率 Ours 76%、蒸馏 46%、图像 47%，仿真侧 86/79/74%。
7. **其他骨干（5.1 末、图 6(a)、A.8、表 7）：** 换 π0.5 作基座，Close/Lift/Stand 真机 12→18、6→11、7→13（图 6(a) 读数）；Pick-and-Place 上强基座 π0.5 真机 17/20→17/20（无提升），GR00T-N1.5 弱基座 9/20→16/20（表 7）。
8. **分析（5.3、图 7）：** 基座动作偏离目标时残差修正更强；成功回合的完成时间缩短 9–22%。
9. **自我改进（3.4、5.4、图 6(c,d)）：** 用残差修正后的成功 rollout 与原示范合并，对基座做 SFT：平均真机成功率 42%（基座）→ 47%（用基座 rollout 重训）→ 59%（用残差 rollout 重训）；平均回合长度 275→252→196 步。
10. **作者自述局限（第 7 节）：** 依赖实时 6-DoF 位姿跟踪（遮挡、杂乱会失败）；任务相关物体需人工指定；位姿观测弥合视觉域差但不弥合动力学域差（摩擦、夹爪柔顺）；残差只能修正轻度偏差，无法从远离基座训练分布的状态恢复；亚毫米精度或很小物体可能超出位姿估计精度。
11. **失败案例（A.11）：** 位姿偏移 / 漂移导致往错误方向修正、夹爪遮挡丢失目标、分割锁定到镜面反射；三类均为通过置信度门限的感知失败。
12. **涌现行为（A.9，作者自述）：** 真机上观察到 Cube Lift / Pick-and-Place 的抓取前旋转立方体、Stack Cube 的纠正性推动、Close Drawer 的持续接触推动，作者称这些行为不在示范数据中（仅为定性关键帧展示，图 10）。

## 对 wiki 的映射

- [paper-object-centric-residual-rl-vla](../../wiki/entities/paper-object-centric-residual-rl-vla.md)
- [residual-policy-learning](../../wiki/methods/residual-policy-learning.md)
