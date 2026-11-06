# Retargeting Matters 全文通读（运动重定向视角）

> 来源归档（原文通读 · 支撑 `wiki/concepts/motion-retargeting.md`）

- **标题：** Retargeting Matters: General Motion Retargeting for Humanoid Motion Tracking
- **作者：** João Pedro Araújo, Yanjie Ze, Pei Xu, Jiajun Wu, C. Karen Liu（Stanford）
- **类型：** paper / retargeting / humanoid motion tracking
- **原文：** <https://arxiv.org/abs/2510.02252>（2026-11 通过 arXiv PDF v1 获取全文，pdftotext 转文本）
- **代码：** <https://github.com/YanjieZe/GMR>（本次未读仓库，仅依据论文）
- **一句话说明：** 在 BeyondMimic 免调奖励的条件下系统评估重定向质量对人形动作跟踪的影响，并提出 GMR（非均匀局部缩放 + 两阶段差分 IK）。
- **核对范围：** 通读全文 §I–§VI（引言、相关工作、§III 评测方法、§IV GMR 五步、§V 结果含表 I–III、用户研究、结论与局限）及参考文献列表；该论文无附录。未核对 GMR 仓库代码与论文网站，也未复现任何数值。
- **通读：** 已完成

## 核心摘录（均出自原文）

- **问题（§I）：** 常规流程是运动学重定向 + RL 模仿；脚滑、地面穿透、自穿透等伪影常被留给 RL 策略去纠正，或直接丢弃数据。作者假设：只要奖励 / 域随机化工程足够多，伪影可被掩盖；去掉这些工程后重定向质量就起决定作用。
- **相关工作（§II）：** 朴素做法直接复制关节旋转，会出现悬浮、足穿透 / 滑动、末端漂移，且需处理人体 SO(3) 关节到机器人（多为旋转副）的转换；vanilla WBGR 只做关键连杆朝向 IK 且忽略尺度；HumanMimic 用位置匹配 + 人工缩放系数；H2O / PHC 用 SMPL 拟合机器人形体并以梯度下降求 IK，慢且不考虑接触状态，SMPL 对形态差异大的机器人覆盖不佳；ProtoMotions 用沿世界轴的全局缩放 + Mink 差分 IK；KungfuBot 沿用 ProtoMotions 但关闭缩放。学习式重定向多需成对数据或语义标签，已有人形工作只覆盖上半身。
- **基线（§III-A）：** PHC：先把 SMPL 形状参数拟合到机器人骨架，再对根位姿与关节角做逐帧位置误差最小化（式 1，Adam / Adadelta，限位靠 clamp），后处理把最低点置 0；ProtoMotions：按世界轴缩放 + Mink，后处理把高度设为源动作最低高度；Unitree：闭源高质量重定向数据集。
- **GMR 五步（§IV，图 2）：** ① 人–机关键 body 映射并设位置 / 朝向权重；② 静止姿态下的笛卡尔空间朝向（及局部位置）对齐，缓解 toed-in；③ **非均匀局部缩放**：以人身高 / 参考身高给出全局因子，再乘每个关键 body 的自定义局部因子，相对根节点缩放（式 2），根平移用均匀因子缩放，这是避免脚滑的关键；④ 第一阶段 IK：仅约束各 body 朝向 + 末端位置（式 4），根位姿由缩放后的人体根位置与 yaw 初始化，用 Mink 差分 IK（式 5，带关节限位，最多 10 次迭代、收敛阈值 0.001）；⑤ 第二阶段 IK：以上一步为初值，换一组权重、约束全部关键 body 的位置与朝向（式 6）。序列上逐帧顺序求解、以前一帧结果作初值；最后用 FK 求最低高度并从全局平移中减去以消除悬浮 / 穿透。
- **评测设置（§III-B–E）：** LAFAN1 的 21 段（5 秒–2 分钟，不含手脚外的复杂接触），重定向到 Unitree G1；BVH 经 SMPL(-X) 拟合转换供 PHC / ProtoMotions 使用；BeyondMimic 逐段训练，IsaacSim 无域随机化 100 次、有域随机化 4096 次、MuJoCo/ROS sim2sim 100 次；成功 = 跑完整段且锚点身体高度 / 朝向偏差不超阈值；跟踪误差 Eg-mpbpe / Empbpe / Empjpe；用户研究 20 人，每人 45 题，比较 GMR 与其余三者对源动作的忠实度。
- **结果（§V，表 I–II）：** 21 段中 11 段对所有方法成功率均 >98%，3 段满分；其余 7 段差异大。Unitree 近乎满分，GMR 与 ProtoMotions 紧随其后（个别例外：GMR 的 Dance 5、ProtoMotions 的 Run (stop & go)），PHC 最差（Dance 1 / 2 在三种设置下几乎为 0）。跟踪误差均值：Eg-mpbpe PHC 247.8 / GMR 104.1 / PM 139.7 / Unitree 77.2 mm。2 分钟的 Dance 3 在 PHC 重定向下可成功，说明长序列本身不是难点。
- **关键伪影（§V-B，图 3）：** PHC 的 Dance 1 / 2 有明显地面穿透（一例约 60 cm）；ProtoMotions 的 Run (stop & go) 双腿自相交；GMR 的 Dance 5 有腰部 roll/pitch 突跳（约占完整数据集不足 2%，源于优化阶段，统一权重下偶发，可能需按动作调权重）。结论：物理不一致的高度、自穿透、关节值突跳应避免。
- **用户研究（§V-C，图 4）：** GMR 被认为比 PHC、ProtoMotions 更忠实于参考；Unitree 比 GMR 更忠实，但用户更难区分二者。
- **首帧影响（§V-D，表 III）：** 同一策略仅改变参考起始帧即可显著改变 sim2sim 成功率（如 Turn 1 在 PHC 下从 start=0 的 14% 到 start=49 的 100%）；建议保证起始姿态机器人可安全到达、结束于稳定姿态。
- **局限（§VI）：** 动作全部来自 LAFAN1；只用 G1（受 BeyondMimic 代码库限制）；不涉及与环境 / 物体 / 其他机器人的交互；GMR 偶需调权重。

## 对 wiki 的映射

- [`wiki/concepts/motion-retargeting.md`](../../wiki/concepts/motion-retargeting.md)
- 关联档案：[gmr_retargeting_matters_arxiv_2510_02252.md](gmr_retargeting_matters_arxiv_2510_02252.md)（摘要 + 仓库入口核对，通读未完成）
