# Grounded Action Model: 3D Grounding as a Foundation for Robotics

> 来源归档（paper · ingest）

- **标题：** Grounded Action Model: 3D Grounding as a Foundation for Robotics
- **短名：** GAM（Grounded Action Model；**勿与** RCL 清单中 Geometric Action Model 缩写混淆）
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2609.23863>（v2，2026-09-25）
- **PDF：** <https://arxiv.org/pdf/2609.23863>
- **项目页：** <https://grounded-action-model.github.io/>
- **代码：** <https://github.com/GehaoZhang6/Grounded-Action-Model>（**待发布** — README「Code coming soon」）
- **机构：** 西北大学（Northwestern University）；华盛顿大学（University of Washington）；新加坡国立大学（National University of Singapore）
- **作者：** Gehao Zhang, Weikai Huang, Shailesh Shailesh, Yiyan Peng, Jiafei Duan, Ranjay Krishna
- **入库日期：** 2026-10-01
- **一句话说明：** 以冻结 WildDet3D 将语言/点/框提示解析为对象中心 2D 特征与度量 3D 几何，经多流 transformer + flow matching 输出动作块；RoboTwin 2.0 / LIBERO-PRO / 双臂 YAM / Franka+Molmo2 上报告 SOTA 或强鲁棒性。

## 开源状态（步骤 2.5）

- **结论：** **待发布** — 项目页链 GitHub，但仓内 **无** 训练/推理脚本与权重（2026-10-01 核查）。

## 核心摘录

1. **问题：** 现有 VLA / WAM 预训练骨干 **不直接要求** 度量 grounding，对象「是谁、在哪」多靠机器人演示 **隐式** 学；操纵策略却必须显式知道相关物体与空间位置。
2. **范式：** **Grounded Action Models（GAM）** — 以 **3D grounding** 为 foundation，把语言、2D 点或 2D 框 **统一** 成共享的对象中心表示（目标视觉特征 + 度量几何）。
3. **Grounding 栈：** 语言指令经 **Flan-T5** span tagging 抽对象短语；点/框直接指定对象；查询送入冻结 **WildDet3D**，得 2D/3D box、深度图与 backbone 特征。
4. **Token 化：** **Image tokens** — 对 backbone 特征 16×16 池化，保留与检测框或 URDF 投影机械臂轮廓重叠的格子；**Detection tokens** — 由预测深度 + 3D box 采点云，经 **φ**（MLP + 八象限池化）得形状特征，并与 3D 位置/尺度拼接；**State history** — 当前与上一帧关节角。
5. **动作头：** 条件融合后，**12 层 MM-DiT** 四流（image / detection / state / noisy action chunk）联合注意力 + **flow matching** 预测 **H** 步绝对关节目标与夹爪；训练 **仅优化动作头**，**G 冻结**。
6. **系统用法：** 可 **端到端** 运行；也可作 **低层控制器**，由 Molmo2 等高层规划器通过多种模态下发子目标，支撑 **长时程与记忆依赖** 操作。
7. **RoboTwin 2.0：** 50 任务平均成功率 **55.3%**（Spatial Forcing **52.0%**）；场景随机化 **47.6%**（Abot-M0 **30.4%**）；动作策略 **仅用 clean-scene 示范** 训练。
8. **LIBERO-PRO：** 16 种扰动设定平均 **61%**（π₀.₅ **53%**）；目标 ** relocated / 新指定** 时增益最大。
9. **真机：** 双臂 **YAM** 视觉偏移下 **17/20** 成功（π₀.₅ **4/20**）；**Franka + Molmo2** 长时任务 step completion **64.7%** ID / **49.8%** OOD。
10. **消融（文内）：** 仅 detection tokens **16.0%**、仅 image tokens **20.3%**，完整模型 **46.8%**（Easy/Hard 子集口径以原文为准）— 几何与外观 **互补**。

**对 wiki 的映射**

- 实体页：[`wiki/entities/paper-grounded-action-model-3d-grounding.md`](../../wiki/entities/paper-grounded-action-model-3d-grounding.md)
- 项目页：[`sources/sites/grounded-action-model.md`](../sites/grounded-action-model.md)
- 官方仓：[`sources/repos/grounded-action-model.md`](../repos/grounded-action-model.md)
