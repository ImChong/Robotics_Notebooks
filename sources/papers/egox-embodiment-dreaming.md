# EgoX：Egocentric Cross-Embodiment Manipulation with Embodiment Dreaming

> 来源归档（paper / 官方原始论文）

- **中文标题：** EgoX：借助 Embodiment Dreaming 的第一视角跨本体操作学习
- **英文标题：** EgoX: Egocentric Cross-Embodiment Manipulation with Embodiment Dreaming
- **论文 PDF：** https://egox-dream.github.io/assets/egox.pdf
- **项目页：** https://egox-dream.github.io/
- **官方代码：** https://github.com/egox-dream/egox
- **机构：** Carnegie Mellon University、Bosch Center for Artificial Intelligence、University of Texas at Arlington、UC Berkeley、NVIDIA
- **年份：** 2026
- **核对日期：** 2026-11-13
- **核对范围：** 官方 PDF（37 页，项目站 2026-11-12 更新版本）全文，含封面、摘要、§1 引言、§2 相关研究、§3 方法与公式 (1)–(13)、§4 全部实验及图 5–9、§5 结论与局限、参考文献、附录 A.1–A.11（算法 1、表 1–7、训练参数、坐标系/控制器、数据统计、成功率和仿真协议）；同时对照项目主页、README 及模型/训练/部署源码。PDF 由 Wolfram PDF PagePlaintext 逐页读取，图表数值与表格配合原文图注核对；未执行 GPU/仿真复现。
- **通读：** 已完成

## 核心信息（已依据论文正文及附录复核）

1. 统一模态坐标：躯干/速度在 root frame，腕姿态在 egocentric frame，末端关键点在 wrist frame；关节状态保留本体自身空间。
2. MXT+：按本体及观测模态设置独立 tokenizers，以共享 Transformer trunk 汇聚；动作模态通过各自 action experts 输出动作块。
3. Embodiment Dreaming：训练时预测未来第一视角图像、机器人与末端关节状态的潜在特征，采用 EMA 教师编码器及 embodiment tokenizer 形成训练目标；部署不执行 Dreaming。
4. 5 项操作任务、6 种机器人（5 真机+1 仿真人形），共 13 个任务–本体组合；人类示范另用于预训练。
5. 跨本体预训练使平均归一化任务进度从 0.63 升到 0.80；OOD 从 0.53 升到 0.77。与「任务成功率」及「毫米级误差」不是一个指标。
6. Dreaming 独立消融在 7 个设置上将平均归一化进度从 0.63 提升至 0.78；Scoop 跨本体潜空间 GW 距离均值下降 38.52%，而跨本体预训练本身令其下降 21.30%（§4.2、图 7/9）。
7. **成功率（不同于进度）：** 附录 A.10 / 图 20：跨本体预训练平均成功率 **0.35 → 0.56**，OOD **0.18 → 0.52**，ID **0.44 → 0.58**；Dreaming 的独立消融平均成功率 **0.39 → 0.62**。
8. **训练集规模：** 附录 A.8 / 表 6：**4,851** 条示范轨迹、共 **34.38 小时**（人类 2,396 条 / 9.26 h；五真机共 1,706 条；仿真 G1 749 条 / 7.51 h）。
9. **关键超参：** 附录 A.3 / 表 1：30 Hz 示范，动作块 60 帧（2 s），Dreaming 未来偏移 **{20,40,60}** 帧，8 个本体潜变量查询 token，EMA 衰减 **0.999**，Dreaming 权重 **0.3**；预测余弦方向损失与幅值 smooth L1。
10. **控制执行：** 附录 A.7 / 表 5：上层动作目标 30 Hz，机器人专用控制器为 50–400 Hz；需坐标变换、IK/手指重定向、机体稳定控制等独立模块。推理保留学生 embodiment tokenizer，移除 DreamHead 和 EMA 目标教师。

## 开源与复现限制

- 项目页公开 GitHub；仓库中存在模型、训练、微调、仿真评测代码入口。
- 官方 README 说明数据集和 VR 示范采集流水线尚未发布，需自行提供兼容示范和 checkpoint；未实际运行实验。
- **原文核对已完成：** 详见 §3.1–3.4、§4.1–4.2、附录 A.1–A.11；未来仍需跟踪数据集/权重发布与代码真实运行复现。

## 对 Wiki 的映射

- [EgoX 主节点](../../wiki/entities/paper-egox-mxt-plus.md)
- [EXE 方法对照](../../wiki/entities/paper-exe-cross-embodiment.md)

## 来自原文的局限与评价读法

- **数据采样偏差：** 附录 A.8、§5 显示各本体数据量很不均衡（LocoMan-Dex2 仅 37 条、0.35 小时），训练仍按本体等量抽取 batch，小样本本体可能被过采样。
- **OOD 的定义：** 相对目标本体/目标任务微调示范中没有出现的物体、颜色、位置或布局；**其他本体预训练数据可能包含这些条件**，不等于所有训练数据未见。
- **评价区分：** 归一化进度计入中间阶段部分奖励，成功率为整项任务完成的试验占比；附录 A.9 定义 P=(s+b)/Smax，附录 A.10 单独给出成功率。
- **真实部署约束：** 论文 §5 指明新本体需手动定义模态对应、坐标变换和低层控制接口；当前论文仅覆盖一种仿真人形与两种厨房仿真任务。
