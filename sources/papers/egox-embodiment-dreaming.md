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
- **核对范围：** 已完整阅读官方项目页 Method、Embodiments & Tasks、Results 以及预训练、数据构成、Dreaming、模块化与潜在几何各消融说明；阅读官方仓库 README、模型、训练及推理关键源码。论文 PDF 约 25 MB，当前网页读取工具因大小限制拒绝读取；未完成 PDF 正文及附录的逐页核对。
- **通读：** 未完成

## 核心信息（来自官方项目页及源码，待论文全文复核）

1. 统一模态坐标：躯干/速度在 root frame，腕姿态在 egocentric frame，末端关键点在 wrist frame；关节状态保留本体自身空间。
2. MXT+：按本体及观测模态设置独立 tokenizers，以共享 Transformer trunk 汇聚；动作模态通过各自 action experts 输出动作块。
3. Embodiment Dreaming：训练时预测未来第一视角图像、机器人与末端关节状态的潜在特征，采用 EMA 教师编码器及 embodiment tokenizer 形成训练目标；部署不执行 Dreaming。
4. 5 项操作任务、6 种机器人（5 真机+1 仿真人形），共 13 个任务–本体组合；人类示范另用于预训练。
5. 跨本体预训练使平均归一化任务进度从 0.63 升到 0.80；OOD 从 0.53 升到 0.77。与「任务成功率」及「毫米级误差」不是一个指标。
6. Dreaming 独立消融在 7 个设置上将平均进度从 0.63 提升至 0.78；Scoop 潜在空间跨本体 GW 距离均值下降 38.52%。

## 开源与复现限制

- 项目页公开 GitHub；仓库中存在模型、训练、微调、仿真评测代码入口。
- 官方 README 说明数据集和 VR 示范采集流水线尚未发布，需自行提供兼容示范和 checkpoint；未实际运行实验。
- **待办：** 阅读 PDF 全文、附录，复核项目页数值与方法公式；后续核对数据集与权重释放状态。

## 对 Wiki 的映射

- [EgoX 主节点](../../wiki/entities/paper-egox-mxt-plus.md)
- [EXE 方法对照](../../wiki/entities/paper-exe-cross-embodiment.md)
