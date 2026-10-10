# H-JEPA: End-to-End Learning of Hierarchical World Models for Visual Planning（arXiv:2610.06805）

> 来源归档（paper · H-JEPA）

- **标题：** H-JEPA: End-to-End Learning of Hierarchical World Models for Visual Planning
- **类型：** arXiv 预印本
- **arXiv：** <https://arxiv.org/abs/2610.06805>
- **HTML：** <https://arxiv.org/html/2610.06805>
- **项目页：** <https://h-jepa.com/>
- **代码：** <https://github.com/kevinghst/H-JEPA>
- **作者：** Wancong Zhang、Basile Terver、Michael Rabbat、Yann LeCun、Randall Balestriero
- **机构：** NYU、Advanced Machine Intelligence（AMI Labs）、INRIA Paris、Brown University
- **版本：** v1，2026-10-05
- **论文许可：** arXiv 页面标注 CC BY 4.0
- **入库日期：** 2026-10-10
- **一句话说明：** 端到端训练多个不同潜空间、不同时间尺度的 action-conditioned JEPA world models，并从高层到低层逐级生成子目标与原始动作。

## 方法与评测摘要

H-JEPA 在每个层级分别学习观测编码器、动作编码器和潜状态预测器，并用 SIGReg 约束表征避免坍缩。层级联合训练，高层在更稀疏的时间尺度预测更抽象的潜状态。规划时先从顶层朝目标优化，再把上层预测轨迹作为下层子目标，最终由第一级生成 primitive actions；闭环评测中执行一段动作后重新观测并规划。

论文评测包括 Visual AntMaze、FourRoom Distractors、Push-T、OGBench Cube，以及 DROID 真机遥操作视频。三层 H-JEPA 在论文指定的 Visual AntMaze 规划设置中报告 **73.3% ± 3.5%** 成功率；单层 LeWM 基线为 **18.0% ± 3.5%**，且分层方法在 success–compute 曲线上表现更好。DROID 结果使用离线末端执行器路径 Fréchet fidelity，不是实体机器人闭环成功率。

## 阅读边界

- H-JEPA（2610.06805）与 **Hamiltonian JEPA: Action-Conditioned World Models with an Inherited Control State**（2609.33497）是不同论文；缩写相同，不应合并。
- H-JEPA 与 HWM（2604.03208）都进行分层潜空间规划；关键差异是 H-JEPA 为不同层学习独立潜表示，HWM 的各层使用共享潜空间。
- 论文的成功率和规划计算对比依赖其任务、数据、层数、优化器预算及评测协议，不能外推为通用机器人性能。
- DROID 实验为离线规划保真度评估；作者也将实体机器人闭环验证列为后续工作。

## 对 wiki 的映射

- [H-JEPA 论文与项目详情](../../wiki/entities/paper-h-jepa-visual-planning.md)
- [Model-Based RL](../../wiki/methods/model-based-rl.md)
- [HWM：Hierarchical Planning with Latent World Models](../../wiki/entities/paper-hwm-latent-world-model-planning.md)
