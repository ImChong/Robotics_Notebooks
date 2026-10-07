# Continual Humanoid Motion Learning 来源归档

- **论文：** https://arxiv.org/abs/2610.04231
- **HTML v1：** https://arxiv.org/html/2610.04231v1
- **PDF：** https://arxiv.org/pdf/2610.04231
- **代码与配置：** https://anonymous.4open.science/r/continual-humanoid-learning-35D3
- **作者：** Zhewen He, Hao Huang, Geeta Chandra Raju Bethala, Chong Yu, Tao Chen, Anthony Tzes, Yi Fang
- **机构：** NYU Abu Dhabi；Fudan University
- **论文类型：** arXiv preprint，2026-10-03

## 方法摘录

Similarity-guided LoRA-PNN 采用任务增量 progressive neural network：每个新任务增加 actor column，冻结之前列，通过 lateral connections 迁移表示；新列从动作最相似旧列继承，并按相似度决定 LoRA rank。动作相似度分两级：局部窗口用 DTW 对齐，窗口/动作集用均匀质量 optimal transport 比较。特权教师向学生蒸馏，学生从最近 10 帧可部署关节位置、速度与动作历史中推断控制。

任务为 walk、run、jump、dance、fight、fall-and-up，使用 LAFAN1 与 Kungfu 运动数据。每项 50k iterations、1,024 并行环境、每任务 100 个评测 episode；实机学生策略部署在 Unitree G1。

## 结果口径

Similarity-guided linear-rank variant：AA=0.945、AIA=0.964、FWT=0.125；论文对照 KungfuBot2 的 FWT 为 0.079。消融中最多节省 94.5% 可训练参数、40.8% 时间。Isaac Gym→MuJoCo 成功率 96.13%；G1 上 30 个动作、每个 10 次尝试，成功率 90.33%。

## 限制

- 任务增量协议提供当前 task identity，不等于无标识技能发现。
- 参数隔离不代表容量恒定；PNN 列/连接仍随任务增加。
- 实机为动作跟踪部署，不是开放场景自主持续学习。
- 代码托管于匿名项目页；复现步骤与授权以该页实际说明为准。
