# HandsOn 论文来源归档

## 元数据

- **题目：** HandsOn: A Large-Scale Simulation Benchmark for Generalizable Dexterous Manipulation
- **类型：** 论文（cs.RO）
- **作者：** Tianwei Luo、Yiming He、Hongjin Wu、Ting Pan、Debin Ren、Haorui Guo、Junlin Yang、Yuxin Zuo、Zhehao Kou、Kaisen Yang、Zhuoqun Lai、Yi Ding、Jiale Yu、Yuchen Zhang、Zijie Qiu、Haozhan Li、Ning Ding
- **机构：** 清华大学、北京邮电大学、NatureWill、北京大学
- **预印本：** https://arxiv.org/abs/2610.26765
- **版本 / 日期：** v1，2026-10-19
- **项目页：** https://llltttwww.github.io/HandsOn-website/（核查：[项目页归档](../sites/handson-project-page.md)）
- **代码：** https://github.com/TsinghuaC3I/HandsOn（归档：[仓库来源](../repos/handson.md)）
- **数据：** https://huggingface.co/datasets/Cyan666/HandsOn-Datasets（归档：[数据集来源](../datasets/handson_datasets.md)）

## 论文核心内容

HandsOn 提出面向跨具身灵巧操作的可组合仿真平台与 benchmark：支持 5 种机械臂、6 种灵巧手及约 200 个刚体、关节物体与可变形资产，并配套 66 项人工核验任务。Core-50 提供示范与闭环执行评测；Transfer-16 不提供目标任务示范，用于测量资产/关系迁移、技能组合与可变形操作。

论文与官方发布材料报告 18,000 条精选人类示范、30 Hz 多视角观测，以及多类策略的 benchmark 结果；最佳多任务 checkpoint 在 Core-50 上达到 29.1% task-macro success。GPT-6 控制器不使用 HandsOn 示范或 benchmark 微调，在 Core-50 / Transfer-16 分别达到 5.2% / 1.9%。这些成功率是特定评测协议下的闭环任务指标，不等同于动作预测误差。

## 开源与复现边界（2026-10-25）

- 官方项目页链接到代码仓库与 Hugging Face 数据；仓库公开仿真任务、评测协议和模型适配入口。
- 当前公开数据为 PhysX 后端采集版：18,000 条轨迹，15,178 train / 2,822 validation；Newton 后端数据标注为待发布。
- 官方仓库没有随附论文策略 checkpoint；端到端复现论文结果还需要取得对应模型权重并遵循模型各自的安装与推理环境。
- 仿真示范与定性真机 rollout 可支持研究和复现探索，但不能单独证明真实机器人安全性或完整 sim-to-real 泛化。

## Robotics_Notebooks 映射

- 项目实体：[HandsOn](../../wiki/entities/paper-handson.md)
- 项目页：[handson-project-page](../sites/handson-project-page.md)
- 仓库：[handson](../repos/handson.md)
- 数据集：[handson_datasets](../datasets/handson_datasets.md)
