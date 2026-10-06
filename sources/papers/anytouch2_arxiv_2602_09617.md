# AnyTouch 2 — 论文来源归档

## 书目信息
- **题名**：AnyTouch 2: General Optical Tactile Representation Learning For Dynamic Tactile Perception
- **作者**：Ruoxuan Feng, Yuxuan Zhou, Siyu Mei, Dongzhan Zhou, Pengwei Wang, Shaowei Cui, Bin Fang, Guocai Yao, Di Hu
- **发表**：ICLR 2026；arXiv:2602.09617
- **论文**：https://arxiv.org/abs/2602.09617
- **会议页面**：https://proceedings.iclr.cc/paper_files/paper/2026/hash/073c8584ef86bee26fe9d639ec648e28-Abstract-Conference.html
- **项目页**：https://gewu-lab.github.io/AnyTouch2/
- **代码**：https://github.com/GeWu-Lab/AnyTouch2

## 摘要与贡献
AnyTouch 2 面向动态触觉感知提出 ToucHD 数据金字塔和通用光学触觉表示学习方法。ToucHD 汇总 2,426,174 个接触样本，覆盖仿真接触、真实操作和力配对数据。模型训练目标组合视频掩码重建/帧差、语义与物体/跨传感器匹配，以及力和力变化预测；评测包括静态属性、动态物理属性、跨传感器泛化和真实机器人操作。

ToucHD 的公开划分为 Sim 1,118,896 帧、Mani 584,842 帧、Force 722,436 个触觉-力样本。Sim 部分覆盖 5 类传感器、6 种原子动作和 1,043 个物体；Mani 覆盖 46 项操作任务。各子集样本构成和传感器不可视为完全同质的数据。

## 方法拆解
1. **动态层级数据**：从受控按压、滑动/旋转、原子接触动作延伸到真实操作和力监督，显式扩大动态接触覆盖。
2. **时序掩码目标**：重建触觉视频帧及相邻帧差，保留接触变化信息。
3. **跨语义与跨传感器目标**：用语义描述、同物体匹配和跨传感器配对约束表示。
4. **力监督**：预测接触力及其变化，令表示带有与交互动力学相关的信号。
5. **由静到动的评测**：包括静态属性、滑移/布料等动态属性、传感器迁移和抓取、擦白板、USB 插入、芯片移动等任务。

## 证据边界与开放情况
官方项目资料将其定位为通用动态光学触觉表征，而非端到端机器人策略。仓库目前包含数据预处理和 Sparsh 评估等代码；README 仍将 real-world code 标为待补全。公开模型权重与部分 ToucHD 数据需按 Hugging Face 页面流程申请/提交信息；力数据页也存在访问表单。README 列出的各评测数值应连同其任务、传感器和输入帧设置阅读，不能直接当作跨任务统一分数。

## 对应知识节点
- 论文详情：[paper-anytouch2](../../wiki/entities/paper-anytouch2.md)
- 项目详情：[project-anytouch2](../../wiki/entities/project-anytouch2.md)
- 前作：[AnyTouch 论文](../../wiki/entities/paper-anytouch.md)、[AnyTouch 项目](../../wiki/entities/project-anytouch.md)
- 主题：[触觉感知](../../wiki/concepts/tactile-sensing.md)、[视触觉融合](../../wiki/concepts/visuo-tactile-fusion.md)
