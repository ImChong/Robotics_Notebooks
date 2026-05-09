# hipan

> 来源归档（ingest）

- **标题：** HiPAN / Hierarchical Posture-Adaptive Navigation
- **类型：** paper
- **来源：** IEEE RA-L（arXiv 预印本）
- **入库日期：** 2026-05-09
- **最后更新：** 2026-05-09
- **一句话说明：** KAIST 提出的四足机器人在非结构化 3D 环境中仅靠机载深度的分层导航框架，含 Path-Guided Curriculum Learning（PGCL）与姿态自适应低层控制接口。

## 核心论文摘录（MVP）

### 1) HiPAN: Hierarchical Posture-Adaptive Navigation for Quadruped Robots in Unstructured 3D Environments (Jeong et al., 2026, RA-L)
- **链接：** <https://arxiv.org/abs/2604.26504>
- **项目页：** <https://sgvr.kaist.ac.kr/~Jeil/project_page_HiPAN/>
- **核心贡献：** 双层 RL：高层从深度与相对子目标输出 5 维导航指令（平面速度 + 体高/横滚姿态），低层 posture-adaptive locomotion 跟踪；训练期用特权地图上的全局路径采样子目标做 **Path-Guided Curriculum Learning**，逐步拉长有效导航视界以减轻短视与局部极小；高低层均采用 teacher–student（PPO 教师 + DAgger 蒸馏），部署仅学生策略与机载深度，避免显式 3D 建图的开销。
- **对 wiki 的映射：**
  - [HiPAN（方法页）](../../wiki/methods/hipan.md)
  - [Locomotion](../../wiki/tasks/locomotion.md)
  - [Curriculum Learning](../../wiki/concepts/curriculum-learning.md)
  - [Sim2Real](../../wiki/concepts/sim2real.md)
  - [Privileged Training](../../wiki/concepts/privileged-training.md)

## 当前提炼状态

- [x] 摘要与贡献要点已写入
- [x] 已映射到 wiki 方法页与相关概念/任务页交叉引用
