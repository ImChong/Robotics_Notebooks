# Geometry-Aware 4D Video Generation for Robot Manipulation

> 来源归档（ingest · REALab 14 篇盘点）

- **标题：** Geometry-Aware 4D Video Generation for Robot Manipulation
- **类型：** paper
- **状态：** ICLR 2026
- **原始链接：**
  - arXiv：<https://arxiv.org/abs/2507.01099>
  - 项目页：https://robot4dgen.github.io/
- **代码：** https://robot4dgen.github.io/
- **作者：** Zeyi Liu, Shuang Li, Eric Cousineau, Siyuan Feng, Benjamin Burchfiel, Shuran Song
- **机构：** Stanford University; Toyota Research Institute
- **入库日期：** 2026-08-18
- **一句话说明：** Geometry-Aware 4D Video（ICLR 2026）：跨视角点图对齐监督的多视角一致 4D RGB-D 视频；无推理期相机位姿；位姿追踪恢复 EE 轨迹训策略。

## 核心论文摘录（MVP）

### 问题与贡献

- **摘录要点：** 用跨视角点图对齐监督 4D 视频生成，使多相机 RGB-D 未来帧时空几何一致，再经位姿追踪器提取末端轨迹训练操作策略。
- **对 wiki 的映射：**
  - [wiki/entities/paper-geometry-aware-4d-video-generation.md](../../wiki/entities/paper-geometry-aware-4d-video-generation.md)

### 方法与结果（归纳）

- **方法：** 双视角 RGB-D 输入 → U-Net 预测未来点图与 RGB；训练期 cross-view pointmap 对齐；推理不需相机外参。
- **评测：** 仿真操作任务新视角泛化；长时程双臂任务时空对齐优于基线。

## 当前提炼状态

- [x] 公众号盘点 + arXiv/项目页交叉核对
- [x] wiki 实体页：`wiki/entities/paper-geometry-aware-4d-video-generation.md`
