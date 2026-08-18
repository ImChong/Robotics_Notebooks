# One Demo Is Worth a Thousand Trajectories: Action-View Augmentation for Visuomotor Policies

> 来源归档（ingest · REALab 14 篇盘点）

- **标题：** One Demo Is Worth a Thousand Trajectories: Action-View Augmentation for Visuomotor Policies
- **类型：** paper
- **状态：** CoRL 2025
- **原始链接：**
  - arXiv：<https://arxiv.org/abs/2606.19586>
  - 项目页：https://chuerpan.com/1001-demos.github.io/
- **代码：** https://chuerpan.com/1001-demos.github.io/
- **作者：** Chuer Pan, Litian Liang, Dominik Bauer, Eric Cousineau, Benjamin Burchfiel, Siyuan Feng, Shuran Song
- **机构：** Stanford University; Columbia University; Toyota Research Institute
- **入库日期：** 2026-08-18
- **一句话说明：** Action-View Augmentation（CoRL 2025）：单次鱼眼手眼示范→鱼眼 3DGS 场景编辑+轨迹优化→千条增广轨迹；提升 OOD 位姿/避障成功率。

## 核心论文摘录（MVP）

### 问题与贡献

- **摘录要点：** 从单次真实手眼示范重建鱼眼 3DGS 场景，用轨迹优化生成千条物理可行、视角一致的动作–图像对，增广 visuomotor 训练。
- **对 wiki 的映射：**
  - [wiki/entities/paper-action-view-augmentation.md](../../wiki/entities/paper-action-view-augmentation.md)

### 方法与结果（归纳）

- **方法：** 单次扫描+示范 → 鱼眼适配 3DGS → 轨迹优化生成无碰撞路径 → 多视角鱼眼渲染。
- **评测：** 仿真与真机多操作任务；同场景与增广障碍场景成功率均提升。

## 当前提炼状态

- [x] 公众号盘点 + arXiv/项目页交叉核对
- [x] wiki 实体页：`wiki/entities/paper-action-view-augmentation.md`
