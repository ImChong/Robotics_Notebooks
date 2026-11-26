# SuperGPC: training a humanoid autoregressive motion policy fast, on one GPU

- **原文：** <https://www.dollhouserobots.com/posts/super-gpc>
- **作者与机构：** Carlos Diaz-Padron，Dollhouse Robotics。
- **发布：** 2026-11-25，官方研发博客，非独立学术论文。
- **核对范围：** 通读官方博客正文、训练耗时表、真机演示说明、数据配比与限制；比对官方训练仓库的 README、规格与训练入口，以及公开模型卡的评测和导出接口。
- **通读：** 已完成
- **代码：** [DollhouseRobotics/super-gpc](https://github.com/DollhouseRobotics/super-gpc)，核查提交 `22b9a9d80f1431c9b7e161bcfc8c65f8b5ea0e8b`。
- **权重：** [G1-25-DOF-GPC](https://huggingface.co/dollhouserobotics/G1-25-DOF-GPC)，模型卡标注 Apache-2.0；原训练动捕数据未随权重分发。

## 原文要点

- 作者的三段对比：跟踪器由 132 h（4 GPU、PPO）降到 16.8 h；自回归先验由 65 h（4× RTX 5090）降到 12.2 h；摇杆适配器由 6.1 h 降到 46 min。新版三段均用一张 RTX 5090，合计报告约 29.7 h；不含另训的躺倒适配器。
- 训练对比并非受控实验：语料约 225 h → 56 h，机器人模型、模拟器及摇杆奖励亦不同；墙钟数排除停机间隔，也不是达到固定跟踪回报阈值的耗时。
- 文章真机片段标注 0:37 从摇杆切到躺倒、0:46 切回，同一先验上的两种 CoLA 各用约一秒混合；未报告跨多次执行的切换成功率。
- 模型卡将指标标为仿真结果，并说明 Stage 1 仅用训练语料评估。公开 ONNX 是 25 个身体控制自由度的接口，训练机器人配置还含两轴头部。

## 关联

- [SuperGPC 实体](../../wiki/entities/supergpc.md)
- [原始 GPC 论文节点](../../wiki/entities/paper-gpc-generative-pretrained-controllers.md)
