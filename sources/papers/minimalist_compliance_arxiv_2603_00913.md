# Minimalist Compliance Control

> 来源归档（ingest · REALab 14 篇盘点）

- **标题：** Minimalist Compliance Control
- **类型：** paper
- **状态：** RSS 2026
- **原始链接：**
  - arXiv：<https://arxiv.org/abs/2603.00913>
  - 项目页：https://minimalist-compliance-control.github.io/
- **代码：** 截至入库日项目页未列公开 GitHub（或 N/A 综述）
- **作者：** Haochen Shi, Songbo Hu, Yifan Hou, Weizhuo Wang, C. Karen Liu, Shuran Song
- **机构：** Stanford University
- **入库日期：** 2026-08-18
- **一句话说明：** Minimalist Compliance（RSS 2026）：电机电流/电压+雅可比估计外力→任务空间导纳；跨 ARX/G1/LEAP；可插 VLM/IL/模型基策略；项目页未列 GitHub。

## 核心论文摘录（MVP）

### 问题与贡献

- **摘录要点：** 不用力传感器、不用学习：用现成电机电流/电压与雅可比估计外力，驱动任务空间导纳控制，即插即用接到任意高层策略后。
- **对 wiki 的映射：**
  - [wiki/entities/paper-minimalist-compliance-control.md](../../wiki/entities/paper-minimalist-compliance-control.md)

### 方法与结果（归纳）

- **方法：** 电机扭矩模型 + 雅可比映射估计外力矩 → 弹簧–质量–阻尼导纳更新位姿参考；与 VLM/扩散/模型基策略正交。
- **评测：** 机械臂、LEAP 灵巧手、两台人形：擦白板、画图、煎蛋、球体旋转等；相对 RL 柔顺基线更稳跟踪且力更合理。

## 当前提炼状态

- [x] 公众号盘点 + arXiv/项目页交叉核对
- [x] wiki 实体页：`wiki/entities/paper-minimalist-compliance-control.md`
