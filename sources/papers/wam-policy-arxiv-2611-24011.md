# WAM-Policy — 四足前腿操作的世界动作模型与在线蒸馏
- **类型：** paper / world-action-model / loco-manipulation
- **论文：** https://arxiv.org/abs/2611.24011
- **HTML：** https://arxiv.org/html/2611.24011
- **项目：** https://aigeeksgroup.github.io/WAM-Policy/
- **代码：** https://github.com/AIGeeksGroup/WAM-Policy
- **模型：** https://huggingface.co/AIGeeksGroup/WAM-Policy
- **基准数据：** https://huggingface.co/datasets/AIGeeksGroup/QLM-Bench
- **核对范围：** 论文架构、OPSD/M-OPD 机制、Stage1/Stage2 评测和真机限制；对照官方 README、训练/评测脚本与部署入口。
- **通读：** 已完成
- **入库日期：** 2026-11-23

## 核心摘录
1. 更换机械臂 WAM 的视觉/动作接口，适配四足前腿作为末端的移动操作。
2. Stage 1 的 OPSD 使用更多采样步的同一教师，监督同分布低步数学生。
3. Stage 2 对任务专用教师进行 RL 后训练，通过 M-OPD 融合学生。
4. QLM-Bench 10 任务、1000 条合成示范；Stage1 教师 77.0%，OPSD 学生 76.7% 且采样约快 14 倍，Stage2 仿真 85.0%。
5. Unitree Go2 Air 真机的 Stage1 学生为 7/30；不能将仿真结果外推。
6. 主分支 Apache-2.0，部署目录及 QLM-Bench 另有 CC BY-NC 4.0。

## 对 wiki 的映射
- [WAM-Policy 独立论文实体](../../wiki/entities/paper-wam-policy.md)
- [官网](../sites/wam-policy-project.md)
- [源码](../repos/wam-policy.md)
- [移动操作纵深](../../roadmap/depth-loco-manipulation.md)
