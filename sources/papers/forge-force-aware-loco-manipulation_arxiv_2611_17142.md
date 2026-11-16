# FORGE：力感知多粒度人形移动操作（2611.17142）

> 论文一手原文归档

- **英文题目：** FORGE: Force-Aware Multi-Granularity Control for Efficient Humanoid Loco-Manipulation
- **arXiv：** https://arxiv.org/abs/2611.17142
- **HTML 全文：** https://arxiv.org/html/2611.17142v1
- **PDF：** https://arxiv.org/pdf/2611.17142v1
- **机构：** 上海人工智能实验室、清华大学、浙江大学、上海交通大学、香港中文大学
- **官方项目：** https://bb0928.github.io/forge/
- **官方代码：** https://github.com/bb0928/FORGE
- **入库日期：** 2026-11-16
- **核对范围：** 通读并核对 arXiv v1 III 命令掩码/动量观测器/残差控制、IV 表 I/III/IV/V 和项目页，以及官方仓库 README 与 scripts；与用户提供的 29 篇 PDF 周报核对研究问题、实验口径与局限。另阅读官方 GitHub README 与实际入口脚本。
- **通读：** 已完成
- **一句话说明：** 四组命令（底座速度、足端平面、双腕位姿、力设定）按激活掩码动态生效；下身冻结 locomotion prior，残差网络负责上身与底座协调。腕臂动量观测器加仿真特权力监督的学习补偿估计低频外力，经虚拟目标更新改变腕位姿。

## 原文摘录（提炼）
1. **问题与动机：** 机器人接近物体、摆脚、操作腕端与长期接触需要不同精度命令；全身人体轨迹难采集，腕力传感器又不总可用。
2. **方法：** 四组命令（底座速度、足端平面、双腕位姿、力设定）按激活掩码动态生效；下身冻结 locomotion prior，残差网络负责上身与底座协调。腕臂动量观测器加仿真特权力监督的学习补偿估计低频外力，经虚拟目标更新改变腕位姿。
3. **实验：** 真机 20 次每方法的响应延迟 22 ms、足端 MPE 5.42 cm、腕 MPE 4.15 cm；仿真双臂夹箱平均法向作用力 8.52 N（不是实机力跟踪误差）。遮蔽消融有效 rollout 从 15.69±1.32 s 提升 19.46±0.52 s。跨载荷 0、0.5、1、2 kg 成功率 100/100/90/90%，各 20 次。
4. **边界与局限：** 估测为腕部低频等效三维力，不是通用六维力矩传感。官方代码仅公开 stage2 残差训练与仿真评估，基本策略训练和真机部署脚本尚在 TODO。

## 开源状态
官方仓库公开 scripts/train_stage2_residual.sh 和 scripts/test_res_pf.sh，以及 legged_gym、rsl_rl 子目录；README 将基础策略训练和真机部署列在未开源 TODO。

## 对 wiki 的映射
- [FORGE](../../wiki/entities/paper-forge-force-aware-loco-manipulation.md)
