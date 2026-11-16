# What World Model Does Active Object Search Need? A Study under Imperfect Detection｜BAMOS：不完美检测下的主动物体搜索

- **类型：** 研究原始论文
- **论文：** https://arxiv.org/abs/2611.15689 （v1）
- **HTML：** https://arxiv.org/html/2611.15689v1
- **PDF：** https://arxiv.org/pdf/2611.15689
- **代码：** https://github.com/gokyeongryeol/BAMOS
- **核对范围：** arXiv 原始版本的研究任务、核心方法与算法、主要实验评测、结论及限制；不将第三方摘要作为通读依据。没有在本次执行模型训练或真机复现。
- **通读：** 已完成
- **入库日期：** 2026-11-16
- **wiki：** [论文详情](../../wiki/entities/paper-bamos-search-2611-15689.md)

## 方法与评测核对

- **研究问题：** 利用检测和未检测事件更新目标位置后验，以信息增益决定观察视角与停止条件。
- **方法原理：** 从深度建立占据和可见性，结合 detector 的 mask 与漏检更新目标位置及尺寸 belief；再根据预计检测事件的信息增益选视角，以联合概率阈值决定停下而非首个命中。
- **原文结果：** Isaac Sim 与 HSSD 场景、多种误检/漏检条件的消融说明未检更新和可靠停止尤其重要。
- **局限与范围：** 假设静态场景、近似准确相机及机器人位姿；实际移动场景与位姿漂移尚未验证。
