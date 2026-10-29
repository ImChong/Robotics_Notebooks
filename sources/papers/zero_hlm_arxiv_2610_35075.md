# Zero-HLM（arXiv:2610.35075）

> 来源归档（paper）

- **标题：** Zero-HLM: Generalizable Humanoid Loco-Manipulation without Real Robot Data
- **作者：** Dianyi Yang, Mengya Liu, Zhumei Wang, Yu Liu, Xiaochen Zhang, Yixuan Li, Le Ma, Yuhang Yang, Weiqi Huang, Wei Liang, Baoxiong Jia, Siyuan Huang
- **类型：** arXiv 预印本，v1 2026-10-26
- **论文全文：** <https://arxiv.org/html/2610.35075v1>
- **项目页：** <https://young-bit.github.io/zero-hlm/>
- **入库日期：** 2026-10-29
- **开放状态：** 项目页本次访问失败，代码和数据是否公开待核实，不推断为未开源。

## 核心摘录（论文归纳）

- 手机图像构建 Gaussian 外观参考，并由智能体配置网格碰撞体、关节及物理参数；物理场景与视觉表示共用尺度坐标系（§3.1）。
- 每任务 15 条仿真遥操作成功示范；对目标位置重定向操作，A* 与 SONIC 规划行走，再用 SONIC 闭环执行。失败后可按机器人位姿、对象平面误差及手腕高度修正并重试；只保留通过任务、稳定性与碰撞检查的轨迹（§3.2）。
- 在 3 个 G1 任务、9 个技能阶段生成 4,800 条轨迹；视觉来自 Gaussian 背景与仿真机器人、对象合成，动作包括 64 维 SONIC latent 和 2 维手指命令（§3.3、§4.1）。
- Base 真机评估每任务 20 次：GR00T N1.7 每任务 800 条仿真训练的完整任务均值 66.7%；100 条真机训练为 46.7%；同为 100 条时仿真为 43.3%（表 8）。
- Expanded 五组配置合计 100 次试验，扩展仿真训练与 Base 真机数据训练分别成功 69 和 23 次；覆盖范围与训练数据量不同（表 9）。

## 对 wiki 的映射

- [Zero-HLM](../../wiki/entities/paper-zero-hlm.md)
