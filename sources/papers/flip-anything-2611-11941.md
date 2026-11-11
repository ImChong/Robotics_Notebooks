# Flip Anything with Agentic Robotics Self-Learning

- **中文标题：** 通过智能体式机器人自主学习翻转物体
- **论文：** <https://arxiv.org/abs/2611.11941>
- **项目：** <https://berkeleyautomation.github.io/flipanything/>
- **核验日期：** 2026-11-11
- **核对范围：** arXiv:2611.11941v1（UC Berkeley，2026-11-09）短论文全文：Abstract、§1 Introduction、§2 Method、§3 Results、Figures 1–5、References [1]–[12]；该 v1 未单列附录；对照项目主页。
- **通读：** 已完成

## 核心方法

任务目标是利用双臂将不同形态的物体翻转到目标面。编码智能体在仿真中编写/修改运动原语、运行测试并利用结果迭代，形成可重复使用的技能，而非每个物体由人工手工重写控制器。

## 官方报告

- 仿真学习过程在作者设置中将成功率从 **24.7%** 提高到 **95.0%**（342/360）。
- 研究还展示部分真机翻转迁移；仿真中的成功率不等于真实世界的通用成功率。

## 公开状态

项目页可读，但本次未确认独立且可运行的官方代码仓及具体脚本入口；不能声称完全开源、也不能根据泛化框架虚构源码时序图。

## 全文核验补充（2026-11-11）

- **具体方法**：Claude Fable **5.1**（不是 5.2）作为 coding agent，以 GaP（Graph-as-Policy）harness 驱动 RoboSimStudio；模拟端使用 Newton rigid-body、MuJoCo solver，输入 24 个物体 CAD 与 YAM 双臂 URDF。得到的图状代码综合 RGB-D 和关节本体状态，选择双臂交接、支撑/推倒、推压、重抓和恢复策略。
- **迭代评测**：24 对象 × 15 次 = 360 episodes。初版 **89/360 = 24.7%**；引入 stand-and-topple/push 后 **290/360 = 80.6%**；最佳调优版本 **342/360 = 95.0%**。论文将测试结果反馈用于下一轮代码修改，不是严格隐藏的 held-out 独立测试。
- **真实迁移**：三种仿真中未见物体，夹具 **2/3**、蓝色胶带 **3/3**、折叠螺旋螺丝刀 **1/3**，没有针对物体单独调参。
- **纠错**：原先归档中的“冻结 held-out 90.8%（109/120）”在论文 v1 正文及 References 中均找不到，现删除；不得将其记作已证实结果。
- **局限**：真机只有 9 次、3 种物体的探索性测试；没有足够证据外推为任意对象成功率。

## 对 wiki 的映射

- [Flip Anything 主节点](../../wiki/entities/paper-flip-anything-agentic-self-learning.md)
