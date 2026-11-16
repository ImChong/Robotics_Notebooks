# 扩展社会力、MPC 与 CBF 的人形群体重定位（2611.14853）

> 论文一手原文归档

- **英文题目：** Can I Join You? Extended Social Force Based Model with Predictive Control and Barrier Functions for Humanoid Group Repositioning Amidst Obstacles
- **arXiv：** https://arxiv.org/abs/2611.14853
- **HTML 全文：** https://arxiv.org/html/2611.14853v1
- **PDF：** https://arxiv.org/pdf/2611.14853v1
- **机构：** 西蒙菲莎大学（Simon Fraser University）
- **入库日期：** 2026-11-16
- **核对范围：** 通读并核对 arXiv v1 III 场景定义、IV ESFFM/MPC/CBF、V 用户研究、VI 8 条件实验和偏好、VII 结论；与用户提供的 29 篇 PDF 周报核对研究问题、实验口径与局限。
- **通读：** 已完成
- **一句话说明：** ESFFM 加入障碍排斥、群心牵引、成员间排斥与速度衰减；MPC 生成前瞻加速度；CBF 对机器人间距、障碍边界、速度和加速度建立约束，由 QP 选择最接近名义值的安全加速度。

## 原文摘录（提炼）
1. **问题与动机：** 社交机器人在人群加入、离开或穿过时要保持社交队形与障碍安全；纯社会力模型会在狭窄空间震荡碰撞。
2. **方法：** ESFFM 加入障碍排斥、群心牵引、成员间排斥与速度衰减；MPC 生成前瞻加速度；CBF 对机器人间距、障碍边界、速度和加速度建立约束，由 QP 选择最接近名义值的安全加速度。
3. **实验：** 8 组环境/群体条件中，带 CBF 的两个控制器在 7/8 条件未检测碰撞，ESFFM+MPC 为 5/8，原始 SFFM 为 1/8。40 人视频用户研究对若干拥挤布局更偏好组合控制；每配置在每条件的客观消融仅有一段预录外部轨迹。
4. **边界与局限：** 外部遥控参与者不受群体 CBF 约束，无法证明整个交互安全；个人空间违规率与几何碰撞并不同步；研究主要为视频/群体对照。

## 开源状态
未从本次原文核实到作者发布的可运行算法代码入口；有第三方框架引用不表示本论文完整开源。

## 对 wiki 的映射
- [Can I Join You?](../../wiki/entities/paper-social-force-mpc-cbf-group.md)
