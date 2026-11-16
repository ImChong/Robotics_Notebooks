# Tutorial: Vision-Based Formation Control with Differential-Drive Robots（arXiv:2611.11164）

> 来源归档（paper）

- **标题：** Tutorial: Vision-Based Formation Control with Differential-Drive Robots
- **作者：** Christian Marowski, Byung-Cheol Min（SMART Laboratory，Luddy School of Informatics, Computing, and Engineering, Indiana University Bloomington）
- **类型：** paper（教程，非算法新方法论文）
- **arXiv：** <https://arxiv.org/abs/2611.11164>
- **PDF：** <https://arxiv.org/pdf/2611.11164v1>（v1，2026-11-08，cs.RO）
- **代码：** https://github.com/SMARTlab-IU/formation-control-tutorial
- **入库日期：** 2026-11-11
- **一句话说明：** 用 ArUco 标记 + 顶视相机 + Python，在 HamsterS 差速小车上实现集中式环形编队控制的分步教程。
- **核对范围：** 读了 v1 PDF 全文 20 页（摘要、第 1-13 节、致谢、参考文献），含全部代码摘录、公式 (1)-(17)、表 1-2 与图 1-6 的图注及图内文字（经 pdftotext 抽取，未逐图目视像素）；本文无附录与补充材料。GitHub 仓库在本环境访问返回 403，README 与源码未核实（未运行任何代码）。
- **通读：** 已完成

## 开源状态

- 论文摘要与第 1、13 节声明仓库提供 Python 实现、安装说明与演示；仓库链接见上。本环境无法打开仓库，**仓库内容与许可证未核实**（入库日 2026-11-11）。

## 核心摘录

1. **系统（第 2 节）：** 差速小车 + 固定顶视 USB 相机 + 一台中央电脑；每车一个 ArUco 标记（示例字典 DICT_7X7_100，第 3.2 节）；集中式，任一所需标记丢失则全体停车；车载接近传感器做基础反应式避障。
2. **位姿估计（第 5 节）：** 位置为四角均值（式 1），朝向由上边中点减下边中点的向量求 atan2（式 2-3），全部在图像像素坐标（x 右、y 下，角度顺时针增）下。
3. **编队几何与控制（第 6 节）：** 圆上等距槽位 p*_i = R(cos(2πi/N), sin(2πi/N))（式 4），期望相对位移 d*_ij = p*_j − p*_i（式 6），误差 e_ij = (p_j − p_i) − d*_ij（式 8），控制向量 u_i = Σ_{j≠i} e_ij（式 11）。第 6.3 节明确指出一致性文献的收敛 / 时延结论「不建立」本相机式 turn-then-drive 实现的收敛或时延容忍。
4. **差速运动控制（第 7 节）：** 先转后走；|e_θ| 超过 ANGLE_THRESHOLD 原地转，转速 min(KP_ANGLE·|e_θ|, 40)；否则前进，速度 min(KP_DISTANCE·‖u_i‖, 40)；‖u_i‖ ≤ DISTANCE_THRESHOLD 时停（式 12-16）。示例参数（第 4.1 节）：N=4，R=100 像素，KP_DISTANCE=0.7，KP_ANGLE=0.5，DISTANCE_THRESHOLD=10 像素，ANGLE_THRESHOLD=30 度，OBS_THRESHOLD=70，OBS_SPEED=50。
5. **评估（第 10 节、图 1）：** 平均成对编队误差 E(t)（式 17）；图 1 为四台真机一次运行的 0-7 s 定性序列，论文自称「qualitative illustration」；**无定量结果表，无多次重复、对比基线或成功率统计**。
6. **局限（第 12 节）：** 像素坐标而非物理单位（无标定）；集中式 + 固定相机 + 全部标记须可见；槽位固定、无动态重分配；停止条件基于合向量幅值而非逐对误差界；无无碰撞路径规划与显式车间距离约束；「减小编队误差不保证无碰撞或在所有工况收敛」。
7. **成本与替代（第 13 节）：** HamsterS 约 100 美元 / 台（写作时）；无实体机器人者可用 Georgia Tech 的 Robotarium（需改用其接口）。
8. **AI 使用声明（致谢）：** 生成式 AI 用于语言润色与图表准备，作者称已审核。

## 对 wiki 的映射

- [paper-formation-control-tutorial](../../wiki/entities/paper-formation-control-tutorial.md)
