# SCRUB：无机械臂四足的接触式表面坐标系估计（arXiv:2611.06649）

> 来源归档（paper）

- **标题：** SCRUB: Contact-Based Surface-Frame Estimation for Force-Motion Control with an Armless Quadruped
- **作者：** Sourav Selvaraj, Juan Pablo Espinosa Chessal, Mahdi Agheli
- **机构：** Department of Robotics Engineering, Worcester Polytechnic Institute（伍斯特理工学院）, Worcester, MA, USA（论文首页脚注；通讯作者 Mahdi Agheli）
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2611.06649>（v1，2026-11-04，cs.RO）
- **PDF：** <https://arxiv.org/pdf/2611.06649v1>
- **代码 / 项目页：** 论文未给出（全文只引用第三方 OCS2、legged_control 仓库）
- **投稿状态：** 首页脚注写明已投 IEEE，待发表
- **入库日期：** 2026-11-09
- **核对范围：** 读了 arXiv v1 PDF 全文（pdftotext 抽取）：摘要、I 引言、II 系统与假设（含 Assumption 1–3）、III 表面坐标系估计（含 Proposition 1）、IV 控制、V 实验与讨论（Table III、IV，Fig. 5、6 的说明与数据）、VI 结论、致谢、参考文献 [1]–[34]。论文无附录或补充材料章节。图的像素内容（Fig. 1–6 的曲线）未逐幅目视，只读了图注、轴数字和正文给出的数值；参考文献只核对了条目，未读被引论文。未运行任何代码（论文无公开代码）。
- **通读：** 已完成
- **一句话说明：** 用 Kalman 滤波融合「与摩擦系数无关的力线索」和「按步态周期平均、扣除按压速度的速度线索」估计表面法向，让无臂 Go1 四足用躯干固定工具在未建图的近竖直墙面上横移、绕圆柱滑动。

## 核心摘录

1. **问题（摘要、I）：** 擦拭 / 打磨 / 检测需要沿表面法向压、沿表面滑；法向在未建图表面上未知。已有无需摩擦系数的法向估计器都假设瞬时工具速度沿表面，步态行走会违反该假设（步态纹波 + 控制器进退工具）。
2. **方法（III）：** 乘性 EKF（MEKF），状态是估计法向到真法向的小倾角 θ=[θu θm]，1 kHz 运行。力行：力在估计滑动平面外的分量 r_f=-m̂ᵀf_e，观测横滑倾角 θm；Proposition 1 在摩擦与运动方向相反、速度在真法向与滑动方向所在平面内的条件下，r_f 为零当且仅当 m̂ᵀn=0，与摩擦系数无关（式 2）。速度行：减去指令按压速率 v_n^c 后，在一个步态周期窗口 Tw=0.5 s 内对残差与灵敏度分别取均值（式 4），观测沿滑倾角 θu。
3. **控制（IV）：** 质心 MPC（100 Hz，1.0 s 视域，27 状态 / 27 输入）把工具反力 λ 作为状态，带一阶滞后 τF=0.05 s、拖曳代价与副法向锥 µb=0.02；深度环用弹簧模型加力误差积分调参考深度（式 6）；WBC 为逆动力学 QP，把规划反力前馈为外力。
4. **平台（V-A）：** Unitree Go1（原装质量 12 kg），50 mm PETG 工具体，工具尖在躯干系 [0.349 0 0] m，串联 IIT FT17 六维力传感器；系留的离线 PC 上 MPC 单次求解约 8.3 ms；表面为向后倾 10° 的 OSB 平墙与半径 0.303 m 的石膏板圆柱；爬行步态，周期 0.5 s。
5. **主结果（V-B，Table III，每格 5 次取中位数）：** 12 mm/s、10 N 任务载荷下，墙 5/5、柱 5/5 完成，无跌倒；法向 RMSE 2.29° / 2.48°；原始力 RMSE 2.97 N / 2.06 N；±2 N 带内占比 63.2% / 81.2%；柱轨道扫过 360.1°。速度与载荷扫描（8/12/16 mm/s × 5/10/15 N，两个表面各 5 次，共 90 次）完成 89 次，唯一失败是 5 N 的一次圆柱跌倒；5 N 墙面 15 次因起点更靠前被操作员在约 0.95 m 的可用端点停下并计为完成。
6. **对照与消融（V-C，Table IV，圆柱）：** 固定法向、摩擦补偿法向、瞬时（逐样本）速度行、去速度行、去力行，五种估计器变体轨道完成数均为 0/5（扫过角中位数 16.1°、33.0°、214.0°、24.5°、260.2°）；Scalar MPC 2/5、No tool force 0/5。摩擦补偿基线只是对 [14] 摩擦补偿部分的重实现。
7. **作者承认的局限（V-B 讨论、VI）：** 速度与 µk 取自 SCRUB 自己的运行且 SCRUB 的块先跑，可能偏向 SCRUB；估计器假设样本独立，可能高估自身精度；无里程计漂移状态；力行噪声未含步态纹波速度；未检查力行增益保持为正；估计器在 MPC 之外运行，MPC 反力无接触力矩，工具模型忽略保险杠剪切；两个规划器消融同时改变了 WBC 前馈，不能单独说明 MPC 反力模型的作用。未来工作：更快步态与 16 mm/s 以上速度、变曲率、非平面路径、变化摩擦、MPC 内法向估计、接触力矩、视觉预览、WBC 关节阻抗。
8. **其他：** 致谢称作者用 Claude（Anthropic）算汇总统计、写绘图代码、查参考文献并协助调试机器人软件。

## 与周报说法的核对

见 [paper-scrub-surface-frame-estimation-armless-quadruped](../../wiki/entities/paper-scrub-surface-frame-estimation-armless-quadruped.md)「与周报说法的核对」一节。

## 对 wiki 的映射

- [paper-scrub-surface-frame-estimation-armless-quadruped](../../wiki/entities/paper-scrub-surface-frame-estimation-armless-quadruped.md)
