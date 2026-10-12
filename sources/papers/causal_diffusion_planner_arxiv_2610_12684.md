# Causal Diffusion Planner（非抓取多物体放置）论文来源归档

## 元数据

- **题目：** Non-Prehensile Multi-Object Placement with Mobile Robots via Diffusion-Based Trajectory Planning
- **作者：** Lance Pharand, Megnath Ramesh, Mark Chiu, Yash Vardhan Pant, Stephen L. Smith
- **单位：** Electrical and Computer Engineering, University of Waterloo（arXiv HTML 页首）
- **预印本：** https://arxiv.org/abs/2610.12684 （cs.RO，v1，2026-10-08）
- **HTML：** https://arxiv.org/html/2610.12684v1
- **代码：** 论文写 “Robot demo videos and code available on our project website”，指向匿名仓库 https://anonymous.4open.science/r/Causal_Diffusion_Planner-0181 （已核实为真实代码仓库，见 [sources/repos/causal_diffusion_planner.md](../repos/causal_diffusion_planner.md)）
- **入库日期：** 2026-10-12

## 论文要解决的问题

移动机器人（SE(2) 位姿）推多个箱子到小容差目标区时，现有推物研究多针对粗略目标区，精确放置多在机械臂 + Push-T 上研究。移动机器人修正小误差往往需要绕物体做大幅重定位。论文将其形式化为带紧容差的多目标分配问题：每个物体有一个“自身占地 + 余量 δ”的方形目标区（实验 δ∈{4, 2} cm，箱边长 8 cm）；物体放入目标且机器人不再接触才算放置，已放置物体不得被扰动；用 TSP 为箱子分配目标并排序。

## 方法与训练信息

1. **因果分解：** p(τ^R, τ^O | c) = p(τ^R | c) · p(τ^O | τ^R, c)。先去噪机器人轨迹（Robot 阶段），再以干净机器人轨迹的 Tweedie 估计 τ̂^R_0 为条件去噪物体轨迹（Object 阶段）。
2. **网络：** 两阶段共享一个 DiT 风格的 Transformer 解码器，带分离的时空注意力；掩码使干净机器人 token 不会关注带噪物体 token；v-prediction；机器人条件 token 采用信号比例门控 h^R_t = ᾱ_t φ(τ̂^R_0) + (1−ᾱ_t) h_∅。Push-T 模型约 17.8M 参数，移动机器人扩散模型约 75–90M。预测时域 T_p 原文未给出。
3. **联合基线：** 同一网络同时去噪机器人和物体轨迹。
4. **引导（五项代价）：** C_place（沿推方向的非对称终点误差，过冲惩罚为欠冲的 25 倍，加横向平方误差）；C_pred（按末端速度恒定假设预测落点）；物体碰撞与机器人碰撞两项屏障代价（Ψ(x)=e^x−x−1）；C_retreat（需要撤离时惩罚机器人靠近目标物体）。采用 K 样本蒙特卡洛估计与 softmax 权重（移动机器人实验 K=16），逐项梯度归一化并带门控阈值 η。
5. **Best-of-N：** Push-T 选“预测物体轨迹终点最接近目标”的样本，N∈{1, 8, 16, 32} 扫描；移动机器人选引导代价最小的样本，主设置为 best-of-16。
6. **数据：** Push-T 使用 Hugging Face pusht_keypoints；移动机器人使用约 500 条 δ=2 cm 的人类遥操作演示，撞到已放置箱的回合被剔除。

## 论文报告的结果

- **Push-T（每行 75 回合）：** Causal[100 DDIM] 成功率 54.7%（最大覆盖 94.1%，最终覆盖 91.6%），Causal[50 DDIM] 53.3%，Diffusion Forcing 44.0%，Joint 40.0%。接触一致性（CC，越低越好）：Causal 10.8–11.6%，Diffusion Forcing 29.3%，Joint 18.3%。延迟中位数：Causal[50] 2008 ms，基线约 2300 ms，Causal[100] 4257 ms。官方 LeRobot Diffusion Policy 作为参考引用（65% 成功率，0.3B 参数，图像输入），未重跑。
- **三个仓库式多箱环境（U-Shape、Pallet Staging、Cross-Dock Staging）：** Causal Best-of-16 相比不带引导的 Diffusion Policy 基线，箱子放置比例（BPF）平均提升 21.5 个百分点，最高 30.9；平均每次重规划 2812 ms，联合去噪 Best-of-16 为 5088 ms，快 1.8 倍。Causal + MC 引导（DPS）相对 DP 平均 +8.1 点、最高 +18.8 点。Pallet Staging 的 2500 个无引导计划中 3.6% 过冲超 1 cm、0.64% 超 2 cm，引导后降至 2.6% 与 0.20%。箱数原文正文未给。
- **TurtleBot 3 Burger 真机：** Causal + MC 引导不做微调，在部分重建的 Cross-Dock Staging 上放置 4 种未见物体（三棱柱、圆柱、五边形、星形），每方法 10 次试验、3 cm 容差；头顶相机跟踪 AprilTag（20 Hz），底层位姿控制器 30 Hz，仅用四个目标槽。结果：Causal 放置 85.0% 的物体、60% 回合全部完成；Diffusion Policy 37.5%、0% 回合全部完成。

## 复现与解释边界

- 全局位姿依赖头顶相机；机载相机会在推箱时丢失目标标签。
- 二值成功率低（Causal Best-of-16 为 7%–37%）；任何已放置箱被扰动即判失败。Best-of-16 的首要失败是被推箱碰到已放置箱（约 33% 回合），引导类方法与基线的首要失败是机器人碰到已放置箱（约 45%）。
- 真机只有 10 次试验/方法和四种物体，结论为趋势性。
- 仓库为匿名审稿版（README 写 under review for ICRA 2027）；附录未单独核对。

## Robotics_Notebooks 映射

- 详情节点：wiki/entities/paper-causal-diffusion-nonprehensile-multi-object-placement.md
- 汇总页：wiki/overview/embodied-station-34-papers-spatial-grounding-technology-map.md

## 一手来源

- arXiv：<https://arxiv.org/abs/2610.12684>
- HTML：<https://arxiv.org/html/2610.12684v1>
- 匿名代码仓库：<https://anonymous.4open.science/r/Causal_Diffusion_Planner-0181>
