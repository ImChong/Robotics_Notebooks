# SG-ODOM（arXiv:2611.20710）原论文全文核对

- **类型：** 论文 paper
- **英文标题：** SG-ODOM: Event-Driven Proprioceptive State Estimation with Stability-Gated Contact Updates for Legged Robots
- **中文标题：** SG-ODOM：采用稳定性门控接触更新的事件驱动腿式机器人本体里程计
- **作者：** Francesco Iotti, Chiyun Noh, Franco Angelini, Manolo Garabini, Ayoung Kim, Maurice Fallon, Frank Dellaert
- **机构：** 意大利比萨大学（Università di Pisa）；首尔国立大学（Seoul National University）；牛津大学（University of Oxford）；佐治亚理工学院（Georgia Institute of Technology）。按 arXiv HTML v1 作者行及脚注逐人核对；原文不是「机构未说明」。
- **论文：** https://arxiv.org/abs/2611.20710
- **HTML 全文：** https://arxiv.org/html/2611.20710v1
- **PDF：** https://arxiv.org/pdf/2611.20710
- **版本：** arXiv:2611.20710v1，2026-11-16
- **代码：** https://github.com/ori-drs/SG-ODOM
- **入库日期：** 2026-11-19
- **核对范围：** 通读 arXiv HTML v1 摘要、§I 引言、§II 相关工作、§III 状态/接触模型及 InEKF 与 Fixed-Lag 两后端、§IV 事件驱动更新和接触门控公式 (1)–(9)、§V-A–D 超时消融/接触分类消融/三平台实验/代表轨迹、§VI 结论、参考文献与正文 Fig.1–6 图注、Table I–VII 数据；对照官方 GitHub README、ROS2 入口、门控与后端核心代码。阅读 HTML 而非 PDF 逐页图像核查；未运行算法或复现实验。
- **通读：** 已完成

## 核心摘录与证据

1. **问题动机（§I–II）：** 足端触地≠可作为静止里程计参考，冲击/载荷转移会使足端仍在运动；每个关节采样都融合高度相关的接触量测也不一定增益。研究核心是「何时进行运动学更新」，不是更换 IMU 本身。
2. **状态与模型（§III）：** 浮动基状态 `X=(R,p,v)∈SE₂(3)`；足端 FK 量测 `z_i^B=Rᵀ(f_i^N-p_B^N)+n_i`，新触地的支撑足世界系位置由当前姿态及 FK 初始化；使用同一个接触前端比较两个估计后端。
3. **稳定门控（§IV-B，Eq.6–9）：** 将关节力矩残差借足端 Jacobian 估算 GRF（考虑 Pinocchio/RNEA 模型项），只用有符号竖直力；加载判据 `F_i≥F_on` 与短时力变化 `ΔF_i≤ΔF_max` 同时持续满足 `T_on` 才确认「估计可用接触」；释放采用持续性条件，拒绝触地瞬间的过渡段。
4. **事件调度（§IV-A）：** 高采样率持续处理 IMU、接触消息；新确认的 touchdown 触发一次即时接触修正。无新 touchdown 时，距上次修正超过 `Δt_max` 的下一条接触消息触发超时修正；每次修正重置计时。不等同于固定频率的接触降采样。
5. **两后端（§III-B）：** SG-ODOM-InEKF 扩展状态含每足 foothold slot，以 IMU 传播及接触 EKF 更新；SG-ODOM-FL 以事件状态、IMU 预积分、每个接触 episode 的足端 landmark、IMU bias random walk 在固定滞后窗口中优化并边缘化旧状态。
6. **更新频率消融（Table I–II）：** `Δt_max` 试验 0.01–0.30 s；0.05 与 0.10 s 附近表现较好，后续统一 0.10 s。ANYmal Fixed-Lag 的平移 RPE：0.01 s 为 0.406 m，0.10 s 为 0.157 m，0.30 s 为 0.418 m；频率越高不必然越准。
7. **接触策略消融（Table III–IV）：** 稳定门控下 InEKF 的 ANYmal/Spot/G1 平移 RPE 分别 0.252/0.210/0.115 m，FL 为 0.157/0.185/0.109 m；三平台均优于同后端的简单力阈值条件。具体工程实现的 G1 门控简化需同时查看官方代码。
8. **ANYmal D（Table V）：** GrandTour 七条任务、1 m 路径段 `RPE_t` 平均：SG-ODOM-FL 0.157 m，ANYbotics 原厂估计器 0.208 m（约降低 24.5%）；SG-ODOM-InEKF 0.252 m。七条中 FL 赢五条，HEAP-1 原厂更好。参考为 Leica MS60 的位置轨迹，**未报告 RPE_r**。
9. **Spot（Table VI）：** GaRLILEO 四条，SG-ODOM-FL 平均 `RPE_t=0.185 m`，Pronto 0.226 m；`RPE_r` FL 0.821°、InEKF 0.814°。轨迹参考混有地图辅助 LiDAR 和 Vicon 的不同来源。
10. **Unitree G1（Table VII）：** 自采 QUAD 112.5 m、SLOPE 64.4 m（4°），FL 的 `RPE_t=0.109 m`、`RPE_r=3.859°`；Unitree 原厂 0.248 m、5.940°。**参考是 Surfel-LIO 输出而非独立绝对真值**，只能称相对一致性。
11. **局限（§V、VI）：** FL 各平台 IMU bias random-walk 参数经经验调整而非统一 Allan 标定；G1 仅慢速铺装地面，未验证碎石/冰雪高动态人形；需要未来自适应稳定条件与视觉/激光互补融合。本文是里程估计指标，非全身控制策略成功率或部署时延证明。

## 对 wiki 的映射

- [独立 SG-ODOM 论文详情节点](../../wiki/entities/paper-sg-odom-2611-20710.md) — 稳定接触判据、双后端、三平台 1 m RPE、工程边界及 Mermaid
- [State Estimation](../../wiki/concepts/state-estimation.md) — 本体预测/接触更新何时修正的工程案例
- [传统模型控制纵深 Stage 4](../../roadmap/depth-classical-control.md) — 真机 WBC/TSID 的状态估计入口
- [源码与运行入口核对](../repos/sg-odom.md) — ROS2 bag replay、分类器差异、真实调用时序
