# Swarm of Micro Flying Robots in the Wild (Science Robotics 2022)

- **作者：** Xin Zhou, Xiangyong Wen, Zhepei Wang, Yuman Gao, Haojia Li, Qianhao Wang, Tiankai Yang, Haojian Lu, Yanjun Cao, Chao Xu, Fei Gao（浙江大学工业控制技术国家重点实验室 / 浙江大学湖州研究院 / HKUST）
- **链接：** https://www.science.org/doi/10.1126/scirobotics.abm5954 ；补充开发文档：`ego-planner-v2` `swarm-playground` 文档
- **状态：** primary_full_text_checked
- **核对范围：** 已通读 *Science Robotics* 2022 论文全文（1–1373 行：TEEM 设计准绳、MINCO 时空联合轨迹参数化与约束转录、通用惩罚项 $J_s/J_t/J_d/J_o/J_w/J_f/J_v$、基于 UWB 的去中心化 VIO 漂移矫正与分层安全兜底、114 mm 掌上无人机硬件构型、野生竹林 10 机穿飞/编队/遮挡目标跟踪/密集互避实飞与 MADER/EGO-Swarm 基准对照）及 `ego-planner-v2` 官方 `swarm-playground` 文档
- **通读：** 已完成

## 核心要点提取

1. **掌上全自主微小型集群硬件平台（< 300 g，114 mm 轴距）**
   - 轴距仅 **114 mm**，含 100 g 电池整备重量 **< 300 g**（两块 3000 mAh 7.4 V LiPo 串联，4 颗 1404 6000 kV 电机配 3 寸三叶桨，推重比 **2.4**，倒置下吹布置避免下洗流吹机身，悬停功耗 ~120 W，续航 **11 min**）。
   - **机载传感与算力**：仅配 Intel RealSense **D430**（深度+双目灰度）、PX4 纳米级飞控（STM32 H7 + BMI088 IMU，省去气压计/磁力计/GPS）、**NVIDIA Xavier NX**（6 核 CPU / 384 核 GPU / 8 GB RAM，非多线程视频追踪场景 CPU/GPU 占用率均 **< 40%**）以及 Nooploop LTPS **UWB**（DW1000 芯片），机间通过单 AP WiFi 或对等 Ad-hoc 网络广播轨迹参数。

2. **MINCO 时空联合轨迹优化与约束转录（Eq. 1–24）**
   - 基于旋翼微分平坦性（$s=3$ 加加速度链），用 **MINCO** 分段五次多项式表达平坦输出轨迹 $p(t) = \mathcal{M}_{q, T}(t)$：将传统密集多项式耦合优化压缩为相邻路点 $q \in \mathbb{R}^{3 \times (M-1)}$ 与各段时长 $T \in \mathbb{R}^M$，经带状矩阵方程 $M(T)c = b(q)$ 的带状 PLU 分解实现 **$O(M)$ 线性复杂度**前向求值与梯度回传（桌面端每段仅约 **1 μs**）。
   - 采用二步约束转录（惩罚函数积分 + 有限等距采样求和），将总时间 $J_t$、平滑能量 $J_s = \int \|p^{(3)}(t)\|^2 dt$、动力学上限 $J_d$（$v_m, a_m, j_m$）、单平面无 ESDF 障碍距离 $J_o$（$d_o = (p-s)^T v$，平面生成仅约 **0.1 ms**）、椭圆加权机间防撞 $J_w$（$z$ 轴降权以减缓下洗流干扰）以及任务项（编队 $J_f$ / 视角对齐追踪 $J_v$）统一化为无约束目标，由 **L-BFGS** 实时求解，并结合 Ceres 优化 UWB 相对测距与 VIO 漂移修正（Eq. 24）。

3. **TEEM 与野外/基准实验验证**
   - 提出 **TEEM** 四维设计准则（Trajectory optimality, Extensibility, Economical computing, Miniature size），通过纯机载感知与弱耦合轨迹广播（配合深度图移除已知他机点云干扰 + 碰撞后台检查 + 紧急制动兜底）在无 GPS 与动捕环境下完成四大野外测试：
     - **密集竹林穿飞（10 机）**：穿行窄至 **< 30 cm** 的野生竹间隙（目标距起点 65 m），借助时间维度协同自动排队避让而非机械改高度浪费能量。
     - **森林编队、4 机抗遮挡人机跟踪（YOLOv5 TensorRT + 卡尔曼滤波）与 3 m 半径随机换位密集互避（10 机，7 cm 安全球模型，中途人工加障与关停地面定位锚点零碰撞）**。
   - 仿真对比 **MADER** 与前代 **EGO-Swarm**（1–3 m/s 期望速度，1–2 m 障碍间距）：两基线因缺乏/受限实时时间剖面联合优化常绕大弯让行，本方案显著降低控制努力（jerk 积分 $S_m$）并缩短计算耗时；`Swarm Playground` 支持 40 机交互式仿真与手柄动态避障测试。

## 局限性与边界条件

- 机间协同依赖共享无线链路广播各自轨迹系数与时间戳；若无线通信完全中断，退化为依赖各自局部深度图的独立避障，高密互避协同能力受限。
- 约束转录后的 L-BFGS 属于惩罚法软约束，必须依赖求解后二次可行性校验（post-check）及失败降级紧急制动（10 ms 重试），无法在数学上单次保证绝对零违约。
