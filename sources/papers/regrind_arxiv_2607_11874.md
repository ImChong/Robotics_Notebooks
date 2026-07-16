# regrind_arxiv_2607_11874

> 来源归档（ingest）

- **标题：** REGRIND: A Minimalist Retargeting-Guided Reinforcement Learning Recipe for Dexterous Manipulation
- **类型：** paper
- **来源：** arXiv abs / arXiv HTML / 项目页 / GitHub
- **原始链接：**
  - <https://arxiv.org/abs/2607.11874>
  - <https://arxiv.org/html/2607.11874>
  - <https://www.yunhaifeng.com/REGRIND/>
  - <https://github.com/yunhaif/regrind>
- **作者：** Yunhai Feng, Natalie Leung, Jiaxuan Wang（Cornell University）；Lujie Yang, Haozhi Qi（Amazon FAR）；Preston Culbertson（Cornell University）
- **入库日期：** 2026-07-16
- **一句话说明：** 从**单次光学动捕人手–物体演示**出发，用 **interaction mesh 交互保留重定向**（OmniRetarget 同族 Laplacian 形变能 + Drake/MOSEK SQP）生成机器人参考轨迹，再以 **残差 RL + 物体关键点跟踪奖励 + RSI 参考状态初始化 + 训练时 SE(3) 增广** 在仿真中闭环学习，经系统辨识后 **零样本** 部署到 **LEAP / WUJI** 灵巧手，完成剪刀、螺丝刀等 **contact-rich 工具操作**；系统实验对比 SPIDER、DexMachina、Mink IK+RL，并总结灵巧操作 sim2real 比 loco 更敏感。

## 核心摘录

### 1) REGRIND（Feng 等，arXiv:2607.11874，2026）
- **链接：** <https://arxiv.org/abs/2607.11874>；HTML：<https://arxiv.org/html/2607.11874>；项目页：<https://www.yunhaifeng.com/REGRIND/>；代码：<https://github.com/yunhaif/regrind>
- **问题：** 人形 WBT 已验证「重定向参考 + RL 跟踪」简洁配方，但灵巧 **contact-rich manipulation** 涉及接触模式/力精细调节，纯运动学重定向常产生 **穿透、接触结构丢失**，下游 RL 难学或 sim2real 失败。
- **核心管线（real-to-sim-to-real）：**
  1. **3D 人类演示：** MANO 手部关键点 + 物体 6D 位姿（铰接物体含关节角）；剪刀来自 ARCTIC，螺丝刀为自采光学 mocap。
  2. **交互保留重定向：** 物体 + 人手语义关键点 Delaunay 四面体建 **interaction mesh**；最小化源/机器人 mesh 的 **Laplacian 坐标差** + 时序平滑；逐帧 SQP（Drake + MOSEK）；保留 hand–object 空间与接触关系（ formulation 继承 [OmniRetarget](https://arxiv.org/abs/2606.16272)）。
  3. **参考引导 RL：** **残差动作** 叠在名义参考关节上 → PD；**物体-centric 关键点距离** 指数跟踪奖励（无需显式接触先验）；**RSI** 从重定向轨迹采样重启 + 小扰动；**训练时增广**：初始物体位姿 ±5 cm / ±30°，对整条参考做时间插值 SE(3) 变换（不重跑重定向优化）。
  4. **Sim2Real：** 域随机化（摩擦、电机增益、时延等）、观测噪声、推力/重力课程；部署时 **MoCap 直接提供物体位姿**（隔离感知误差）；需 **系统辨识**。
- **平台与任务：** UR5e + **16-DoF LEAP** / **20-DoF WUJI**；四组 **剪刀 / 螺丝刀 × LEAP / WUJI**；LEAP 因尺寸使用 3D 打印放大工具。
- **仿真结果（Table 1，1024 rollout 级）：** REGRIND 四任务 SR **98.7–99.8%**，关键点误差 **5.3–6.5 mm**；DexMachina 剪刀任务 SR **0–22%**；Mink IK+RL 多数 **0–3%**；SPIDER 作为开环/MPC 轨迹跟踪演示，四任务 SR **0%**（轨迹偏离演示、不适合 residual RL 初始化）。
- **真机结果（Table 2–3）：** LEAP-Scissors **9/10**、LEAP-Screwdriver **10/10**、WUJI-Screwdriver **9/10**；**WUJI-Scissors 0/10**（非反驱电机 + 剪刀 mesh 误差）；随机初态泛化与演示初态接近（Table 3）。
- **相对基线洞见：** interaction-preserving 重定向显著改善 RL 初始化与正则；DexMachina 无交互语义时易 ** exploit 仿真 artifact**，真机 screwdriver 亦不稳定。
- **局限：** 部署依赖 **MoCap 物体状态**；仍需仔细系统辨识；单演示 + 增广覆盖有限。
- **代码：** MIT；仓库含预计算重定向轨迹，重定向模块依赖 **Drake/MOSEK**（可选安装）。
- **对 wiki 的映射：**
  - 新建 [REGRIND（重定向引导灵巧操作 RL）](../../wiki/methods/regrind-retargeting-guided-rl.md)
  - 新建 [REGRIND 论文实体页](../../wiki/entities/paper-regrind-dexterous-manipulation.md)
  - 在 [TopoRetarget](../../wiki/methods/toporetarget-interaction-preserving-dexterous-retargeting.md)、[SPIDER](../../wiki/methods/spider-physics-informed-dexterous-retargeting.md)、[Motion Retargeting Pipeline](../../wiki/concepts/motion-retargeting-pipeline.md)、[Manipulation](../../wiki/tasks/manipulation.md) 中补充交叉引用

## 当前提炼状态

- [x] 摘要 + 方法主线（重定向 / RL / 增广 / sim2real）已摘录
- [x] 四任务仿真与真机数字已摘录
- [x] wiki 方法页与论文实体页待落盘
