# Crocoddyl 论文与官方仓库通读（ICRA 2020 + loco-3d/crocoddyl README）

> 来源归档（原文通读 · 支撑 `wiki/entities/crocoddyl.md`）

- **标题：** Crocoddyl: An Efficient and Versatile Framework for Multi-Contact Optimal Control（Mastalli 等，ICRA 2020）及官方仓库文档
- **类型：** paper + repo
- **原文：**
  - 论文 <https://arxiv.org/abs/1909.04947>（v2，2020-03-11，7 页，PDF 全文 pdftotext 提取）
  - 官方仓库 README（2026-11-07 取自 <https://raw.githubusercontent.com/loco-3d/crocoddyl/devel/README.md>，devel 分支）、`PUBLICATIONS.md`、`notebooks/01_introduction_to_crocoddyl.ipynb`
- **许可：** BSD-3-Clause（README 徽章与许可说明）
- **入库日期：** 2026-11-07
- **一句话说明：** 在给定接触序列下做多接触最优控制的 DDP 系库：解析稀疏导数（基于 Pinocchio）+ 李群状态 + 新的 FDDP 求解器（接受不可行初值、保留 gap）。
- **核对范围：** 论文正文全部章节（摘要、§II 问题与接触/冲击模型、§III FDDP 前向 / 后向 / 步长接受、§IV 步态与高动态动作结果、§V 结论）通读；参考文献仅浏览。README 通读（特性列表、安装方式、文档与 notebook 清单、引用与论文清单）；`PUBLICATIONS.md` 与 notebook 01 只读了开头介绍部分，**未阅读**源码（C++ 求解器实现）、其余 notebook 与 Doxygen 文档，也未运行任何示例或复现论文数值。
- **通读：** 已完成

## 核心摘录（均出自原文）

- **问题（论文式 (1)）：** 状态 x=(q,v) 位于流形上，控制 u=(τ,λ) 含关节力矩与接触力；接触被建模为完整约束 φ(q)=0，用 KKT 线性系统同时解加速度与接触力（式 (2)），并借 Baumgarte 型 PD 稳定项处理漂移（式 (3)）。该动力学忽略摩擦锥与关节限位，因此可以用无约束 DDP 求解；不等式约束可用惩罚、有效集或增广拉格朗日加入（论文称为后续工作）。冲击用 e∈[0,1] 的恢复系数写成脉冲动力学（式 (6)），完全非弹性取 e=0。
- **效率来源：** 稀疏解析导数（RNEA 导数与运动学导数，经 LDU / Cholesky 分解，式 (4)–(5)）、问题结构利用与数据共享；论文称多线程只并行导数计算，4–8 线程约把单次迭代耗时减半，超线程反而变慢。
- **FDDP 要点：** 经典 DDP 的前向 rollout 总是闭合动力学 gap，初值需要先给控制序列、全球化策略差；FDDP 让后向传播接受不可行轨迹（Value 梯度按 gap 修正 V_x⁺ = V_x + V_xx f̄），前向 rollout 保持 gap 随 (1−α) 收缩，步长 α=1 时退化为经典 DDP；步长用 Goldstein 条件（b1=0.1，b2=2），因为不可行迭代里 ΔJ 可能是上升方向。作者强调这等价于对只含等式约束的多重打靶 KKT 系统做牛顿法，且不增加额外决策变量。
- **结果：** 步态（walk / trot / pace / bound）与跳跃、前空翻均由预定义接触序列和步时直接产出；约 10–34 次迭代收敛，总计算 < 0.5 s；积分步长通常 1e-2 s（双足行走 3e-2 s），60–115 个节点，地平线 0.6–3 s；单次迭代频率最高 859.6 Hz（jump-4f，i9-9900K，60 节点），接触数目不影响耗时、随节点数线性增长。高动态动作中 FDDP 前几次迭代保持 gap 开启，且闭合后常见超线性收敛。
- **论文自述的局限 / 取舍：** 接触序列（含步时）需预先给定；评测刻意忽略摩擦锥与力矩限制；初值用姿态线性插值 + 准静态力矩；结果均为离线计算耗时（并非闭环部署）。
- **仓库 README：** 自述为"robot control under contact sequence 的最优控制库"，求解器含 FDDP、OdynSQP、Intro、Ipopt 等；Pinocchio 提供解析稀疏导数与微分几何；支持数值微分与 CppAD 自动微分、OpenMP 多线程、Boost Python 绑定、CppADCodeGen 代码生成；依赖 Pinocchio、Eigen、eigenpy、Boost；可通过 pip / conda / apt(robotpkg) / Nix / 源码安装；推荐 6 个 notebook（入门、cartpole 摆起、unicycle、acrobot、codegen、规模化到机器人）；引用清单列出 FDDP / Intro、逆动力学 TO（Inverse-Dynamics MPC via Nullspace Resolution）、Endpoint-Explicit DDP 等后续工作。
- **notebook 01 开头：** Crocoddyl 求解"多接触最优控制问题（MCOP）"：目标为积分代价，约束含动力学 ẋ=f(x,u)、等式 h=0、不等式 g 界与 x∈X、u∈U。

## 对 wiki 的映射

- [`wiki/entities/crocoddyl.md`](../../wiki/entities/crocoddyl.md)

## 参考来源

- Mastalli et al., ICRA 2020, arXiv:1909.04947；仓库 <https://github.com/loco-3d/crocoddyl>
