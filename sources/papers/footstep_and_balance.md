# footstep_and_balance

> 来源归档（ingest）

- **标题：** 步态规划与平衡恢复核心论文（ZMP / Capture Point / DCM）
- **类型：** paper
- **来源：** ICRA / IROS / IJRR / IEEE TRO
- **入库日期：** 2026-04-14
- **最后更新：** 2026-04-14
- **一句话说明：** 覆盖 ZMP 预览控制、Capture Point、DCM 步位规划等双足平衡控制核心方法

## 核心论文摘录

### 1) Biped Walking Pattern Generation by Using Preview Control of Zero-Moment Point（Kajita et al., ICRA 2003）
- **链接：** <https://ieeexplore.ieee.org/document/1241826>
- **核心贡献：** 奠定 ZMP 预览控制范式：用线性倒立摆（LIP）模型建模质心动力学；预览未来步骤的 ZMP 参考轨迹；实现 HRP 系列机器人稳定行走；此后成为 humanoid 行走控制 20 年的工程基础
- **关键公式：** LIP 质心方程 $\ddot{x} - \omega^2(x - p) = 0$，$\omega = \sqrt{g/z_c}$
- **对 wiki 的映射：**
  - [lip-zmp](../../wiki/concepts/lip-zmp.md)
  - [footstep-planning](../../wiki/concepts/footstep-planning.md)
  - [capture-point-dcm](../../wiki/concepts/capture-point-dcm.md)

### 2) Capture Point: A Step toward Humanoid Push Recovery（Pratt et al., Humanoids 2006）
- **链接：** <https://ieeexplore.ieee.org/document/4115602>
- **核心贡献：** 提出 Capture Point（CP）概念：机器人一步内可恢复平衡的落脚点；基于 LIP 动力学推导 CP 解析解；为双足扰动恢复提供几何直觉和可行性判据
- **关键公式：** $\xi = x + \dot{x}/\omega$（Capture Point = 位置 + 速度缩放）
- **对 wiki 的映射：**
  - [capture-point-dcm](../../wiki/concepts/capture-point-dcm.md)
  - [balance-recovery](../../wiki/tasks/balance-recovery.md)
  - [footstep-planning](../../wiki/concepts/footstep-planning.md)

### 3) Capturability-based Analysis and Control of Legged Locomotion（Koolen et al., IJRR 2012）
- **链接：** <https://journals.sagepub.com/doi/10.1177/0278364912452673>
- **核心贡献：** 将 Capture Point 推广为 N-step Capturable：定义可用 N 步恢复平衡的状态空间；引入 DCM（Divergent Component of Motion）等价于 1-step CP；为实时步位规划提供理论框架
- **关键洞见：** DCM = 系统发散分量，控制 DCM = 控制平衡；N-step capturability 递归定义
- **对 wiki 的映射：**
  - [capture-point-dcm](../../wiki/concepts/capture-point-dcm.md)
  - [balance-recovery](../../wiki/tasks/balance-recovery.md)
  - [lip-zmp](../../wiki/concepts/lip-zmp.md)
  - [footstep-planning](../../wiki/concepts/footstep-planning.md)

### 4) Online Walking Motion Generation with Automatic Footstep Placement（Herdt et al., 2010）
- **链接：** <https://hal.science/hal-00718198>
- **核心贡献：** 基于 LIP + MPC 的在线步位自动生成：无需预设步位，MPC 自动决定落脚点；支持实时速度命令输入；与 ZMP 约束统一在同一 QP 框架内求解
- **关键洞见：** 步位也是优化变量 → 步态更自然、更具鲁棒性
- **对 wiki 的映射：**
  - [footstep-planning](../../wiki/concepts/footstep-planning.md)
  - [lip-zmp](../../wiki/concepts/lip-zmp.md)
  - [capture-point-dcm](../../wiki/concepts/capture-point-dcm.md)

### 5) Footstep Planning on Uneven Terrain with Mixed-Integer Convex Optimization（Deits & Tedrake, 2014）
- **链接：** <https://arxiv.org/abs/1403.7171>
- **核心贡献：** 将不平整地形步位规划表述为混合整数凸优化（MICP）；用 IRIS 算法提取安全多面体区域；在 Atlas 上验证多步规划；展示凸优化在接触规划中的工程可行性
- **对 wiki 的映射：**
  - [footstep-planning](../../wiki/concepts/footstep-planning.md)
  - [balance-recovery](../../wiki/tasks/balance-recovery.md)

## 当前提炼状态

- [x] 论文摘要填写
- [x] wiki 页面映射确认
- [ ] 关联 wiki 页面的参考来源段落已添加 ingest 链接
