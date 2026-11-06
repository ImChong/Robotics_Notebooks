# Sentis & Khatib ICRA 2006 全身控制框架通读（WBC 概念视角）

> 来源归档（原文通读 · 支撑 `wiki/concepts/whole-body-control.md`）

- **标题：** A Whole-Body Control Framework for Humanoids Operating in Human Environments
- **作者：** Luis Sentis, Oussama Khatib（Stanford AI Lab）
- **类型：** paper / conference（ICRA 2006, pp. 2641–2648）
- **原文：** <https://khatib.stanford.edu/publications/pdfs/Sentis_2006_ICRA.pdf>（2026-11 获取 8 页全文并以 pdftotext 通读）
- **一句话说明：** 把任务优先级（零空间投影）、算子空间力控、浮基动力学与支撑接触整合成"约束 > 操作任务 > 姿态"的三层 WBC 框架。
- **核对范围：** 通读全文 §I–§V 与参考文献表（含式 1–25、图 1–10）。论文自述浮基 + 支撑接触部分"仅简述"，姿态任务的动力学与真机结果留待后续期刊版，文中明确"尚无实现结果"；这些未在本文出现的内容未核对。未阅读其前置论文 [13][14]。
- **通读：** 已完成

## 核心摘录（均出自原文）

- **三类控制原语（§II-A，图 1）：** 约束（接触、关节限位、避碰、平衡）、操作任务（手、头/视线、脚、质心、髋）、姿态（髋高、上身朝向、对称、力矩最小）；行为 = 多个原语的聚合，各原语是独立的"控制实体"（图 2）。
- **浮基运动学（§II-B，式 1）：** 以 6 个虚拟无驱动自由度描述基座，任务运动学相对浮基原点定义，使全身关节资源可被协调，而非逐肢体局部雅可比。
- **优先级层级（§II-C，式 2）：** Γ = Γ_constraints + N_c^T(Γ_tasks + N_t^T Γ_postures)；约束作为首要任务（"never violated"），操作任务投影到约束零空间 J_t|c = J_task N_c（式 4），姿态再投影到任务零空间 J_p|t|c（式 5）；对比此前把约束当次级目标的方法（Liegeois 梯度投影等）。
- **算子空间力控（式 3, 7–10）：** Γ = J^T F；带约束任务惯量 Λ_t|c = (J_t|c A^-1 J_t|c^T)^-1；力矩 Γ_task = J_t|c^T(Λ̂ F* + μ̂ + p̂) 可使任务空间解耦为 ẍ = F*。
- **阻抗与混合控制（式 11–13）：** F* 取阻抗律（M_des, D_des, K_des）或位置/力选择矩阵混合控制。
- **可行性监测（§II-D）：** 任务在约束下不可行时 J_t|c 奇异，因而可运行时检测；可只控可行方向（舍弃奇异方向）。
- **非接触约束（§III，式 14–19）：** 关节限位、近身障碍、自碰撞用势场（反应式）约束控制；多约束雅可比竖直拼接；约束之间是否再分优先级"尚不清楚"。
- **平衡（式 20）：** 平衡应归约束还是任务作者仍在研究，折中方案是在约束与任务之间新增平衡优先级层。
- **浮基 + 支撑接触（§IV，式 21–25）：** 假设支撑点加速度为零（地面刚硬、摩擦足够）得 A·[ẍ_base; q̈] + b + g = N_support^T S_act^T Γ_act；任务雅可比 J_t|c = J_task N_c S̄_act S_act N_support；仿真"越墙伸手"（图 10）。
- **局限（§V）：** 不确定性（接触与近身障碍难定位）是主要担忧，需全层级阻抗控制；论文称"尚无实现结果"。

## 对 wiki 的映射

- [`wiki/concepts/whole-body-control.md`](../../wiki/concepts/whole-body-control.md)
- 既有摘要归档：[sentis_khatib_icra_2006_whole_body_control_framework.md](./sentis_khatib_icra_2006_whole_body_control_framework.md)
