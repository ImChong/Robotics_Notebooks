# Flying Knots — Learning Deformable Object Manipulation Using Task-Level Iterative Learning Control

> 来源归档（ingest）

- **标题：** Learning Deformable Object Manipulation Using Task-Level Iterative Learning Control（仓库 README 亦写作 *Learning Dynamic Rope Manipulation Using Task-Level Iterative Learning Control*）
- **类型：** paper
- **来源：** arXiv preprint
- **原始链接：**
  - <https://arxiv.org/abs/2602.21302>
  - PDF：<https://flying-knots.github.io/static/flying_knot_paper.pdf>
  - 项目页：<https://flying-knots.github.io/>
  - 代码：<https://github.com/krish-suresh/flying_knots_public>
- **机构：** Carnegie Mellon University（卡内基梅隆大学）
- **作者：** Krishna Suresh、Chris Atkeson
- **入库日期：** 2026-07-02
- **一句话说明：** 提出 **Task-Level Iterative Learning Control（任务级迭代学习控制）**：用 **单次人类示教 + 简化绳模型**，在 **xArm7 真机** 上通过 **critical-point 任务误差 + 逆模型 QP** 迭代修正 Bézier 命令，在 **7 种绳索**（链、乳胶管、编织/绞合绳，直径 7–25 mm、线密度 0.013–0.5 kg/m）上 **≤10 次试验 100% 成功**，且多数绳型间 **2–5 次试验可迁移**。

## 核心论文摘录（MVP）

### 1) 问题：可变形体动态操作的高维欠驱动

- **链接：** <https://arxiv.org/abs/2602.21302>
- **摘录要点：** 绳索等可变形体 **自由度近乎无限**、动力学 **欠驱动**；人类与机器人都难以稳定完成 **非平面（non-planar）动态打结** 类任务。主流路线依赖 **大规模示教** 或 **海量仿真**，样本与 sim2real 成本高。
- **对 wiki 的映射：**
  - [Flying Knots](../../wiki/entities/paper-flying-knots.md) — 问题定义与「单示教 + 真机 ILC」定位

### 2) 任务：Flying Knot 与 critical point

- **链接：** <https://flying-knots.github.io/>
- **摘录要点：**
  - **Flying knot**：双手甩动绳索使其中段 **自碰撞成结** 的动态操作 benchmark。
  - **Critical point**：绳段 **首次自碰撞** 的时刻/状态；ILC 以该点的 **任务空间误差** 为主目标，辅以 follow-through 跟踪。
  - 示教经 Vicon 采集 → 清洗 rope-marker 链 → 标注 critical point → 轨迹优化 IK 得初始机器人 Bézier 命令。
- **对 wiki 的映射：**
  - [Flying Knots](../../wiki/entities/paper-flying-knots.md) — 任务语义、示教管线与 critical-point 目标

### 3) Task-Level ILC：逆模型 QP 更新命令

- **链接：** <https://arxiv.org/abs/2602.21302> §IV；<https://github.com/krish-suresh/flying_knots_public/blob/main/docs/architecture.md>
- **摘录要点：**
  - 每轮迭代：执行当前命令 **u** → 测量/仿真绳状态 **x** → 在 critical point 计算任务误差 → 线性化机器人+绳模型得 **Δx = M Δu** → 解 **Eq. 3 逆模型 QP** 得 **Δu** → **u**<sub>k+1</sub> ← **u**<sub>k</sub> − **Δu**。
  - 绳动力学主用 **粒子链模型**（刚度/阻尼沿链；可选圆柱碰撞力）；亦提供 Drake 刚体链与 PyElastica Cosserat 杆备选。
  - 命令参数化为 **Bézier 曲线**；初始猜测由 **Eq. 4 轨迹优化 IK**（Drake SNOPT）从人手轨迹生成。
- **对 wiki 的映射：**
  - [Flying Knots](../../wiki/entities/paper-flying-knots.md) — Mermaid 四阶段管线与 ILC 机制归纳

### 4) 硬件实验、绳型泛化与开源实现

- **链接：** <https://arxiv.org/abs/2602.21302> §V；<https://github.com/krish-suresh/flying_knots_public>
- **摘录要点：**
  - 平台：**xArm7** + **Vicon** mocap；`uv sync` 管理依赖；数据根目录 `$FLYING_KNOT_DATA`。
  - **7 种绳索** 全部在 **≤10 trials** 达 **100% 成功率**；绳型间迁移多数 **2–5 trials**。
  - 开源四阶段脚本：`human_capture` → `clean_demo` → `compute_ik` → `learning`；逆模型在 `simulation/inverse_model.py`（Clarabel QP）。
- **对 wiki 的映射：**
  - [Flying Knots](../../wiki/entities/paper-flying-knots.md) — 实验、局限与代码映射
  - [flying_knots_public 仓库](../../sources/repos/flying_knots_public.md)

## 引用（arXiv BibTeX）

```bibtex
@misc{suresh2026learningdynamicropemanipulation,
      title={Learning Dynamic Rope Manipulation Using Task-Level Iterative Learning Control},
      author={Krishna Suresh and Chris Atkeson},
      year={2026},
      eprint={2602.21302},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2602.21302},
}
```
