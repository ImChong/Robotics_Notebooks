# AthenaZero: A low-inertia, bimanual robot for dynamic manipulation（Science Robotics 2026）

> 来源归档（ingest）

- **标题：** AthenaZero: A low-inertia, bimanual robot for dynamic manipulation
- **类型：** paper / hardware-platform / dynamic-manipulation / bimanual / quasi-direct-drive
- **期刊：** Science Robotics, Vol. 11, Issue 118（2026-09-16）
- **DOI：** <https://doi.org/10.1126/scirobotics.aee1868>
- **PDF：** <https://www.science.org/doi/pdf/10.1126/scirobotics.aee1868>
- **作者（首作者组）：** Andrew S. Morgan、Gregory Xie、Capprin Bass 等（RAI Institute 大团队；末位含 Alfred A. Rizzi、Annan Mozeika、Nicolas Rojas、Lael Odhner）
- **机构：** 机器人与人工智能研究所（RAI Institute）；现代汽车集团（Hyundai Motor Group）资助
- **项目页 / 博客：** <https://rai-inst.com/resources/blog/bimanual-robot-for-dynamic-manipulation/>
- **代码：** [effective_mass_analysis](../repos/effective_mass_analysis.md) — MIT；有效质量椭圆复现
- **数据：** [Zenodo 21939225](https://doi.org/10.5281/zenodo.21939225) · [Zenodo 22002793](https://doi.org/10.5281/zenodo.22002793) — 冲击测试、刚度评测、Fig.5–6 复现数据
- **入库日期：** 2026-09-17
- **一句话说明：** RAI 首款 **低惯量准直驱双臂** 原型 **AthenaZero**：通过 **传动远端化** 与 **5:1 级低减速比** 把腕部 **有效质量** 压到人臂量级（约 **3.97 kg** vs FR3 **29 kg**），无腕部力矩传感器；以 **棒球式投/接/打** 与人机对传验证 **人类节奏** 动态操作，并开源 **有效质量分析** 工具与评测数据。

## 开源核查（步骤 2.5，截至 2026-09-17）

| 资源 | 状态 |
|------|------|
| **有效质量分析代码** | **已开源** — [rai-opensource/effective_mass_analysis](https://github.com/rai-opensource/effective_mass_analysis)（MIT）；`uv run plot-inertia-ellipse` |
| **Zenodo 数据包** | **已发布** — 冲击/刚度 CSV、`effective_mass_analysis-main.zip`、Fig.5–6 复现 |
| **简化 AthenaZero MJCF** | **随分析仓提供** —  primitive 视觉网格 + 论文报告转子惯量/减速比 |
| **完整 CAD / 真机控制栈 / 棒球任务控制器** | **未列公开 URL** |
| **抛接学习栈** | 见 [robot_juggling arXiv:2608.26800](../papers/robot_juggling_arxiv_2608_26800.md) — **仍确认未开源** |

**结论：** **部分开源** — 可复现 **有效质量对比与 Fig.5–6 数据**；不可复现完整硬件制造与棒球/对传控制 demo。

## 摘要级要点

- **问题：** 商用协作臂减速比常 **>80:1**，反射惯量放大 → 接触时 **有效质量** 高、难柔顺、难匹配 **人类 cadence** 动态操作。
- **设计目标：** 在 **>3 kg** 负载下同时最大化 **控制 authority**（快速加速）并最小化 **有效质量**；使人臂级 **力透明 / 反驱** 与大力输出可切换。
- **构型：** 1-DoF 躯干 + 双 7-DoF 臂 + 双 6-DoF 欠驱动手（**27 关节 / 22 执行器**）；身高约 **1.6 m**，臂展约 **1.8 m**。
- **执行器：** 四套定制 **准直驱** 行星减速（效率 **≥97%**）；多数关节 **5:1**（`<11:1`）；**无** 六维力传感器，**电机电流** 估力矩。
- **传动远端化：** 电机质量收向 **躯干**，降低摆臂时的 **运动质量**；腕部并联机构减速比 **随构型略变**（分析仓给中性 workspace 值）。
- **棒球验证：** 投 **>30 m/s**；短距 **7.3 m** 接 **>14 m/s**、打 **>14 m/s**（82% 接触）；机机/人机对传与 **~3 min**  batting practice。
- **对比：** 腕部中性构型有效质量 **3.97 kg** vs 人臂 **2.76 kg** vs FR3 **29.21 kg** vs UR5e **34.72 kg**；整臂接触热图仍显著低于协作臂。

## 核心摘录（面向 wiki 编译）

### 1) 有效质量与 human cadence

- **摘录要点：** 优化指标是 **接触点有效质量**；高惯量迫使控制器 **降速** 才能柔顺；低减速比 + 近端质量布局 → Fluid 加速/减速。
- **对 wiki 的映射：**
  - [paper-athenazero](../../wiki/entities/paper-athenazero.md) — 设计哲学
  - [Contact-Rich Manipulation](../../wiki/concepts/contact-rich-manipulation.md) — 利用接触而非回避

### 2) 准直驱与 sensing

- **摘录要点：** 反射惯量 ∝ 减速比²；97% 行星齿轮 + 电流力矩传感 → 去掉 FT 传感器漂移/脆弱点；**反驱** 允许外力回传（wind-up 投球链）。
- **对 wiki 的映射：**
  - [paper-athenazero](../../wiki/entities/paper-athenazero.md) — 工程实践表

### 3) 棒球三项 + 人机闭环

- **摘录要点：** 动态轨迹优化投掷；阻抗匹配接球；实时估计球路 batting；机机 **8** 次对传、人机 **12** 次；非最小 jerk 即可解释人类式 motion（kinetic chain / task readiness impedance）。
- **对 wiki 的映射：**
  - [paper-athenazero](../../wiki/entities/paper-athenazero.md) — 评测节
  - [paper-robot-juggling-athenazero](../../wiki/entities/paper-robot-juggling-athenazero.md) — 同平台软件线

### 4) 局限

- **摘录要点：** 不适合 **长时间静态持重**（散热）与 **高 endpoint 刚度** 轨迹（焊接）；适合 **力ful 装配** 靠柔顺容错。
- **对 wiki 的映射：**
  - [paper-athenazero](../../wiki/entities/paper-athenazero.md) — 局限与风险

## 对 wiki 的映射

- [paper-athenazero](../../wiki/entities/paper-athenazero.md)（Science Robotics 硬件论文实体 + 有效质量复现时序图）
- 交叉：[Robot Juggling / AthenaZero 学习](../../wiki/entities/paper-robot-juggling-athenazero.md)、[ZEST](../../wiki/entities/paper-zest.md)、[Sumo](../../wiki/methods/sumo.md)、[Manipulation](../../wiki/tasks/manipulation.md)

## 参考来源

- [Science Robotics DOI](https://doi.org/10.1126/scirobotics.aee1868)
- [RAI 博客](https://rai-inst.com/resources/blog/bimanual-robot-for-dynamic-manipulation/)
