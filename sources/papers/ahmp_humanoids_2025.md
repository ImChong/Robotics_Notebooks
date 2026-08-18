# AHMP: Agile Humanoid Motion Planning With Contact Sequence Discovery（Humanoids 2025）

> 来源归档（ingest）

- **标题：** AHMP: Agile Humanoid Motion Planning With Contact Sequence Discovery
- **缩写 / 框架：** **AHMP**（Agile Humanoid Motion Planning）
- **类型：** paper / humanoid / trajectory-optimization / contact-planning / cem
- **会议：** 2025 IEEE-RAS 24th International Conference on Humanoid Robots（Seoul），pp. 33–40
- **DOI：** <https://doi.org/10.1109/Humanoids65713.2025.11203211>
- **HAL：** <https://hal.science/hal-05072261>（v3，2025-11-07）
- **代码：** <https://github.com/hucebot/ahmp>（BSD-2-Clause）— 归档见 [`sources/repos/ahmp.md`](../repos/ahmp.md)
- **内层 TO 最新实现：** <https://github.com/upatras-lar/se3_trajopt>
- **项目页：** <https://lar.upatras.gr/projects/ibrics.html>
- **视频：** <https://www.youtube.com/watch?v=yIyk8GPU9YE>
- **作者：** Ioannis Tsikelis、Evangelos Tsiatsianas、Chairi Kiourt、Serena Ivaldi、Konstantinos Chatzilygeroudis、Enrico Mingo Hoffman
- **机构：** Inria / Université de Lorraine / CNRS（LORIA，Nancy）；帕特雷大学 LAR；Archimedes / Athena RC
- **入库日期：** 2026-08-18
- **一句话说明：** 外层 CEM-MD 并行采样接触构型（末端二进制编码 + 对数时长），内层用 SE(3) 切空间全身 TO 评可行性；Talos 在扶手走廊与烟囱场景上于数分钟内给出动态多接触计划。

## 开源状态（步骤 2.5）

- **项目页核查（2026-08-18）：** IBRICS 页列出 AHMP 摘要与实验，**无直接 GitHub 按钮**。论文与 README 声明代码为 [hucebot/ahmp](https://github.com/hucebot/ahmp)。
- **仓库核查：** 含 `src/cem/`、`src/nltrajopt/`、`src/examples/cem_exps/trajopt_parallel.py`、`run_docker.sh`、`ci/` Docker。入口：`python trajopt_parallel.py --exp handrails|chimney --robot talos`（可选 `--dz`、`--abl`）。
- **依赖边界：** README 写 IPOPT 内用 **HSL MA97**（学术许可）；可改线性求解器。内层 TO 最新代码指向 `upatras-lar/se3_trajopt`。
- **结论：** **已开源**（规划 / 并行评测入口齐全）。**无真机部署脚本**。

## 摘录 1：问题与主张（§I–§II）

- **痛点：** 多接触规划要么先定 stance 再规划、要么把接触塞进单一 NLP、要么 MIP/手工步态；人形敏捷动作还要全身动力学，长时域很难。
- **主张：** 双层优化。外层黑盒 CEM-MD 发现接触序列；内层全身动力学 TO 用 SE(3) 切空间线性化，从而能用现成欧式 NLP（Ipopt）而不是流形专用求解器。
- **相对先前 CEM-MD 步态发现：** 每个离散–连续对编码一次接触构型 \(C=(eec,d)\)（\(K\) 个末端 → \(eec\in[0,2^K-1]\)），时长在对数空间采样，避免负时长与「未使用相位」。

**对 wiki 的映射：** 升格 [`wiki/entities/paper-ahmp.md`](../../wiki/entities/paper-ahmp.md)；与 [SE(3) 切空间 TO](../../wiki/entities/paper-se3-tangent-to.md)、[FARO](../../wiki/entities/paper-faro-feasibility-aware-robot-motion-optimization.md)、[DSMS](../../wiki/methods/dsms-contact-implicit-multiple-shooting.md) 对照「接触谁来定」。

## 摘录 2：内层 TO 与切空间积分（§III–§IV）

- 状态 \(x=[q^\top,v^\top]^\top\)，控制含加速度与接触力；逆动力学映射到关节力矩，浮动基 6 维力矩约束为 0。
- 接触相位内：摩擦锥、法向力 >0、足端贴地且 \(\dot p_i=0\)；非接触：\(\lambda_i=0\)、离地。
- 浮动基决策变量取 \(\xi_k\in\mathfrak{se}(3)\)；差分为 \(\mathrm{Exp}(\xi_2)\ominus\mathrm{Exp}(\xi_1)\)；积分为 \(\mathrm{Log}(\mathrm{Exp}(\xi)\oplus\mathcal{V}_b h)\)——在离散步长下是精确 retraction，避免四元数欧拉积分 + 归一化引入的非线性。
- 适应度：给定 Ipopt 迭代上限后的**约束违反量**（目标是尽快找到可行计划，不是最优代价）。

**对 wiki 的映射：** 流程总览画 CEM 并行 → TO 可行性；源码时序对齐 `trajopt_parallel.py`。

## 摘录 3：实验（§V）

| 场景 | 设定 | 成功率 / 时间 |
|------|------|----------------|
| Handrail corridor | Talos 前移 3 m；手只能碰扶手、脚只能碰地；终态人工固定稳定站姿 | **20/20（100%）**，平均墙钟 **<200 s**；项目页中位约 100 s |
| Chimney 1 m | 两墙距机器人各 0.5 m；初态手/脚已贴墙（代码里手工加 stance） | **约 85%**（5 次 CEM 迭代内） |
| Chimney 3 m | 超参与接触构型数不变 | **约 50%** |
| 消融（烟囱 3 m） | elite 占种群 30% / 50% / 80% | **约 50% elites** 更早压低违反、更高短时成功机会 |

超参（两环境共用骨架）：\(N=8\)、\(M=4\)、\(K=5\)、发现可行即停。平台：Talos 1.75 m / 95 kg / 32 DoF；Pinocchio + Ipopt；Xeon Gold 5222 16 核。**优化中不检查自碰撞**。

**对 wiki 的映射：** 结论强调「分钟级可行计划」与「烟囱更高目标要更多接触构型或调参」；不要把项目页 10-run 数字和论文 20-run 混写成同一表。

## 建议 wiki 动作

- 新建 **`wiki/entities/paper-ahmp.md`**
- 交叉 [轨迹优化](../../wiki/methods/trajectory-optimization.md)、[SE(3) 表示](../../wiki/formalizations/se3-representation.md)、[FARO](../../wiki/entities/paper-faro-feasibility-aware-robot-motion-optimization.md)
