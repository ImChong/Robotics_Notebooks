# PRISM: Polynomial Representations for Interaction-Structured Motor Control（arXiv:2607.23473）

> 来源归档（ingest）

- **标题：** PRISM: Polynomial Representations for Interaction-Structured Motor Control
- **缩写：** **PRISM**
- **类型：** paper / motor-control / polynomial-representation / proprioception / rl / imitation-learning / sensorless-compliance
- **arXiv：** <https://arxiv.org/abs/2607.23473>（Submitted 2026-07-26；PDF：<https://arxiv.org/pdf/2607.23473>；HTML：<https://arxiv.org/html/2607.23473v1>）
- **项目页：** <https://lsh3163.github.io/prism/> — 归档见 [`sources/sites/lsh3163-prism-github-io.md`](../sites/lsh3163-prism-github-io.md)
- **代码：** <https://github.com/lsh3163/prism> — 归档见 [`sources/repos/prism.md`](../repos/prism.md)
- **作者：** Seung Hyun Lee、Stella X. Yu
- **机构：** University of Michigan, Ann Arbor（Computer Science and Engineering）
- **入库日期：** 2026-07-30
- **一句话说明：** 用**因式分解多项式模块**显式、可学习地暴露本体感觉变量间的乘积交互；插入 RL actor 与 Diffusion/SmolVLA 本体条件通路，在不加力/触觉传感器的前提下提升人形 locomotion 与接触丰富操作，并优于同容量更大 MLP。

## 开源状态（步骤 2.5）

- **项目页核查（2026-07-30）：** 头部 / 按钮明确链到 [lsh3163/prism](https://github.com/lsh3163/prism)。
- **仓库核查：** 含可安装 `prism_robot`、单元测试、BFM-Zero / LeRobot-SmolVLA 补丁与 `RESULTS.md` / `REPRODUCIBILITY.md`。
- **结论：** **已开源**（独立 conditioner + 上游补丁式复现路径）。上游仿真、数据与权重不随本仓分发；顶层 LICENSE 仍标注 finalize 中。

## 摘录 1：问题与主张（§1–§2）

- **痛点：** 策略观测是位置/速度/命令/历史等**一阶物理量**；功率、科氏/离心、滑移、接触冲量、柔顺等关键线索常来自**乘积/高阶交互**，标准 MLP 必须隐式发现。
- **主张：** （1）把多项式交互做成可学习表征；（2）用**因式分解**避免枚举全部单项式；（3）只改本体条件通路，部署不加传感器、不改动作接口。
- **定位对照：** 不同于显式估计物理量或加力觉/柔顺控制器；PRISM 是**架构归纳偏置**，无额外物理监督。

**对 wiki 的映射：** 升格 [`wiki/entities/paper-prism.md`](../../wiki/entities/paper-prism.md)；与 [Humanoid-Gym](../../wiki/entities/humanoid-gym.md)、[BFM-Zero](../../wiki/entities/paper-bfm-zero.md)、[Diffusion Policy](../../wiki/methods/diffusion-policy.md) 互链。

## 摘录 2：方法（§3）

- 输入划分 \(o_t=(x_t,c_t)\)：\(x_t\) 为部署可得本体/历史，\(c_t\) 为图像/语言等其余条件。
- 二阶默认形式：\(u=W_1x+b_1,\ v=W_2x+b_2,\ \psi_2=u+\alpha_2\odot(u\odot v)\)；\(\alpha\) 近零初始化 → 训练初接近线性投影。
- 递归到度数 \(K\)：\(\psi_k=\psi_{k-1}\odot\bigl(1+\alpha_k\odot(W_k x+b_k)\bigr)\)，再经投影 \(z=g_\eta(\psi_K)\)（实现里常接 MLP + RMSNorm）。
- **RL：** 本体经 PRISM 后进 actor；PPO 目标/动作空间/低层 PD 不变；特权信息仅给 critic。
- **IL：** 替换 Diffusion Policy / SmolVLA 的线性本体条件层，视觉–语言–动作通路不变。

**对 wiki 的映射：** 实体页「流程总览」+「源码运行时序图」对齐 `PRISMConditioner` 与 integrations 入口。

## 摘录 3：实验要点（§4 / 项目页）

| 设定 | 关键数字 / 读法 |
|------|----------------|
| Humanoid-Gym | 生存率 **92.5%** vs MLP 51.0 / Larger MLP 52.25（参数与 PRISM 对齐仍拉不开）→ **结构 > 容量** |
| LIBERO + DP | 成功率 **91.0%** vs DP 63.8 / MCC-Sensorless 47.8 / MCC-Oracle 64.5 → 无 force 输入仍可超 Oracle 柔顺基线 |
| BFM-Zero | tracking EMD Mean **1.224** vs 1.269 / Larger 1.264（Nominal/低摩擦/载荷均降） |
| SmolVLA | LIBERO Avg **66.55**（+3.05）vs Larger +1.40；Long 套件 **53.4** |
| 探针 | joint power MSE −14.0%；contact impulse MSE −19.9%；slip PCC +9.6% |
| 涌现交互 | locomotion 上 velocity–memory / cross-joint velocity 等乘积项对动作敏感（post-hoc 命名） |

**对 wiki 的映射：** 「结论」强调：选型时优先试多项式本体条件，而非盲目加宽 MLP 或加部署力觉。

## 建议 wiki 动作

- 新建 **`wiki/entities/paper-prism.md`**（含流程总览 + 源码运行时序图 + 结论）。
- 新建 **`sources/sites/lsh3163-prism-github-io.md`**、**`sources/repos/prism.md`**。
- 交叉：[paper-bfm-zero](../../wiki/entities/paper-bfm-zero.md)、[humanoid-gym](../../wiki/entities/humanoid-gym.md)、[diffusion-policy](../../wiki/methods/diffusion-policy.md)、[manipulation](../../wiki/tasks/manipulation.md) / [locomotion](../../wiki/tasks/locomotion.md) 轻量回链。
