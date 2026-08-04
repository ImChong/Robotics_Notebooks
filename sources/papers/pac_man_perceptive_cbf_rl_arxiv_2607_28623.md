# PAC-MAN: Perception-Aware CBF-RL for Whole-Body Safety in Humanoid Dodgeball（arXiv:2607.28623）

> 来源归档（ingest）

- **标题：** PAC-MAN: Perception-Aware CBF-RL for Whole-Body Safety in Humanoid Dodgeball
- **缩写：** **PAC-MAN**
- **类型：** paper / humanoid perception-aware CBF-RL + dodgeball
- **arXiv：** <https://arxiv.org/abs/2607.28623>
- **PDF：** <https://arxiv.org/pdf/2607.28623>
- **项目页：** <https://lzyang2000.github.io/perceptive_cbf_rl/>
- **浏览器 Demo：** <https://lzyang2000.github.io/perceptive_cbf_rl/demo/>
- **代码：** <https://github.com/lzyang2000/perceptive_cbf_rl>
- **发表日期：** 2026（arXiv preprint）
- **作者：** Lizhi Yang, Junheng Li, Aaron D. Ames
- **机构：** 加州理工学院（Caltech）AMBER Lab
- **入库日期：** 2026-08-01
- **一句话说明：** 训练期用全身 Link/Joint-CBF 指导 + AMP 风格先验，部署只吃机载分割掩膜深度与本体感觉；Unitree G1 零样本躲避球 **19/20**、**0** 跌倒。

## 核心论文摘录（MVP）

### 1) 问题：机载感知下的全身反应式安全（Abstract / §I）

- **链接：** <https://arxiv.org/abs/2607.28623>
- **核心贡献：** 人形躲避球把「短时窗感知 + 全身避碰 + 保持平衡」压成安全关键任务。经典 CBF 需要威胁状态；部署时球可能短暂可见、出视野或只剩稀疏深度像素。PAC-MAN 主张把 **威胁表征与屏障设计当成耦合问题**：策略只能内化其观测支撑得住的安全结构。
- **对 wiki 的映射：**
  - [PAC-MAN 论文实体](../../wiki/entities/paper-pac-man-perceptive-cbf-rl.md)
  - [Control Barrier Function](../../wiki/concepts/control-barrier-function.md)
  - [Safe RL](../../wiki/methods/safe-rl.md)
  - [Safety Filter](../../wiki/concepts/safety-filter.md)

### 2) Link-CBF vs Joint-CBF：训练期屏障强度分级（§III-B）

- **核心贡献：**
  - **Link-CBF（部署配置）**：每连杆 clearance $h_i=\lVert p^b-p_i\rVert-(\rho^b+\rho_i)$，用最紧约束的 $\min_i\mathrm{clip}(\dot h_i+\alpha h_i,-c,0)$ 作奖励；**仅训练期**进入回报，部署无在线滤波器。
  - **Joint-CBF（指导 / 特权滤波）**：对最受威胁点做关节速度半空间投影 $v^\star$，训练时加校正/缓冲代价；`+filter` 在测试期保留投影，但需 **真值球状态**，作仿真安全上限而非真机配置。
  - 总回报：$r_t=r^{\mathrm{core}}_t+r^{\mathrm{cbf}}_t+\lambda_s r^{\mathrm{style}}_t$（core 为骨盆距离项；style 为 AMP）。
- **对 wiki 的映射：**
  - [Control Barrier Function](../../wiki/concepts/control-barrier-function.md)
  - [Safety Filter](../../wiki/concepts/safety-filter.md)
  - [Privileged Training](../../wiki/concepts/privileged-training.md)
  - [AMP](../../wiki/methods/amp-reward.md)

### 3) 感知制度：掩膜深度把 Sim2Real 压到感知层（§III-C）

- **核心贡献：** 部署观测仅含机载可得信号：分割掩膜后的深度 $\mathcal{D}_t$ + 关节/角速度/投影重力/上一步动作；球状态永不进策略输入。仿真用分割像素保留球深度、其余置远平面，池化到 **16×9**，稀疏时间堆叠 `[0,3,8,18]`（50 Hz）。真机用 **EfficientTAM** 分割 + ZED 深度，复现同一观测契约。固定相机 / 主动云台 / 状态 oracle 三档对照说明：**更强屏障需要更高可观测性**。
- **对 wiki 的映射：**
  - [Sim2Real](../../wiki/concepts/sim2real.md)
  - [人形策略观测输入](../../wiki/concepts/humanoid-policy-observation-inputs.md)
  - [Unitree G1](../../wiki/entities/unitree-g1.md)

### 4) 评测与真机：any-link 接触 + 零样本 G1（§IV）

- **核心贡献：**
  - 成功定义：任意连杆不得被球接触，且不跌倒；比仅护骨盆更严。
  - 仿真（固定相机 · deployment）：Link-CBF **89%**，优于 no-barrier **86%** 与裸 Joint-CBF **76%**；`+filter` 需特权球态。
  - 真机：固定相机 Link-CBF 零样本上 Unitree G1，**19/20（95%）** 手动投掷躲开、**0** 跌倒；语义分割使同一策略可躲不同球。
  - 训练栈：mjlab + rsl_rl PPO/AMP；约 100 s 重定向人类躲避动作作 prior。
- **对 wiki 的映射：**
  - [mjlab](../../wiki/entities/mjlab.md)
  - [AMP_mjlab](../../wiki/entities/amp-mjlab.md)
  - [CLF vs CBF](../../wiki/comparisons/clf-vs-cbf.md)

## 开源状态（2026-08-01 项目页核查）

- **已开源（MIT）**：项目页列 Code → [lzyang2000/perceptive_cbf_rl](https://github.com/lzyang2000/perceptive_cbf_rl)；含训练、benchmark、硬件 `deploy/`（ZED + EfficientTAM + ONNX + Unitree DDS）与 `deploy/ckpts/dodge_link_cbf.onnx`。
- 归档：[`sources/sites/perceptive-cbf-rl-github-io.md`](../sites/perceptive-cbf-rl-github-io.md)、[`sources/repos/perceptive_cbf_rl.md`](../repos/perceptive_cbf_rl.md)。

## 对 wiki 的映射（汇总）

- 沉淀实体页：[`wiki/entities/paper-pac-man-perceptive-cbf-rl.md`](../../wiki/entities/paper-pac-man-perceptive-cbf-rl.md)
- 交叉升级：
  - [Control Barrier Function](../../wiki/concepts/control-barrier-function.md)
  - [Safe RL](../../wiki/methods/safe-rl.md)
  - [Safety Filter](../../wiki/concepts/safety-filter.md)
  - [Privileged Training](../../wiki/concepts/privileged-training.md)
  - [AMP](../../wiki/methods/amp-reward.md)
  - [Unitree G1](../../wiki/entities/unitree-g1.md)
  - [AMP_mjlab](../../wiki/entities/amp-mjlab.md)
  - [mjlab](../../wiki/entities/mjlab.md)

## 其他公开资料

- 项目页：<https://lzyang2000.github.io/perceptive_cbf_rl/>
- 浏览器 Demo：<https://lzyang2000.github.io/perceptive_cbf_rl/demo/>
- GitHub：<https://github.com/lzyang2000/perceptive_cbf_rl>
- arXiv：<https://arxiv.org/abs/2607.28623>
