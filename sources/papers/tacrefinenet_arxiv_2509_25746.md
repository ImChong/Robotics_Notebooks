# TacRefineNet：Goal-Conditioned Tactile Grasp Refinement for Edge-Prominent Objects

> 来源归档（ingest）

- **标题：** TacRefineNet: Goal-Conditioned Tactile Grasp Refinement for Edge-Prominent Objects
- **类型：** paper
- **机构：** 小米机器人实验室（Xiaomi Robotics）
- **原始链接：**
  - <https://arxiv.org/abs/2509.25746>（v2，2026-07-21）
  - PDF：<https://arxiv.org/pdf/2509.25746>
  - 项目页：<https://sites.google.com/view/tacrefinenet>（见 [`sources/sites/tacrefinenet-google-sites.md`](../sites/tacrefinenet-google-sites.md)）
  - 代码占位：<https://github.com/NoneJou072/tacrefinenet>（见 [`sources/repos/tacrefinenet.md`](../repos/tacrefinenet.md)）
- **入库日期：** 2026-07-23
- **一句话说明：** 面向薄板 / 圆盘 / 细杆等 **边缘突出物体**，用 **纯触觉 + 本体** 的 **Siamese 策略** 直接回归腕部位姿增量，经 **张开—移动—再抓** 外在灵巧伺服闭环做 **局部抓取精修**；**156,007** 仿真样本全程 MuJoCo 训练后 **零样本** 上真机，seen 物体真机在 \(10^\circ/10\,\mathrm{mm}\) 下固定/随机目标成功率 **80.7% / 59.3%**。

## 核心论文摘录（MVP）

### 1) 问题：边缘突出物体的「最后一公里」抓取对齐

- **链接：** <https://arxiv.org/abs/2509.25746>
- **摘录要点：** 薄板、圆盘、细杆等物体接触稀疏，指尖易遮挡，深度传感在薄/反光边缘处噪声大；传统抓取管线与 VLA 在执行末段仍易残留位姿误差，并级联到插入/装配失败。触觉直接测接触界面，适合作为精修主模态。
- **对 wiki 的映射：**
  - [TacRefineNet（论文实体）](../../wiki/entities/paper-tacrefinenet-tactile-grasp-refinement.md) — 「边缘突出物体 × 触觉伺服精修」定位。
  - [Tactile Sensing](../../wiki/concepts/tactile-sensing.md) — 压阻阵列指尖作为图像化触觉输入。

### 2) 方法：Siamese 触觉策略 + 外在灵巧 regrasp 闭环

- **链接：** <https://arxiv.org/abs/2509.25746>
- **摘录要点：** 策略 \(\Delta\mathbf{x}=\pi(\{I_i^{\mathrm{curr}},I_i^{\mathrm{target}}\}_{i=1}^{N}, q^{\mathrm{curr}}, q^{\mathrm{target}})\) 输出 6-DoF 腕部增量（回归为 \([R_9,p_3]\in\mathbb{R}^{12}\)，旋转经 SVD 正交化）。每步：闭合抓取 → 读多指触觉 → 预测腕部修正 → 张开 → 移动腕部 → 再抓。目标由示教得到的参考触觉图 + 关节配置给定；无需外部视觉或物体模型。因欠驱动（11-DoF 手、6 主动关节），用 **external dexterity**（腕部重定位 + regrasp）而非纯指间 gait。
- **对 wiki 的映射：**
  - [Visual Servoing](../../wiki/methods/visual-servoing.md) — 对偶：**触觉伺服 / image-like taxel 误差 → 腕部增量**。
  - [In-hand Reorientation](../../wiki/methods/in-hand-reorientation.md) — 对照：本文走 **外在灵巧 regrasp**，非纯手内指尖重定向。

### 3) 触觉可观 DoF 与仿真数据

- **链接：** <https://arxiv.org/abs/2509.25746>
- **摘录要点：** 每指尖 \(11\times9\) 压阻 taxel（间距约 1.1 mm）成触觉图；MuJoCo 用弹性球接触点仿真橡胶表面。仅采样 **触觉可观** 位姿维，避免 aliasing：杆 \(\{z,\mathrm{roll}\}\)、板 \(\{z,\mathrm{roll},\mathrm{pitch}\}\)、盘 \(\{y,z,\mathrm{pitch}\}\)。15 物体（三类各 5）共 **156,007** 仿真样本；**全程仿真训练、真机零样本**，无真机微调。
- **对 wiki 的映射：**
  - [TacRefineNet](../../wiki/entities/paper-tacrefinenet-tactile-grasp-refinement.md) — 「可观 DoF / aliasing」工程读点。
  - [Sim2Real](../../wiki/concepts/sim2real.md) — 压阻触觉仿真 → 真机零样本边界。

### 4) 交叉组合训练与网络结构

- **链接：** <https://arxiv.org/abs/2509.25746>
- **摘录要点：** 当前/目标触觉样本随机配对（cross-combination），监督腕部位姿差，使采样范围内任意目标无需重训。每指 ViT 编码 → 指身份位置嵌入 → 多指 self-attention 融合 → Siamese 权共享；当前作 query 对目标做 cross-attention，再与当前/目标本体拼接，经 Transformer + 平移/旋转双 MLP 头；MSE 于归一化动作空间；触觉幅值与关节加性增广。
- **对 wiki 的映射：**
  - [TacRefineNet](../../wiki/entities/paper-tacrefinenet-tactile-grasp-refinement.md) — 与「先估绝对位姿再相减」基线对照（直接 \(\Delta\) 更稳）。

### 5) 真机结果、消融与局限

- **链接：** <https://arxiv.org/abs/2509.25746>
- **摘录要点：** 双臂平台 + 11-DoF 五指手；Jetson Orin 上预处理 ~90 ms、推理 ~10 ms ≈10 Hz。Seen 真机：固定目标 \(10^\circ/10\,\mathrm{mm}\) **80.7%**，随机 **59.3%**；五步后均值误差约 **5.2 mm / 3.5°**。未见边缘特征物体真机固定/随机同准则降至 **36.7% / 13.3%**。消融：去触觉近乎崩；多指融合与 ≥3 指关键；直接 \(\Delta\) 显著优于独立绝对位姿估计基线。局限：对称/弱判别接触（如圆盘）难；仅局部可观维、非全局规划；sim2real 与未见物体仍有落差。
- **对 wiki 的映射：**
  - [OmniTacTune](../../wiki/entities/paper-omnitactune-tactile-residual-adaptation.md) — 对照：短真机 RL 残差 vs 本文仿真 BC + 零样本。
  - [T-Rex](../../wiki/entities/paper-trex-tactile-reactive-dexterous-manipulation.md) — 对照：大规模触觉 VLA mid-training vs 局部精修头。

## 开源核查（步骤 2.5）

| 项 | 状态（截至 2026-07-23） |
|----|-------------------------|
| 项目页 | <https://sites.google.com/view/tacrefinenet> — 有 Paper / Video / Code / Dataset 按钮 |
| 代码 | Code → <https://github.com/NoneJou072/tacrefinenet>，仓库 **为空占位**（无可运行 README/脚本） |
| 数据集 | Dataset 按钮当前指向 arXiv abs（与 Paper 同链），**无独立数据集发布页** |
| 结论 | **宣称将开源 / 待发布**：勿写「可复现开源」；wiki 源码运行时序图标 **不适用** |

> **版本注意：** 项目页正文仍接近 v1 叙事（sim 预训练 + 真机微调）；本归档与 wiki 以 **arXiv v2（2026-07-21）** 为准——**全程仿真训练、零样本上真机**。

## 当前提炼状态

- [x] v2 摘要、方法、数据规模、真机主结果、消融与局限已摘录
- [x] 项目页 / 空代码仓开源边界已核查并互链

## BibTeX

```bibtex
@misc{wang2026tacrefinenet,
  title={TacRefineNet: Goal-Conditioned Tactile Grasp Refinement for Edge-Prominent Objects},
  author={Shuaijun Wang and Haoran Zhou and Diyun Xiang and Yangwei You},
  year={2026},
  eprint={2509.25746},
  archivePrefix={arXiv},
  primaryClass={cs.RO},
  url={https://arxiv.org/abs/2509.25746},
}
```
