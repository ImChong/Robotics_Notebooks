# TacRefineNet：Goal-Conditioned Tactile Grasp Refinement for Edge-Prominent Objects

> 来源归档（ingest · arXiv:2509.25746v2 全文核对）

- **标题：** TacRefineNet: Goal-Conditioned Tactile Grasp Refinement for Edge-Prominent Objects
- **类型：** paper（已发表于 IEEE Robotics and Automation Letters / RA-L）
- **机构：** 小米机器人实验室（Xiaomi Robotics）
- **原始链接：**
  - <https://arxiv.org/abs/2509.25746>（v2，2026-07-21）
  - HTML 全文：<https://arxiv.org/html/2509.25746>
  - PDF：<https://arxiv.org/pdf/2509.25746>
  - 项目页：<https://sites.google.com/view/tacrefinenet>（[`sources/sites/tacrefinenet-google-sites.md`](../sites/tacrefinenet-google-sites.md)）
  - 代码仓：<https://github.com/NoneJou072/tacrefinenet>（[`sources/repos/tacrefinenet.md`](../repos/tacrefinenet.md)，2026-08-07 已公开训练与仿真部署代码）
- **入库日期：** 2026-07-23
- **核对范围：** 通读 `arXiv:2509.25746v2` HTML 全文（§I–VI、Table I–V、Fig. 1–8）、项目页 <https://sites.google.com/view/tacrefinenet> 与官方仓库 `NoneJou072/tacrefinenet@8cabb33`。未在真机/GPU 环境运行代码。
- **通读：** 已完成
- **一句话说明：** 面向薄板 / 圆盘 / 细杆等 **边缘突出物体**，用 **纯触觉 + 本体** 的 **Siamese 策略** 直接回归腕部位姿增量，经 **张开—移动—再抓** 外在灵巧伺服闭环做 **局部抓取精修**；**156,007** 仿真样本全程 MuJoCo 训练后 **零样本** 上真机，seen 物体真机在 \(10^\circ/10\,\mathrm{mm}\) 下固定/随机目标成功率 **80.7% / 59.3%**。

## 核心论文摘录

### 1) 问题：边缘突出物体的「最后一公里」抓取对齐（§I）

- **摘录要点：** 薄板、圆盘、细杆等物体接触稀疏，指尖易遮挡，深度传感在薄/反光边缘处噪声大；传统抓取管线与 VLA 在执行末段仍易残留位姿误差，并级联到插入/装配失败。触觉直接测接触界面，适合作为精修主模态。

### 2) 方法：Siamese 触觉策略 + 外在灵巧 regrasp 闭环（§III-A, §III-D）

- **摘录要点：** 策略 \(\Delta\mathbf{x}=\pi(\{I_i^{\mathrm{curr}},I_i^{\mathrm{target}}\}_{i=1}^{N}, q^{\mathrm{curr}}, q^{\mathrm{target}})\) 输出 6-DoF 腕部增量（回归为 \([R_9,p_3]\in\mathbb{R}^{12}\)，旋转经 SVD 对称正交化）。每步：闭合抓取 → 读多指触觉 → 预测腕部修正 → 张开 → 移动腕部 → 再抓。目标由示教得到的参考触觉图 + 关节配置给定；无需外部视觉或物体模型。因欠驱动（11-DoF 手、6 主动关节），用 **external dexterity**（腕部重定位 + regrasp）而非纯指间 gait。

### 3) 触觉可观 DoF 与仿真数据（§III-B, §III-C）

- **摘录要点：** 每指尖 \(11\times9\) 压阻 taxel（间距约 1.1 mm）成触觉图；MuJoCo 用弹性球接触点仿真橡胶表面。仅采样 **触觉可观** 位姿维，避免 aliasing：杆 \(\{z,\mathrm{roll}\}\)、板 \(\{z,\mathrm{roll},\mathrm{pitch}\}\)、盘 \(\{y,z,\mathrm{pitch}\}\)。15 物体（三类各 5）共 **156,007** 仿真样本（90%/10% 划分）；**全程仿真训练、真机零样本**，无真机微调。

### 4) 交叉组合训练与网络结构（§III-D, Table I）

- **摘录要点：** 当前/目标触觉样本随机配对（cross-combination），监督腕部位姿差，使采样范围内任意目标无需重训。每指 ViT 编码 → 指身份位置嵌入 → 多指 self-attention 融合 → Siamese 权共享；当前作 query 对目标做 cross-attention，再与当前/目标本体拼接，经 Transformer + 平移/旋转双 MLP 头；MSE 于归一化动作空间；触觉幅值缩放 \(\sim\mathcal{U}(0.8, 1.2)\) 与关节加性噪声 \(\sim\mathcal{U}(-0.04, 0.04)\)。

### 5) 真机结果、消融与局限（§IV–V, Table II–V）

- **摘录要点：** 双臂平台 + 11-DoF 五指手；Jetson Orin 上预处理 ~90 ms、推理 ~10 ms ≈10 Hz。Seen 真机：固定目标 \(10^\circ/10\,\mathrm{mm}\) **80.7%**（板 84 / 盘 70 / 杆 88），随机 **59.3%**（72 / 56 / 50）；五步后均值误差固定 **5.3 mm / 3.4°**、随机 **5.1 mm / 3.5°**。未见边缘特征物体真机固定/随机同准则降至 **36.7% / 13.3%**（误差 6.0 mm / 5.1° 与 6.1 mm / 4.7°）。消融：去触觉近乎崩（loss 1.89→100.3×10⁻²）；多指融合（73.6% vs 83.7%）与 ≥3 指关键；直接 \(\Delta\) 显著优于独立绝对位姿估计基线（85.9% vs 54.1%）。局限：对称/弱判别接触（如圆盘）难；仅局部可观维、非全局规划；sim2real 与未见物体仍有落差。

## 开源核查（步骤 2.5，2026-11-22 复核）

| 项 | 状态 |
|----|------|
| 项目页 | <https://sites.google.com/view/tacrefinenet> — 含 Website / Video / Paper / Dataset / Code |
| 代码仓 | <https://github.com/NoneJou072/tacrefinenet> — **已开源**（`8cabb33`，2026-08-07 推送完整 DDP 训练 `mivt/scripts/train.py`、MuJoCo 仿真部署评测 `mivt/scripts/deploy_sim.py` 与 `Dockerfile`） |
| 资产 / 模型 | README 提供 Google Drive `assets/` 下载（含预训练 checkpoint、处理后数据集与部分物体模型；因商业保密仅公开部分物体模型） |

## 对 wiki 的映射

- [TacRefineNet（论文实体）](../../wiki/entities/paper-tacrefinenet-tactile-grasp-refinement.md)
- [Tactile Sensing](../../wiki/concepts/tactile-sensing.md) / [Visual Servoing](../../wiki/methods/visual-servoing.md) / [Sim2Real](../../wiki/concepts/sim2real.md)
