# Cepheid: Decoupling Scale from Structure for Cross-Family Motion Completion（arXiv:2610.18871）

> 来源归档（ingest）

- **标题：** Cepheid: Decoupling Scale from Structure for Cross-Family Motion Completion
- **arXiv：** <https://arxiv.org/abs/2610.18871>（HTML：<https://arxiv.org/html/2610.18871v1>，31 页）
- **代码（官方）：** <https://github.com/Su-Terry/cepheid>（归档：[cepheid.md](../repos/cepheid.md)）
- **项目页：** <https://su-terry.github.io/cepheid-website/>（归档：[cepheid-project.md](../sites/cepheid-project.md)）
- **作者：** Chih-Hai Su、Tsung-Wei Ke、Yu-Lun Liu；机构：国立阳明交通大学（Su、Liu）、国立台湾大学（Ke），依据项目页
- **入库日期：** 2026-10-17
- **来源线索：** 具身智能小站 2026-10-17 文章（<https://mp.weixin.qq.com/s/I4XP5JSG9QNmm4hd1nk2JQ>）
- **一句话说明：** 谐波延拓基底 + 无量纲残差先验 + 由观测估计的幅值，跨几何族补全稠密网格运动。

## 开源状态（步骤 2.5）

- **项目页核查（2026-10-17）：** Code 链接指向 `Su-Terry/cepheid`；另链接 Motion 3-to-4（CC BY-NC-SA 4.0）；无数据 / HF 链接；无 coming soon。
- **仓库核查（2026-10-17）：** README 含 Quick start、模型库、局限与引用；目录 `cepheid/`、`scripts/`、`reproduce/`、`configs/`、`tools/` 等；Hub 托管统一模型与 11 个论文检查点；代码 MIT，权重仅限非商用研究；数据集不再分发；2 次提交、8 星（页面所示）。本次未运行代码。
- **结论：** **已开源**（库 + 权重 + 复现脚本；基线 DOMA / NSDP 与 GenBot1K 回放未含）。

## 摘录 1：方法

- 谐波延拓 H；残差 R=F−H；ŝ=σ_obs·exp(b+wᵀz)，z 为 11 维无量纲特征，岭回归在 DT4D / AMASS / Franka 拟合后冻结；MGN-T 骨干，OT-CFM 训练，30 步 Euler，k=8 平均。

**对 wiki 的映射：** [`paper-cepheid.md`](../../wiki/entities/paper-cepheid.md)「核心原理」。

## 摘录 2：结果

- DT4D 相对误差较谐波低 44%–81%；转移矩阵布料 0.3692 对 0.4377、腿式 0.2225 对 0.2474；去谐波延拓布料 1.047；去幅值恢复腿式 0.409。
- 软棒抓取（Isaac Lab，5,525 点，8 个观测点）与动物视频（12 轨迹，补全 120 点）均为定性。

**对 wiki 的映射：** 「实验与评测」「结论」。

## 摘录 3：局限

- 精确位移假设；训练仅随机掩码；视频例无真值。

**对 wiki 的映射：** 「局限与风险」。
