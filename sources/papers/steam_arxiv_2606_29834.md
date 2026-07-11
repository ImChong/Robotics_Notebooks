# STEAM: Self-Supervised Temporal Ensemble Advantage Modeling for Real-World Robot Learning

> 来源归档（ingest）

- **标题：** STEAM: Self-Supervised Temporal Ensemble Advantage Modeling for Real-World Robot Learning
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2606.29834>
- **PDF：** <https://arxiv.org/pdf/2606.29834>
- **机构：** 清华大学、中国科学院自动化所、Stride AI、鹏城实验室、无问芯穹（Infinigence AI）等；通讯 **Xinlei Chen、Chao Yu**
- **策略骨干：** **π₀** + **CFGRL**（Classifier-Free Guidance RL，Frans et al. arXiv:2505.23458）
- **价值模型骨干：** **SigLIP-SO400M** 视觉编码 + **Gemma-3-270M** 语言骨干 + 分箱分类头
- **开源实现：** [RLinf STEAM 文档](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/steam.html) · [RLinf/RLinf](https://github.com/RLinf/RLinf)
- **入库日期：** 2026-07-11
- **一句话说明：** 无标签帧级 advantage：在专家演示帧对上以**长度归一化时间偏移**自监督训练 **M 个时序偏移预测器**，取 **worst-of-N ensemble min** 保守打分混合质量 rollout/纠正数据，再经 **CFGRL** 提纯 **π₀**；四任务真机成功率较 BC 绝对提升 **16.2%–59%**，显著超 **RECAP** 与 **HG-DAgger**。

## 核心摘录（面向 wiki 编译）

### 1) 问题：轨迹级过滤无法处理段内质量变化

- **链接：** arXiv §1 Introduction
- **摘录要点：**
  - 真实机器人数据常混有 **停滞、纠正、次优段**；同一 rollout 可能先进展后失败，干预轨迹含差自主段 + 恢复段。
  - **轨迹级过滤** 会丢掉有用 transition 或保留有害段；需要 **帧级 advantage** 区分局部进展 vs 停滞/倒退。
  - 手工奖励、人工标注、VLM 价值模型均难扩展或在 OOD rollout 状态 **过估计 advantage**。
- **对 wiki 的映射：**
  - [STEAM](../../wiki/entities/paper-steam-advantage-modeling.md) — 问题定义与 RECAP/ARM/TimeRewarder 对照
  - [ROVE](../../wiki/entities/paper-rove-humanoid-vla-intervention.md) — 另一路「混合质量经验 → 价值引导提取」
  - [Teleoperation](../../wiki/tasks/teleoperation.md) — 干预/纠正数据采集语境

### 2) 自监督时序偏移 + 分箱 advantage

- **链接：** arXiv §3.1–3.2；Figure 2
- **摘录要点：**
  - 专家 episode \(\tau_k=(f_{k,1},\ldots,f_{k,L})\) 上帧对 \((f_{k,i},f_{k,j})\) 的 signed stride \(\Delta=j-i\)；**反序帧对** 提供伪倒退监督，无需失败演示。
  - **长度归一化** \(\tilde\Delta=\Delta\cdot L_{\max}/L_{\tau_k}\)：衡量 **时间效率** 而非绝对步数。
  - 将 \(\tilde\Delta\) 离散为 **N 个 bin**（默认 **N=32**）；SigLIP+Gemma 预测分类分布，CE 训练。
  - 固定 lookahead **H** 得标量 advantage：\(A_m=\frac{2}{N}(E[b]-b_{\mathrm{ref}})\in[-1,1]\)。
- **对 wiki 的映射：**
  - [STEAM](../../wiki/entities/paper-steam-advantage-modeling.md) — 公式与 bin/ensemble 消融表
  - [Behavior Cloning](../../wiki/methods/behavior-cloning.md) — 次优数据加权/筛选对照

### 3) Worst-of-N ensemble 抑制 OOD 过估计

- **链接：** arXiv §3.2 Eq.(5)；§4.3 Table 3
- **摘录要点：**
  - \(A_{\mathrm{STEAM}}=\min_{m=1..M} A_m\)；成员在分布内一致、OOD 处分歧 → **保守聚合** 抑制 false positive。
  - 默认 **M=3**；M=1 成功率 **72.7%** → M=3 **92.3%**（towel folding）；M=5 无进一步收益。
  - 与 RLHF reward ensemble 过优化抑制同族（Lakshminarayanan et al. 2017）。
- **对 wiki 的映射：**
  - [STEAM](../../wiki/entities/paper-steam-advantage-modeling.md) — ensemble 机制 Mermaid
  - [Online vs Offline RL](../../wiki/comparisons/online-vs-offline-rl.md) — 离线 advantage 标注语境

### 4) CFGRL 策略提纯（π₀ + 分源 quantile 二值标签）

- **链接：** arXiv §3.3；Appendix A
- **摘录要点：**
  - 对 expert \(\mathcal{D}_{\mathrm{exp}}\) 与 non-expert（rollout + 人类纠正）\(\mathcal{D}_{\mathrm{nexp}}\) **分别取 q-quantile** 阈值，得二值最优标签 \(o_{k,i}\)。
  - 高 advantage 帧作 **conditional**，低 advantage 作 **unconditional**，在 **CFGRL**（扩散/flow 策略的 classifier-free guidance 改进算子）下优化 **π₀**。
  - **无需在线环境交互**；适合真机大规模采样不可行的设定。
- **对 wiki 的映射：**
  - [STEAM](../../wiki/entities/paper-steam-advantage-modeling.md) — CFGRL 三阶段管线
  - [π₀ Policy](../../wiki/methods/π0-policy.md) — 策略骨干
  - [RLinf](../../sources/repos/rlinf.md) — OpenPI/π₀.₅ CFG 训练实现

### 5) 真机四任务与消融

- **链接：** arXiv §4；Table 1–3；Figures 4–7
- **摘录要点：**
  - **ARX 双臂：** towel folding（5 段）、chip checkout（8 段）、cola restocking（4 段）；**Franka 单臂：** pick-and-place（2 段）。
  - 数据混合：专家演示 + 自主 rollout + 人类纠正。
  - **成功率（%）：** STEAM **92.3 / 93.8 / 75 / 80** vs BC **33.3 / 39.5 / 52 / 63.8**；vs RECAP **55.6 / 53.3 / 52.9 / 53.8**。
  - **吞吐（成功 episode/h）：** STEAM 在 towel **58** vs RECAP **39**（RECAP 未滤慢进展帧）。
  - **Bin 消融：** N=2 → 27.3%；N=8 → 54.6%；N=32 → 92.3%。
  - advantage 曲线可定位 **重试、停滞、失败、人类接管后恢复**（Figure 4）。
- **对 wiki 的映射：**
  - [STEAM](../../wiki/entities/paper-steam-advantage-modeling.md) — 结果表与任务设定
  - [Manipulation](../../wiki/tasks/manipulation.md) — 长程桌面/零售操作语境

### 6) RLinf 三阶段工程管线（文档）

- **链接：** <https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/steam.html>
- **摘录要点：**
  - **Step 1：** ensemble progress critic SFT（`steam_value_model_sft.yaml`；`ensemble_size=3`，`num_bins=32`，`data.k=32`）。
  - **Step 2：** worst-of-N 推理写 `meta/advantages_{tag}.parquet`（`label_mode`: threshold 或 quantile）。
  - **Step 3：** 共享 RECAP 的 **CFG training**（`cfg_rl_openpi.sh`；`data.advantage_tag` 对齐）。
  - 数据：**LeRobot** 格式；分 **sft** 与 **rollout** 池；支持 ensemble merge / CPU relabel / 可视化脚本。
- **对 wiki 的映射：**
  - [RLinf 仓库摘录](../../sources/repos/rlinf.md)
  - [LeRobot](../../wiki/entities/lerobot.md) — 数据格式与 OpenPI 策略栈

## 当前提炼状态

- [x] arXiv 全文 + RLinf 官方 STEAM 文档已对齐摘录
- [x] wiki 映射：`wiki/entities/paper-steam-advantage-modeling.md` 新建
- [ ] 待社区复现后补独立项目页 URL（若与 RLinf 文档分叉）
