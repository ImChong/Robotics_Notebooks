# arXiv:2606.29834 — STEAM: Self-Supervised Temporal Ensemble Advantage Modeling for Real-World Robot Learning

> 原始论文归档（含 RLinf 官方源码与文档互证）

- **标题：** STEAM: Self-Supervised Temporal Ensemble Advantage Modeling for Real-World Robot Learning
- **作者：** Zhihao Liu\*, Qiuyi Gu\*, Yu Zhang\*, Kang Chen, Wenbin Li, Jiahang Cao, Kang Li, Yuxiang Gao, Yang Wang, Hongzhi Zhai, Jingyu Liu, Shuo Liu, Hongjie Liu, Yuxin Chen, Jianwei Sun, Jiaming Liu, Chao Yu†, Xinlei Chen†, Jing Xu, Dong Wang, Shanghang Zhang, Feng Gao, Yi Wu, Bo Xu, Tiejun Huang（CASIA、Tsinghua、UCAS、Zhongguancun Academy、Peng Cheng Laboratory、HIT、BIT、ZJU、PKU、Infinigence AI、Striding AI）
- **链接：** <https://arxiv.org/abs/2606.29834> （本文核对版为 arXiv v2，2026-10-08；含 `2606.29834.txt` 与 `2606.29834.html`）
- **项目页：** <https://rlinf.github.io/steam/>
- **代码仓库：** <https://github.com/RLinf/RLinf>（已开源，Apache-2.0；离线 RL Advantage Modeling + CFG-RL 全流程集成）
- **通读：** 已完成
- **核对范围：** 已通读 `2606.29834.txt` 全文（Abstract、Sec. 1–6、Eq. 1–7、Table 1–4、Fig. 1–5 文本与 Appendix A.1–A.7）；并核对官方开源仓库 `RLinf/RLinf` 中 `docs/source-en/rst_source/examples/embodied/steam.rst`、`publications/steam.rst`、`examples/offline_rl/advantage_labeling/steam/train_steam.py`、`process/compute_advantages_ensemble.py`、`examples/offline_rl/policy_optimization/cfg_rl/train_cfg.py`、`examples/offline_rl/config/{steam_value_model_sft,steam_compute_advantages_ensemble,cfg_rl_openpi}.yaml`、`rlinf/workers/sft/{fsdp_steam_sft_worker,fsdp_cfg_worker}.py`、`rlinf/data/datasets/steam/{pair_dataset,binning}.py` 及 `rlinf/models/embodiment/value_model/steam/{ensemble_modeling_critic,modeling_steam}.py`。

---

## 核心摘要与问题定位

真实世界机器人部署中往往能以较低成本积累大量「专家演示 + 人工纠偏（HG-DAgger）+ 自主 rollout（含成功与失败）」混合质量轨迹数据。传统离线强化学习（如 RECAP / π\*₀.₆ 路线）需要拟合任务进度值函数 $V_\psi(o_t, \ell)$ 并回推时间差分优势；当专家演示长度分布集中而失败轨迹、卡死片段大量存在时，值函数极易把状态与伪时间相关性混为一谈，导致退化停滞片段获得过高优势估计，从而污染下游策略优化。

**STEAM** 提出绕开状态值函数拟合，**仅在专家演示集 $\mathcal{D}_{\text{exp}}$ 上以自监督方式直接学习帧对相对时间步差分布**，再通过保守集成（worst-of-$M$ minimum）识别分布外低质量行为，为混合数据集输出可信帧级优势信号并驱动 CFGRL（Classifier-Free Guidance RL）策略微调。

---

## 方法论与数学推导（原文 Sec. 3, Eq. 1–7）

1. **专家帧对构造与双向采样（Eq. 1）：**
   仅从成功专家演示 $\tau_k \in \mathcal{D}_{\text{exp}}$（长度 $L_{\tau_k}$）采样帧对 $(f_{k,i}, f_{k,j})$，偏移 $\Delta = j - i$，采样范围为 $i, j \in \{0,\dots,L_{\tau_k}-1\}$ 且 $1 \le |j - i| \le k_{\max}$（主实验取 $k_{\max} = H = 32$ 步，约 20 fps 下 1.6 秒），同时覆盖正向（$\Delta > 0$）与逆向（$\Delta < 0$，表示退步）。

2. **轨迹长度归一化与连续分箱（Eq. 2–3）：**
   $$\tilde{\Delta} = \Delta \cdot \frac{L_{\max}}{L_{\tau_k}}, \qquad b(\tilde{\Delta}) = \operatorname{clip}\!\left(\left\lfloor \frac{\tilde{\Delta} + \tilde{\Delta}_{\max}}{2\tilde{\Delta}_{\max}} \cdot N \right\rfloor,\ 0,\ N-1\right)$$
   其中 $L_{\max} = \max_{\tau \in \mathcal{D}_{\text{exp}}} L_\tau$，$\tilde{\Delta}_{\max}$ 覆盖最大缩放偏移，$N = 32$ 离散分箱；模型主干采用 `SigLIP-SO400M` 视觉编码器 + `Gemma-3-270M` 多模态主干，交叉熵训练条件分类头 $p_\theta(\Delta \mid f_{k,i}, f_{k,j}, \ell)$。

3. **无基准时间进度与 H 步优势关系（Eq. 4–5）：**
   推理时固定锚点与前景窗口 $j = \min(i + H, L_\tau - 1)$，由期望离散分箱计算无量纲进度估计，并在与绝对时间均值对照后证明其线性正比于截断 $H$ 步优势 $A^{(H)}(s_{k,i})$：
   $$A(f_{k,i}; \theta) = \frac{2}{N}\!\left(\mathbb{E}_{b \sim p_\theta}[b] - \tilde{\Delta}^B_{\tau_{\max}}(i, i+H)\right) \propto \frac{A^{(H)}(s_{k,i})}{\tilde{\Delta}_{\max}}$$

4. **保守集成最坏值聚合（Worst-of-$M$ Min，Eq. 6）：**
   使用 $M = 3$ 个不同随机种子独立初始化并训练的预测器，推理时对每个帧对取最小估计作为最终保守优势：
   $$A_{\text{STEAM}}(f_i) = \min_{m \in \{1,\dots,M\}} A(f_i; \theta_m)$$
   论文 Figure 4 验证：在毛巾折叠重试停滞帧（1200–1350）上，单个成员可能对 OOD 状态给出接近 $+1$ 的假阳性高分，而其余成员将其压低，取 $\min$ 可将整段退化过程压到 $-0.5$ 以下。

5. **分池分位数二值化与 CFGRL 策略优化（Eq. 7 & App. A.2）：**
   将专家池（$q^{\text{exp}} = 0.8$，保留 Top-80%）与非专家池（纠偏 + 自主 rollout，$q^{\text{non-exp}} = 0.3$，保留 Top-30%）**分开按分位数阈值化**转为正/负条件标签 $I_t \in \{0, 1\}$，避免混合分布相互挤压。下游基于 $\pi_0$（flow matching VLA）以 Classifier-Free Guidance 联合训练：$p_{\text{drop}} = 0.1$，推理引导系数 $w = 2.5$，`batch_size = 512`，`lr = 5e-5`，优化 `30,000` 步，动作块长度 $H = 32$。

---

## 实验配置与主要定量结果（v2 核对版）

### 硬件与数据规模（Sec. 4 & App. A.1–A.4）
- **平台：** ARX 双臂真机 + 3 路 Orbbec Astra Pro RGB 相机（头视 + 左/右腕部），绝对末端执行器位姿控制，20 Hz。
- **对比方法评测（Table 1 — 优势信号诊断，离线片段评测）：**
  - Clustering Accuracy / I-AUROC：ReWiND `63.3% / 53.3%`，TimeRewarder `88.3% / 65.1%`，RoboMeter `75.0% / 45.5%`，RECAP `60.0% / 48.3%`，**STEAM `93.3% / 78.8%`**（I-AUROC 超第二名 13.7 个百分点）。
- **四项双臂真机任务评测（Table 2，所有方法统一底层 $\pi_0$ + CFGRL，记录 `Succ (%) / Prog / Throughput (ep/h)`）：**
  1. **Towel Folding**（5 阶段；240 专家 + 125 rollout + 20 纠偏 = 385 ep）：TimeRewarder `53.3 / 3.9 / 27`，RoboMeter `60.0 / 4.1 / 30`，RECAP `55.6 / 2.9 / 39`，**STEAM `92.3 / 4.9 / 58`**
  2. **Chip Checkout**（8 阶段；200 专家 + 63 rollout + 20 纠偏 = 283 ep）：TimeRewarder `70.6 / 5.7 / 30`，RoboMeter `58.8 / 4.8 / 24`，RECAP `53.3 / 5.3 / 24`，**STEAM `93.8 / 7.5 / 48`**
  3. **Table Clearing**（21 阶段长程；203 专家 + 94 rollout + 42 纠偏 = 339 ep）：TimeRewarder `46.7 / 14.5 / 20`，RoboMeter `33.3 / 12.3 / 13`，RECAP `36.4 / 12.5 / 15`，**STEAM `72.7 / 17.7 / 37.5`**
  4. **Cola Restocking**（4 阶段；89 专家 + 49 rollout + 27 纠偏 = 165 ep）：TimeRewarder `64.7 / 2.6 / 64`，RoboMeter `60.0 / 2.7 / 57`，RECAP `52.9 / 2.1 / 46`，**STEAM `75.0 / 3.0 / 90`**
  - **四任务平均成功率：** TimeRewarder `58.8%`，RoboMeter `53.0%`，RECAP `49.6%`，**STEAM `83.5%`**。
- **关键消融与补充实验（Tables 3–4, App. A.5–A.7）：**
  - 离散分箱数 $N$（Towel Folding）：$N=2$ `27.3% / 2.8 / 41`，$N=8$ `54.6% / 3.8 / 51`，$N=32$ `92.3% / 4.9 / 58`。
  - 集成规模 $M$（Towel Folding）：$M=1$ `72.7% / 3.9 / 53`，$M=3$ `92.3% / 4.9 / 58`，$M=5$ `90.9% / 4.6 / 55`。
  - 前瞻窗口（App. A.5，Towel Folding）：$H=16$ 降至 `80.0%`（vs $H=32$ 的 `92.3%`）。
  - 策略更新器对比（App. A.6，Towel Folding）：BC `33.3%`，STEAM+AWR `73.3%`（原文 Table 5? 实际 App. A.6 文本），STEAM+CFGRL `92.3%`。
  - 数据配比洞察（Fig. 5）：全四项双臂任务上，STEAM (Exp) 均优于 BC 与 HG-DAgger，加入纠偏（Exp+Dagg）及自主 rollout（Full）后持续上升。
  - Franka 单臂短程 Pick-and-Place（App. A.7，2 阶段，50 专家 + 594 rollout，$H=k_{\max}=N=16$）：BC `63.8%`，STEAM (Exp) `55.0%`，RECAP `53.8%`，STEAM (Full) `80.0%`——揭示在短程干净但样本极少时，仅靠筛选专家集不如补入海量 rollout 有效。

---

## RLinf 开源实现与文档版本差异核对

- **开源代码定位：** `RLinf/RLinf` 完整开源了 STEAM 离线三阶段管线（文档位于 `docs/source-en/rst_source/examples/embodied/steam.rst` 与 `publications/steam.rst`）：
  1. **Phase 1 — 集成优势模型自监督训练：** `examples/offline_rl/advantage_labeling/steam/run_steam_sft.sh` 调用 `train_steam.py`（配置 `examples/offline_rl/config/steam_value_model_sft.yaml`），通过 `SFTRunner` 启动 `FSDPSteamSftWorker`，加载 `PairDataset`/`PairMixtureDataset` 与 `rlinf/data/datasets/steam/binning.py`，训练 `EnsembleSteamCriticModel`（位于 `rlinf/models/embodiment/value_model/steam/ensemble_modeling_critic.py`）。
  2. **Phase 2 — 集成保守推理与分池打标：** `examples/offline_rl/advantage_labeling/steam/process/run_compute_advantages_ensemble.sh` 调用 `compute_advantages_ensemble.py`（配置 `steam_compute_advantages_ensemble.yaml`），经 `BinaryPairInferenceDataset` 推理并聚合 `member_predicted_values.min(dim=0)`，按 `sft` / `rollout` 池各自设定分位数阈值输出 `meta/advantages_{tag}.parquet` 与 `mixture_config.yaml`。
  3. **Phase 3 — CFG-RL 策略微调：** `examples/offline_rl/policy_optimization/cfg_rl/run_cfg_rl.sh` 调用 `train_cfg.py`（配置 `examples/offline_rl/config/cfg_rl_openpi.yaml`），启动 `FSDPCfgWorker` 与 `CfgMixtureDataset` 读取优势标签训练策略。
- **版本核对备注：**
  1. RLinf 文档 `examples/embodied/steam.rst` 发布时间早于 arXiv v2 修订：文档正文仍展示 v1 四任务中的 `Pick-and-Place`（而 arXiv v2 主表已改为 21 阶段 `Table Clearing`，并将单臂 `Pick-and-Place` 移至 Appendix A.7），且仅列出了 `BC` 与 `RECAP` 对比；arXiv v2 补充了 `TimeRewarder`、`RoboMeter` 与 `ReWiND` 基线并更新了 Chips Checkout 吞吐量统计（`48` vs 文档 `47.5`）。
  2. RLinf 开源默认 CFG 配置 `examples/offline_rl/config/cfg_rl_openpi.yaml` 默认引用 `model/pi0_5`（$\pi_{0.5}$），而论文真机对照实验均固定使用 $\pi_0$ 主干。
