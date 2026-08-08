# DASH: Divergence-Adaptive Supervision Horizons for On-Policy Self-Distillation of Reasoning Models（arXiv:2608.06243）

> 来源归档（ingest）

- **标题：** DASH: Divergence-Adaptive Supervision Horizons for On-Policy Self-Distillation of Reasoning Models
- **缩写 / 框架：** **DASH**（Divergence-Adaptive Supervision Horizons）；基于 **OPSD**（On-Policy Self-Distillation）；对照 RLVR / GRPO
- **类型：** paper / llm-reasoning / self-distillation / rlvr / opsd
- **arXiv：** <https://arxiv.org/abs/2608.06243>（v1 2026-08-06；PDF：<https://arxiv.org/pdf/2608.06243v1>）
- **代码：** <https://github.com/DBtxy/DASH-OPSD>（已开源；归档见 [`sources/repos/dash-opsd.md`](../repos/dash-opsd.md)）
- **权重：** Hugging Face LoRA — `dbtxy/DASH-Qwen3-{1.7B,4B,8B}-LoRA`
- **作者：** ZhiYan Hou\*、Xinyu Tang\*、Hongyan An、Jianjin Zhang、Weizhen Wang、Yunyun Han、Gengsheng Li、Xiangzhao Hao、Haiyun Guo、Wenbin Hu†、Jinqiao Wang、Yafeng Deng†（\* equal；† corresponding）
- **机构：** 中国科学院自动化研究所（CASIA）；EverMind；盛大集团（Shanda Group）；中国科学院大学（UCAS）；武汉人工智能研究院（Wuhan AI Research）；武汉大学（WHU）
- **入库日期：** 2026-08-08
- **一句话说明：** 在 OPSD 已算好的师生分布上，用局部蒸馏信号相对序列均值的间隙构造 **自适应传播门**，经反向多步聚合得到路径依赖的 token 权重，三尺度数学推理全面超过匹配 OPSD 重跑且几乎不增前向开销。

## 开源状态（步骤 2.5）

- **仓库核查（2026-08-08）：** [DBtxy/DASH-OPSD](https://github.com/DBtxy/DASH-OPSD) 含 `opsd_train.py` / `opsd_trainer.py`、`scripts/run_dash_{1b,4B,8B}.sh`、`environment.yml`、`eval/`、NOTICE（基于 TRL GOLD + Zhao et al. OPSD）；HF 已挂三档 LoRA。
- **许可：** API `license` 字段为空；NOTICE 标明第三方 Apache-2.0 / MIT 依赖，本仓未单独声明 SPDX——复现时以仓库文件为准。
- **结论：** **已开源（可运行训练/评测入口 + LoRA 权重）**。wiki 须写 **源码运行时序图**。

## 摘录 1：问题与主张（§I / Abstract）

- **痛点：** RLVR 序列级稀疏奖励信用分配难；OPSD 用特权教师提供稠密 token 监督，但仍用均匀系数 \(1/T\) 聚合局部 KL，**忽略分歧时间结构**。
- **主张：** **DASH** 把 \(r_t-\bar{r}\) 映射为门 \(\lambda_t=\mathrm{sg}[\sigma(-\kappa g_t)]\)，反向 \(A_t=r_t+\lambda_t A_{t+1}\)，得到路径依赖权重；复用 OPSD 已算分布，**无需额外师生前向**。
- **结果：** Qwen3-1.7B/4B/8B × AIME24/25 + HMMT25，相对匹配 OPSD macro **+3.20 / +1.40 / +1.60**。

**对 wiki 的映射：** 升格 [`wiki/entities/paper-dash-opsd.md`](../../wiki/entities/paper-dash-opsd.md)；链 [RLVR-World](../../wiki/entities/paper-shenlan-wm-14-rlvr-world.md)、[AI Auto-Research](../../wiki/concepts/ai-auto-research.md)、EverMind 姐妹篇。

## 摘录 2：方法（§IV）

| 模块 | 要点 |
|------|------|
| **局部损失** | \(d_t=D_{\mathrm{KL}}(\pi_t^T\|\pi_t^S)\)；词汇项 clip \(\tau=0.05\) 得 \(r_t\) |
| **门** | \(g_t=r_t-\bar{r}\)，\(\lambda_t=\mathrm{sg}[\sigma(-\kappa g_t)]\)，\(\kappa=5\)；低于均值开门拉长视界 |
| **聚合** | \(A_T=r_T\)，\(A_t=r_t+\lambda_t A_{t+1}\)；\(\mathcal{L}=\frac1T\sum A_t\) |
| **结构动机** | 固定视界期望梯度含 future-divergence 轨迹项；DASH **不**估 score-function，只改编直接蒸馏系数 |
| **开销** | 重加权路径梯度，步时增幅 <1% |

**对 wiki 的映射：** 流程图 + 时序图对齐 `opsd_train.py` / `scripts/run_dash_*.sh`。

## 摘录 3：实验（§V）

| 设定 | 读点 |
|------|------|
| **数据** | OpenThoughts-Math-30K（29,434）；学生只看题，教师看参考解 |
| **训练** | LoRA r64/α128，lr \(5\times10^{-6}\)，global bs 64，200 step，max len 1024 |
| **评测** | Avg@12；thinking 模式最长 38,912 new tokens |
| **主表** | 九个 benchmark×scale 设定均为展示对比中最高；四种子均值 |
| **消融** | 固定 \(\lambda\) 已优于 OPSD，自适应再 +1.44（1.7B macro）；Inverse-gap 损害；尺度匹配后相对轮廓仍贡献 |

**对 wiki 的映射：** 「结论」强调 **时序自适应系数** 是真影响；不改变教师构造即可挂到现有 OPSD 栈。

## 局限

- 主设定短 rollout（1024）；更长推理链需另验。
- 特权教师依赖参考解，与纯 RLVR 无解设定不同。
- CITATION.cff 仍 TODO；许可 SPDX 未钉死。

## 建议 wiki 动作

- 新建 **`wiki/entities/paper-dash-opsd.md`**（含源码运行时序图）。
- 新建 **`sources/repos/dash-opsd.md`**。
- 交叉更新 RLVR-World、AI Auto-Research、HarnessBank/SkillCorpus（同 EverMind 线）。

## 当前提炼状态

- [x] 论文摘要填写
- [x] wiki 页面映射确认
- [x] 开源状态核查（已开源 + LoRA）
- [x] 源码运行时序图（实体页）
