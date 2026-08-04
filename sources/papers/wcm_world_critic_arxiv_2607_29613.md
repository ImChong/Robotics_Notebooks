# WCM: A World Critic Model for Vision-Language-Action Reinforcement Learning（arXiv:2607.29613）

> 来源归档（ingest）

- **标题：** WCM: A World Critic Model for Vision-Language-Action Reinforcement Learning
- **缩写 / 框架：** **WCM**（World Critic Model）
- **类型：** paper / vla / rl-post-training / critic / world-model / jepa
- **arXiv：** <https://arxiv.org/abs/2607.29613>（v1，Submitted 2026-07-31，cs.RO，CC BY 4.0；PDF：<https://arxiv.org/pdf/2607.29613>）
- **代码：** <https://github.com/sylvestf/WCM>（MIT）— 归档见 [`sources/repos/wcm-world-critic-model.md`](../repos/wcm-world-critic-model.md)
- **项目页：** <https://sylvestf.github.io/wcm-homepage/> — 归档见 [`sources/sites/sylvestf-wcm-homepage.md`](../sites/sylvestf-wcm-homepage.md)
- **权重 / 数据：** <https://huggingface.co/collections/Sylvest/wcm>
- **作者：** Senyu Fei、Xiaopeng Yu、Siyin Wang、Xianzhong Zhao、Jingjing Gong、Xipeng Qiu
- **机构：** 同济大学（Tongji）、上海创智学院（Shanghai Innovation Institute）、复旦大学（Fudan）
- **入库日期：** 2026-08-04
- **一句话说明：** VLA 的 RL 后训练里，critic 普遍只吃**单帧**观测或单帧 VLM 隐层，与机器人控制的**部分可观测**本质错配；WCM 用轻量 **LeJEPA** 架构，让 critic **同时预测未来隐状态与估计价值**，把时序动力学显式写进 critic 表征，而不是只靠标量回报回归。

## 开源状态（步骤 2.5）

- **项目页核查（2026-08-04）：** 主页列出 arXiv / GitHub / Hugging Face 三条链接。
- **仓库核查：** `github.com/sylvestf/WCM`，MIT，约 40 commits；含 `world_critic/`（核心实现）、`configs/`、`scripts/`、`episode_value_video/`，以及 `1_add_returns.sh`（数据预处理）/ `2_run_train.sh`（1 或 8 GPU 训练）/ `3_run_eval.sh` / `4_gen_video.sh`（价值曲线可视化）四步 shell 入口。
- **结论：** **部分开源**——训练 / 评测 / 可视化代码完整；pick-and-place 与 LIBERO-Plus 相关权重与数据已上 HF，**其余 checkpoint 论文写明「逐步开源」**。RL 训练还需各自的 VLA 主干（π₀ / π₀.₅ / OpenVLA-OFT）与其原生训练栈。

## 摘录 1：问题诊断 —— critic 的「状态近似问题」

- **现状：** critic-based 的 VLA RL（相对 GRPO 一类 critic-free 方法）依赖价值估计器，但主流实现把 critic 建在**单帧**观测或单帧 VLM backbone latent 上。
- **错配：** 机器人控制是 POMDP，单帧不足以定状态。
- **朴素补法为什么不行：** 直接把观测历史塞进 critic，在高维视觉空间里复杂度爆炸；更关键的是**纯标量回报回归提供的监督太弱**，学不出跨时间的动力学结构。
- **论文诊断：** 根因是 **state approximation problem** —— 没有显式的世界建模目标，critic 表征就抓不到价值估计所需的时间结构。

**对 wiki 的映射：** 升格 [`wiki/entities/paper-wcm-world-critic-model.md`](../../wiki/entities/paper-wcm-world-critic-model.md)；与 [model-based-rl](../../wiki/methods/model-based-rl.md)、[generative-world-models](../../wiki/methods/generative-world-models.md)、[online-vs-offline-rl](../../wiki/comparisons/online-vs-offline-rl.md) 互链。

## 摘录 2：LeJEPA 架构与三项损失

四个组件：

1. **观测编码器** — 逐帧独立编码为 latent（ViT 或 VLM backbone）
2. **语言条件** — CLIP 编码指令，经 adapter 映射到 latent 空间
3. **因果 Transformer 主干（world predictor）** — 对交叉注意力融合后的视觉历史做因果建模，带语言条件
4. **双头** — 价值头输出 \(\hat V_t\)；世界头用 **gated FiLM** 的动作条件残差更新预测 \(\hat z_{t+1}\)

联合目标：

\[
\mathcal L=\mathcal L_{value}+\lambda\cdot\mathcal L_{pred}+\eta\cdot\mathcal L_{SIGReg}
\]

- \(\mathcal L_{pred}\)：teacher forcing 下预测隐状态与真值隐状态的 L2
- \(\mathcal L_{SIGReg}\)：把 latent 分布往各向同性高斯拉，**防止表征坍塌**（LeJEPA 的正则）
- \(\mathcal L_{value}\)：预测价值与折扣回报的 L2

**对 wiki 的映射：** 实体页画「历史帧 → 编码 → 因果主干 → 价值头 / 世界头」结构图；强调「世界建模目标是 critic 的辅助监督，不是用来做规划的 rollout 模型」。

## 摘录 3：如何接进 RL 管线

| 管线 | 主干 | 算法 |
|------|------|------|
| On-policy | π₀ / π₀.₅（flow matching） | **Flow-SDE**，把原 MLP critic 换成 WCM |
| On-policy | OpenVLA-OFT（自回归） | **PPO** + WCM critic |
| Off-policy | 自回归 | **AWR**（advantage-weighted regression） |
| Off-policy | flow matching | **RECAP** |

价值输出走 GAE 得到优势，再做策略更新；off-policy 侧把 SFT 数据与 rollout 数据放进**统一 buffer**以稳住价值估计。

## 摘录 4：实验（149 任务 / 4 基准）

- **ManiSkill**：25 个 pick-and-place，分 IND / OOD（视觉、语义、执行扰动三类偏移）
- **MetaWorld**：pick-and-place 以外的多样操作
- **CALVIN**：长程任务序列
- **LIBERO-Plus**：跨视角 / 环境 / 初始状态 / 语言 / 噪声 / 布局 / 光照的泛化

**ManiSkill（相对同主干 SFT 基线）：**

| 主干 | SFT | +WCM（IND） | ΔIND | +WCM（OOD） | ΔOOD |
|------|-----|------------|------|------------|------|
| π₀ | 38.4% | 84.4% | **+46.0** | 51.5% | **+33.4** |
| π₀.₅ | 47.0% | 91.9% | **+44.9** | 64.4% | **+38.0** |
| OpenVLA-OFT | 28.1% | 99.0% | **+70.9** | 77.9% | **+59.6** |

**LIBERO-Plus（one-shot SFT → RL）：** π₀ 39.1%→**72.8%**（+33.7）；π₀.₅ 38.0%→**73.7%**（+35.7）；OpenVLA-OFT 29.3%→**74.0%**（+44.7）。

**基线对照：** 仿真 on-policy 比 Flow-Noise / Flow-SDE / π-StepNFT（π 系列）与 GRPO / 标准 PPO（OpenVLA-OFT）；真机 off-policy 比「AWR + Gemma 270M critic」与「RECAP + Gemma 270M critic」。

**真机（WidowX-250S，第三人称 + 腕部相机，7 任务）：** 传送带寿司动态抓取、布料折叠、毛巾折叠、灶台清理（长程）、胡萝卜 / 辣椒 / 香蕉 pick-and-place；每任务约 100 条遥操作轨迹，8 轮 RL × 每轮 50 rollouts。前 50 条测试轨迹上 OpenVLA-OFT+WCM 全面超过 AWR 基线（如传送带 22/50 vs 17/50），π₀.₅+WCM 超过 RECAP（如布料折叠 38/50 vs 37/50）。

**对 wiki 的映射：** 实体页写「IND 提升大、OOD 提升更是卖点」的读法，并提醒真机差距在部分任务上只有 1–5 次成功的量级。

## 摘录 5：消融与失效分析

| 消融 | 结论 |
|------|------|
| 世界预测目标必要性 | MLP + frame-stacking 不行；ViT critic 但 \(\lambda=0\)（无世界预测）**仍然不行** → 有效的是**预测目标本身**，不是「换个更大的 critic」 |
| 历史长度 K | 平均最优 **K=3**；论文假设是刚好覆盖二阶动力学（加速度），再长收益递减 |
| 损失权重 \(\lambda\) | 最优区间 **[0.3, 0.5]**；\(\lambda=0.9\)（预测主导）比 \(\lambda=0.1\)（价值主导）**OOD 更好**；OOD 对 \(\lambda\) 的敏感度远高于 IND（10.6 vs 2.7 个百分点波动） |
| critic 过拟合 | Flow-SDE 有「dropping phenomenon」：OOD 峰值后掉；**把价值恒置零的消融 OOD 反而比 Flow-SDE 好**（说明坏 critic 比没 critic 更糟）；WCM 前 500 步无此退化 |

**其他：** OpenVLA-OFT 零暴露起步 0.78% → RL 后 98.7%；仿真 SFT 直接上真机全失败，RL 同时改善仿真与真机；WCM critic 的 rollouts/hour 优于 Gemma VLM critic 基线。

## 摘录 6：局限

论文未单列 limitations 节，隐含约束：

- 最优历史长度 **K=3 依任务而异**；
- LeJEPA 的计算开销未与基线定量对比；
- 真机只在 **WidowX-250S** 单平台验证（无人形 / 双臂）；
- \(\lambda\) 需按 IND / OOD 取舍调参；
- HF 上仅部分 checkpoint，完整复现仍需自备 VLA 主干与其训练栈。

## 建议 wiki 动作

- 新建 **`wiki/entities/paper-wcm-world-critic-model.md`**（含结构图 + 源码运行时序图 + 结论）。
- 新建 **`sources/repos/wcm-world-critic-model.md`**、**`sources/sites/sylvestf-wcm-homepage.md`**。
- 交叉：[`wiki/methods/model-based-rl.md`](../../wiki/methods/model-based-rl.md)、[`wiki/methods/vla.md`](../../wiki/methods/vla.md)、[`wiki/entities/openvla.md`](../../wiki/entities/openvla.md)、[`wiki/entities/paper-pi05-open-world-vla.md`](../../wiki/entities/paper-pi05-open-world-vla.md)、[`wiki/comparisons/online-vs-offline-rl.md`](../../wiki/comparisons/online-vs-offline-rl.md)。
