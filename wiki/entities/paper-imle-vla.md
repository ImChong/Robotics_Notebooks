---
type: entity
tags: [paper, vla, imitation-learning, real-time, diffusion-policy, sfu, penn]
status: complete
updated: 2026-09-12
arxiv: "2609.10915"
code: https://kianhk6.github.io/IMLE-VLA/
related:
  - ../methods/vla.md
  - ../methods/diffusion-policy.md
  - ../methods/action-chunking.md
  - ../tasks/manipulation.md
  - ../overview/dexterous-wm-humanoid-14-papers-technology-map.md
  - ./paper-dlsrl.md
  - ./paper-show-harness.md
sources:
  - ../../sources/papers/imle-vla_arxiv_2609_10915.md
  - ../../sources/sites/imle-vla-github-io.md
  - ../../sources/blogs/wechat_embodied_station_14_papers_dexterous_wm_humanoid_2026-09-11.md
summary: "IROS 2026：cIMLE 单步动作头替换 π₀.₅ 的 10 步 flow matching；55 Hz / LIBERO 均值 98.0%；Franka 真机四任务全胜且 jerk 降 2.2–3.0×；代码仍待发布。"
---

# IMLE-VLA（arXiv:2609.10915）

**IMLE-VLA**（[IMLE-VLA: Fast Single-Step Action Generation for Vision-Language-Action Policies](https://arxiv.org/abs/2609.10915)，[项目页](https://kianhk6.github.io/IMLE-VLA/)，**IROS 2026**）由 SFU APEX Lab、UPenn 等提出：保留预训练 VLM 骨干，把 **迭代 flow-matching 动作头** 换成 **单步条件 IMLE（cIMLE）生成器**，在 **不牺牲多模态动作覆盖** 的前提下把控制频率拉回实时区间。

## 一句话定义

**VLA 的瓶颈在动作头的多步采样而非 VLM——用 cIMLE 单步生成替换 π₀.₅ 的 10 步 flow matching，L40S 上 55 Hz、LIBERO 均值 98.0%，真机 jerk 与 wall-clock 同步下降。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| IMLE | Implicit Maximum Likelihood Estimation | 隐式极大似然估计 |
| cIMLE | Conditional IMLE | 条件版，用于动作块生成 |
| VLA | Vision-Language-Action | 视觉-语言-动作策略 |
| LIBERO | — | 40 任务桌面操作仿真基准 |
| H | Execution horizon | 开环执行视界（动作块长度） |

## 为什么重要

- **算法瓶颈，非硬件瓶颈：** 项目页指出 π₀.₅ 即换 PyTorch+compile 或 Triton 内核，频率也只到约 20–25 Hz；**10 步 flow matching 循环**才是天花板。
- **多模态不丢：** cIMLE 相对朴素回归头 **避免 mode collapse**；LIBERO-Plus 分布偏移下仍保持 π₀.₅ 级鲁棒，而 OpenVLA-OFT 等 L1 回归头在 shift 下崩溃。
- **真机闭环收益：** Franka Panda 四任务 **全胜 π₀.₅**；更高频率使 **移动目标盘** 等反应性任务从失败变成功。

## 核心信息

| 项 | 内容 |
|----|------|
| **机构** | Simon Fraser University（SFU）、University of Pennsylvania（UPenn）、Amii |
| **骨干** | 应用于 **π₀.₅**（VLM + 动作头） |
| **会议** | IROS 2026 |
| **arXiv** | [2609.10915](https://arxiv.org/abs/2609.10915)（截至 2026-09-12 仍为 **v1**） |
| **项目页** | <https://kianhk6.github.io/IMLE-VLA/> |
| **开源** | **待发布** — 2026-09-12 再核：项目页仍无独立 GitHub |

## 核心原理

### 替换点

| 组件 | π₀.₅ 基线 | IMLE-VLA |
|------|-----------|----------|
| 动作头 | 10 步 flow matching（Euler） | **单步 cIMLE** |
| 推理频率（L40S） | 15 Hz | **55 Hz（3.67×）** |
| 多模态 | 扩散/flow 采样 | cIMLE **保覆盖、免多步** |

### 流程总览

```mermaid
flowchart LR
  obs["多视角图像\n+ 语言指令"]
  vlm["预训练 VLM 骨干\n（冻结或微调）"]
  head["cIMLE 单步动作头"]
  chunk["动作块 H 步"]
  robot["Franka / 仿真"]
  obs --> vlm --> head --> chunk --> robot
```

### 加速的两条路径（读法）

| 路径 | 机制 | 代价 |
|------|------|------|
| **IMLE 单步** | 去掉迭代采样环 | 需重训动作头 |
| **加长 H** | 吞吐 = 频率 × H | H=30 时吞吐 11×，但 **开环更长、反应性下降** |

## 源码运行时序图

**不适用**（截至 **2026-09-12** 无官方可 clone 仓库；发布后应补 `sources/repos/` 并更新本图。）

## 实验与评测

### LIBERO 仿真（50 episodes/task，H=10）

| 方法 | Spatial | Object | Goal | Long | **Avg** | Inf. (×) |
|------|---------|--------|------|------|---------|----------|
| π₀.₅ | 97.2 | 99.0 | 97.8 | 96.0 | 97.5 | 1.0 |
| **IMLE-VLA** | **98.0** | **99.8** | **98.2** | **96.0** | **98.0** | **3.67** |

- **VLA-WC（仿真）：** 0.10 s vs π₀.₅ 1.03 s（**10.3×** 更快）。

### Franka 真机（DROID 训练，四任务各 20 episodes，H=12 vs H=8）

| 任务 | IMLE-VLA | π₀.₅ | Jerk (×) |
|------|----------|------|----------|
| Pineapple in bowl | 19/20 | 15/20 | 2.7× |
| Swap pineapple & cube | 18/20 | 15/20 | 2.2× |
| Pineapple in cabinet | 15/20 | 12/20 | 3.0× |
| Pineapple on moving plate | **16/20** | 12/20 | 2.7× |

- **VLA wall-clock / episode：** 真机 **3.9×–6.6×** 低于 π₀.₅。
- **LIBERO-Plus：** 四轴（背景/初始态/语言/布局）× 五级严重度；IMLE-VLA 保持 π₀.₅ 鲁棒性。

## 与其他工作对比

| 对照 | 差异 |
|------|------|
| [Diffusion Policy](../methods/diffusion-policy.md) / π₀ 族 | 多步采样限频；IMLE-VLA 换 **采样器** |
| [Action Chunking](../methods/action-chunking.md) | 靠加长 H 摊开销；IMLE 从 **根上减每步推理** |
| Shallow-π₀.₅（层蒸馏） | 仍保留迭代头；IMLE 在 **同等 Inf. 优势下 Avg 更高**（98.0 vs 97.0） |
| [DLSRL](./paper-dlsrl.md) | 动 latent 做 RL 适配；IMLE 动 **生成机制** |
| [Show-Harness](./paper-show-harness.md) | 离散微动作接口绕推理；IMLE 仍 **端到端 VLA** |

## 结论

**IMLE-VLA 证明 VLA 实时化的主杠杆是动作头采样步数；cIMLE 在成功率、频率与平滑度上可同时赢 π₀.₅。**

1. **首选场景：** 需要 **高频率闭环**（动态目标、少停顿）的 tabletop VLA 部署。
2. **勿只靠加长 H：** 11× 吞吐诱人，但反应性任务仍依赖 **单步高频**（移动盘案例）。
3. **鲁棒性：** LIBERO-Plus 上优于 L1 回归类 VLA；引用时区分 **仿真 Avg** 与 **真机 20 局** 口径。
4. **开源（2026-09-12 再核）：** 仍 **待发布**；arXiv 无 v2。
5. **工程落点：** 在现有 π₀.₅ 栈上 **只换动作头训练** 的路径清晰，等官方代码后易对标。

## 关联页面

- [VLA（Vision-Language-Action）](../methods/vla.md)
- [Diffusion Policy](../methods/diffusion-policy.md)
- [14 篇技术地图](../overview/dexterous-wm-humanoid-14-papers-technology-map.md)
- [DLSRL](./paper-dlsrl.md)

## 参考来源

- [imle-vla_arxiv_2609_10915.md](../../sources/papers/imle-vla_arxiv_2609_10915.md)
- [imle-vla 项目页归档](../../sources/sites/imle-vla-github-io.md)
- [wechat 14篇盘点](../../sources/blogs/wechat_embodied_station_14_papers_dexterous_wm_humanoid_2026-09-11.md)

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2609.10915)
- [项目页](https://kianhk6.github.io/IMLE-VLA/)
