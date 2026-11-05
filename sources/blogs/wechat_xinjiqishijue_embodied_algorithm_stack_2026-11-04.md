# 2026年了，主流具身智能算法有哪些？一文彻底讲透VLA、扩散策略和世界模型

> 来源归档（blog / 微信公众号转载）

- **标题：** 2026年了，主流具身智能算法有哪些？一文彻底讲透VLA、扩散策略和世界模型
- **类型：** blog
- **来源账号：** 新机器视觉（转自公众号「人工智能AI与算法」；文首注明「本文仅用于学术分享，如有侵权请联系后台作删文处理」）
- **原始链接：** https://mp.weixin.qq.com/s/aIYDRlrluZIsmj3fmMsXIg
- **发表日期：** 2026-11-04
- **入库日期：** 2026-11-04
- **抓取方式：** `scripts/fetch_wechat_article.py`（Chromium + 微信内置 UA），正文 3527 字符
- **一句话说明：** 综述 2026 年具身智能的四层算法栈（VLA 基座、扩散策略保可靠性、RL 啃硬技能、世界模型赌未来泛化），并给出五类场景的算法组合建议。
- **说明：** 转载综述，**没有给出处**。我对文中点名的每个算法回到一手资料核对（arXiv 摘要页、OpenVLA / Octo 仓库 README、论文检索）：15 个复用已有节点，3 个新建，2 个（CSO、Latent PPO）无法定位。市场数据全部是文章表述、未核实。

## 算法 → 节点与一手核对

| 文中名称 | 层 | 节点 | 状态 | 一手核对 |
|----------|----|------|------|----------|
| OpenVLA | VLA | [paper-openvla](../../wiki/entities/paper-openvla.md) | 复用 | README：代码 MIT 许可，但预训练模型源自 Llama 2，受其许可约束（文章只写「MIT许可」，不完整）；LoRA 微调示例用**单张 A100**（文章写「单张 H100」）；7B 与 970K OXE 轨迹未在已读 README 段落逐项核对。 |
| Octo | VLA | [paper-octo](../../wiki/entities/paper-octo.md) | 复用 | README 给出 Octo-Base 93M / Octo-Small 27M 参数（与文章一致），1×RTX 4090 上推理 13 / 17 it/sec；**文章写「20–30 Hz」，与 README 的 13 / 17 it/sec 不一致**（二者口径是否相同未核实）；MIT 许可。 |
| SmolVLA | VLA | [paper-smolvla](../../wiki/entities/paper-smolvla.md) | 复用 | 库内节点已核对 arXiv:2506.01844 与 LeRobot 仓库。 |
| π0 | VLA | [paper-pi0](../../wiki/entities/paper-pi0.md) | 复用 | 库内节点：PaliGemma + Flow Matching 动作专家；文章「50Hz」未在本次核对。 |
| π0.5 | VLA | [paper-pi05-open-world-vla](../../wiki/entities/paper-pi05-open-world-vla.md) | 复用 | 库内节点（arXiv:2504.16054，开放世界泛化）；文章称「分层任务泛化」。 |
| π0.6（π*0.6） | VLA | [paper-rcl-2511-14759-0-6-a-vla-that-learns-from-experience](../../wiki/entities/paper-rcl-2511-14759-0-6-a-vla-that-learns-from-experience.md) | 复用 | 库内节点（arXiv:2511.14759）。arXiv 摘要：用 RECAP 经优势条件化策略做 VLA 的 RL，对应文章「π0.6 融合强化学习」。 |
| π0.7 | VLA | [pi07-policy](../../wiki/methods/pi07-policy.md) | 复用 | 库内方法页；文章称「集成世界模型与具身记忆」，库内页面写 π₀.₇ 使用 MEM 式历史视觉，二者表述不同，未判断。 |
| HiFOR | VLA | [paper-hifor-hierarchical-foresight-vla](../../wiki/entities/paper-hifor-hierarchical-foresight-vla.md) | 复用 | 库内节点（arXiv:2610.40373），与文章「分层预见框架」一致。 |
| S²-VLA | VLA | [paper-s2-vla-state-space-guided](../../wiki/entities/paper-s2-vla-state-space-guided.md) | 新建 | 新建（arXiv:2606.27872，IJCAI 2026）。**库内已有的 paper-s-squared-vla 是另一篇（驾驶 VLA，arXiv:2607.13926），不是文章所指的操作 VLA**，所以单独建节点。 |
| FlowVLA | VLA | [paper-flowvla-visual-cot](../../wiki/entities/paper-flowvla-visual-cot.md) | 新建 | 新建（arXiv:2508.18269，Visual CoT）。库内已有的两个 FlowVLA 节点（pluggable-3d-flow、quantitative-robotic-pouring）是别的论文。 |
| Diffusion Policy | 扩散策略 / 模仿学习 | [paper-diffusion-policy](../../wiki/entities/paper-diffusion-policy.md) | 复用 | 库内节点；文章对 DDPM 去噪、多模态的描述未在本次逐项核对。 |
| ACT | 扩散策略 / 模仿学习 | [paper-act](../../wiki/entities/paper-act.md) | 复用 | 库内节点；「CVAE Transformer、约 50 次演示收敛」为文章表述，未在本次核对。 |
| Two-Steps Diffusion Policy | 扩散策略 / 模仿学习 | [paper-two-steps-diffusion-policy](../../wiki/entities/paper-two-steps-diffusion-policy.md) | 新建 | 新建（arXiv:2510.21991，NeurIPS 2025）。摘要同时给出「少至 5 次 NFE」与「2 次 NFE 解决困难任务」，文章只取了 5 次；论文用 genetic denoising，在 D4RL / Robomimic 14 项任务上评测。 |
| REFINE-DP | 扩散策略 / 模仿学习 | [paper-loco-manip-161-157-refine-dp](../../wiki/entities/paper-loco-manip-161-157-refine-dp.md) | 复用 | 库内节点（arXiv:2603.13707）：分层 DP 规划器 + RL 跟踪器，DPPO/PPO 联合微调；文章「仿真中超过 90%」未在本次核对。 |
| PPO | 强化学习 | [ppo](../../wiki/methods/ppo.md) | 复用 | 库内方法页。 |
| SuperMimic | 强化学习 | [paper-supermimic](../../wiki/entities/paper-supermimic.md) | 复用 | 库内节点（arXiv:2610.28480）已写明：1 张 H100 × 3 小时对比 SONIC 的 128 张 L40 × 7 天，作者保守表述 GPU-hours 减少超过 1000 倍，与文章「千倍、3 小时」一致。 |
| ELEVATE | 强化学习 | [paper-elevate](../../wiki/entities/paper-elevate.md) | 复用 | 库内节点（arXiv:2610.38388）；文章「98.4%–100%」未在库内节点中找到，未核实。 |
| WMPO | 世界模型 | [paper-sa-2511-09515-wmpo-world-model-based-policy-optimization-for-v](../../wiki/entities/paper-sa-2511-09515-wmpo-world-model-based-policy-optimization-for-v.md) | 复用 | 库内节点（arXiv:2511.09515）；ICLR 2026（WebSearch）+ arXiv 摘要页核对：像素级动作条件视频世界模型 + on-policy GRPO，无需真实交互。 |

## 未能定位一手论文的两项

| 名称 | 文章说法 | 核对结果 |
|------|----------|----------|
| CSO | 文章称「通过对齐技能分布精炼策略，LIBERO 成功率从 80.1% 提升至 97.5%」 | 文章未给全称与出处；我用关键词搜索没有找到与之对应的机器人论文（命中的 CSO 是 LLM 智能体方法，Skill1 的 97.5% 是 ALFWorld，均不是），**未能定位一手论文，不建节点** |
| Latent PPO | 文章称「面向轮腿机器人的 Latent PPO」 | 文章未给出处；搜索没有找到名为 Latent PPO 的论文（最接近的 ATRos 为 PPO + 潜在嵌入辅助，是否即所指无法确认），**不建节点** |

## 文章的市场数据（作者表述，未核实）

- 智元「精灵 G2」在龙旗科技工厂 6 天测试累计 64,828 次作业、综合成功率 99.99%
- 2026 年上半年智元出货约 9,700 台、占全球人形机器人市场超过 43%
- 临界点灵巧手在真实工位完成 72 道工序、成功率 98%、年量产突破 15,000 台、整机故障率低于 0.5%
- 2026 年上半年国内具身智能融资额 934 亿元、为 2025 年同期的五倍；中国具身智能市场规模预计 2026 年达 1.09 万亿元
- 全球人形机器人出货结构中娱乐表演、科研仍占 60% 以上，智能制造和仓储物流分别仅 13% 和 5%

> 这些数据没有给出处；文章自己也提醒「演示呈现最优表现，部署要求最不利情况下仍满足产线标准」。

## 对 wiki 的映射

- 总览：[2026 具身智能算法栈：VLA、扩散策略、强化学习与世界模型（一手核对版）](../../wiki/overview/embodied-algorithm-stack-2026-vla-diffusion-rl-world-model.md)
