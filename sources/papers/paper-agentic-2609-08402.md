# Towards Embodied Air-Ground Cooperative Object Search: Benchmark, Dataset and Agentic Method

- **作者：** Boao Yu, Zimo Chen, Junreng Rao, Yue Hu, Zhengqiu Zhu, Yong Zhao, Jianzhe Gao, Pengfei Zhu, Rusheng Ju
- **机构：** National University of Defense Technology（College of Systems Engineering；National Key Laboratory of Digital Intelligent Modeling and Simulation）、Zhejiang University、Tianjin University
- **版本 / 载体：** arXiv:2609.08402v3 [cs.CV]（2026-09-08 提交 v1，2026-11-20 修订 v3；22 页、6 图、20 表，含正文 §1–§7 与 Appendix A–F）
- **原文：** [arXiv HTML](https://arxiv.org/html/2609.08402) / [arXiv Abs](https://arxiv.org/abs/2609.08402)
- **项目页 / 代码：** 已开源（官方仓库 [`Yomier/AGOS-Bench`](https://github.com/Yomier/AGOS-Bench) 兼作项目页，代码采用 Apache-2.0 协议；配套数据集 [`Yomier/AGOS-Dataset`](https://huggingface.co/datasets/Yomier/AGOS-Dataset) 已上线 Hugging Face 与 ModelScope，采用 CC BY 4.0 协议；论文原始 LoRA 实验权重未随仓直接发布，仓库提供完整 SFT 复现训练与并行评测脚本）
- **核对范围：** arXiv:2609.08402v3 全文（Abstract、§1–§7、Appendix A–F、Table 1–20、Figure 1–6 文字与数值）及官方仓库 [`Yomier/AGOS-Bench`](https://github.com/Yomier/AGOS-Bench) 的 README 与源码目录清单
- **通读：** 已完成
- **入库日期：** 2026-11-27

## 原文核对笔记

- **核心任务与闭环设定（§1, §3, Appendix C）：** 定义了城市环境下的具身空地协同目标搜索任务（AGOS）：由 1 架无人机（UAV）与 1 辆无人地面车（UGV）组成的团队仅凭目标车辆的**单张无背景参考图像** $I_{\text{ref}}$（约 12 m 距离、仰角 $10^\circ\text{–}75^\circ$、非部署点拍摄）搜索指定城市场景，在**无路径指令、无目标坐标/方向先验**的条件下，由 UAV 广域发现候选、通过文本消息交接给 UGV 抵近并完成地面实例级确认。
  - **观测与动作空间：** UAV 固定于 80 m 高度、$90^\circ$ 视场角俯视采集 $1024\times 576$ RGB 图像，在 40 m 间距离散网格上做 8 方向水平移动、标注候选、请求协助或悬停观察（规划采样半径 25 m）；UGV 采集 6 视角环形 RGB（单路 $480\times 270$，按 $-45^\circ, 0^\circ, 45^\circ, -135^\circ, 180^\circ, 135^\circ$ 拼成 $1440\times 540$ 的 $3\times 2$ 拼接图），在约 10 m 间距的公开有向路网上前进、倒车、路口转弯、确认、拒绝或停车观察。两智能体依据动作耗时异步推进，单回合预算为 80 个任务步与 1,000 次逻辑模型调用。
  - **成功判据（Eq. 2）：** 仅当 UGV 满足水平距离 $d(\text{UGV}, o) \le 30\text{ m}$、无遮挡视线（LOS）、目标出现在六视角图像中（$\mathrm{Vis}(o, V_{\text{stop}})$），且地面确认裁决为 `MATCH` 时才算成功。9 项指标分为导航类（`SR`、`OSR`、`SPL`、`NE`）、协同类（`ASE` 空中搜索效率、`ACG` 空中候选引导质量、`GVA` 地面验证准确率）与效率类（`DS` 平均 MLLM 调用次数、`DSR` 调用步数加权成功率）。
- **AGOS-Dataset 构建与统计（§4, Appendix D）：** 基于 CARLA-Air v0.1.7 覆盖 6 张城市地图（Town01–05、Town10HD），共 **2,125 回合**，划分为 Train1100（1,100）、Val275（275）、Seen-Test450（450，共享 5 张已见地图）与 Unseen-Test300（300，完全留出 Town03）。目标车辆来自 34 种车型、159 种外观。任务分 Easy（$150\times 150\text{ m}$、无干扰）、Mid（$300\times 300\text{ m}$、起始距离 $\ge 100\text{ m}$）与 Hard（再增 2–3 辆同型异色或同色异型近似干扰车；Hard Unseen-Test300 中同型异色占 73.6%）。1,100 条专家轨迹生成 **27,720 条 Train Examples**（Move Decision 15,214 / Verification 9,048 / Confirmation 3,458）；论文微调实验从 301 条轨迹的 8,212 条样本经 5,120-token 长度过滤后保留 **6,014 条**（3,822 / 1,648 / 544；UAV 5,297 / UGV 717）。
- **AGOS-Agent 双层架构（§5, Appendix E–F）：**
  - **算法与工具层：** 采用联合覆盖规划（Eq. 3）先将仅空中可达单元分给 UAV，再按预计耗时 $\max\{\hat{T}_{\text{UAV}}, \hat{T}_{\text{UGV}}\}$ 贪心分配其余路网单元，并在接受交接或释放候选后重规划；感知工具冻结 `Grounding DINO`（`grounding-dino-base`，框/文本阈值 0.20/0.18，NMS IoU 0.50）与 `SigLIP 2`（`siglip2-base-patch16-224`），关闭 SAM；事件驱动唤醒门控（Eq. 17）在未处理请求、候选到达、补充证据、工具故障或新检测相似度 $s_t(b) \ge \tau_{\text{wake}}=0.55$ 且不在去重表 $\mathcal{M}_{\text{dedup}}$ 时才调用 MLLM；候选框经针孔/高程反投影与路网吸附（Eq. 16）生成世界坐标与目标路点。
  - **可学习 MLLM 层：** UAV 与 UGV 共享同一个 MLLM，通过角色提示词与无 CoT 的单 JSON 接口完成存在判断、实例裁剪卡片核对（`MATCH`/`NON_MATCH`/`UNKNOWN`）、请求接收及地面 6 视角最终确认。
- **核心实验发现（§6, Appendix B）：**
  - **跨基座主结果（Table 2, Unseen-Test300, $n=300$）：** 在 9 个开源端侧与闭源 API 基座中，AGOS-Agent 在 7 个上提升 SR、在全部 9 个上降低 MLLM 调用数 DS。以 `Qwen3.5-4B` 为例，零样本 AGOS-Agent 达 **37.67% SR**（DS 36.3），较同信息同动作同提示词的 `Baseline-Agent`（30.67% SR，DS 55.5）高 **+7.00 pp**（Holm(3) $p=0.0259$，而跨 9 基座族的 Holm(9) $p=0.07777$），远高于直接多模态控制的 `Minimal-Agent`（15.33% SR，DS 70.0，存在 81 次巡逻死循环）；同基座下适配的 `AGC-VLN-OS` 与 `AeroDuo-OS` 分别为 31.33% 与 28.67%；API 基座 `GPT-6 Astra` 零样本达 52.33% SR。
  - **LoRA 微调与角色共享（Table 4 & Table 20）：** 冻结视觉编码器与投影层，仅在 `Qwen3.5-4B` 16 个 q/v 模块（约 0.92M 可训练参数）做 rank-8, $\alpha=16$ LoRA 微调，在 Unseen-Test300 上联合训练将 SR 提至 **57.00%**（+19.33 pp，OSR 74.33%，ACG 81.56%，DS 降至 24.2），在 Seen-Test450 上从 28.22% 提至 **52.00%**（+23.78 pp）；单适配器联合训练与双适配器分角色训练差距 $\le 2.00\text{ pp}$。不过由于 6,014 条训练样本不含 `NON_MATCH`/`REJECT` 监督，微调后 GVA 从 92.59% 降至 78.38%（Unseen-Test300 正确拒识事件由 412/567 降至 3/222）。
  - **决策源对照与消融（Table 3, 5, 9–10）：** 规则阈值确认（C2/C4/C6）导致 93.4%–100% 失败源于误确认；纯规则空中提名（C3）导致 89.1% 失败源于步数耗尽；UGV 单机混合策略（C5）整体 SR 32.33%（DS 64.4），但在 Hard 档位上以 33.0% 反超双机零样本 C1 的 20.0%，说明当空中提名容易把地面车引入干扰物区域时，协同收益取决于空中引导精度。2×2 消融证明仅同时启用联合覆盖（JC）与事件驱动唤醒（EW）才取得最高增益（+7.00 pp）。
- **对 wiki 的映射：** [Towards Embodied Air-Ground Cooperative Object Search: Benchmark, Dataset and Agentic Method](../../wiki/entities/paper-agentic-2609-08402.md)
