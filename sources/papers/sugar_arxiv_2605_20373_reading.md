# SUGAR：A Scalable Human-Video-Driven Generalizable Humanoid Loco-Manipulation Learning Framework

> 来源归档（ingest · 人形 Loco-Manip 161 篇长文 第 076/161 · 原文通读补全）

- **标题：** SUGAR: A Scalable Human-Video-Driven Generalizable Humanoid Loco-Manipulation Learning Framework
- **Loco-Manip 161 分类：** 03 视觉感知驱动的人形移动操作
- **机构：** 北京大学 CFCS（School of Computer Science）、北京航空航天大学（School of Computer Science and Engineering）
- **类型：** paper
- **发表：** arXiv:2605.20373v1（2026-05-19）
- **原始链接：**
  - <https://arxiv.org/abs/2605.20373>
  - 项目页：<https://tianshuwu.github.io/sugar-humanoid/>
  - 官方代码：<https://github.com/tianshuwu/SUGAR>
- **入库日期：** 2026-06-26（原文通读补全：2026-11-14）
- **核对范围：** 通读 `arXiv:2605.20373v1` PDF 全文抽取文本（第 1–1481 行）：覆盖 Abstract、§1 引言、§2 相关工作、§3 三阶段方法（自动化视频先验提取、特权 RL Refiner 与 Progressive State Pool、Tracker BC+RL 蒸馏与 DiT Command Generator 闭环采集）、§4 六任务仿真对比（Table 1）、数据规模缩放（Table 2）、模块消融与真机 10 次试验评测（Table 3）、§5 结论与参考文献，以及 Appendix A–D（PPO/Tracker/Generator 超参、VLM 接触提示模板、奖励/域随机化/早停表 Table 4–9 与算力消耗）。未逐像素复核 Fig. 1/4–8 照片与折线图。
- **通读：** 已完成
- **一句话说明：** 针对第三人称人类交互视频中存在的遮挡、接触穿模与重定向误差，SUGAR 用「自动化运动与 VLM 接触先验抽取 → 特权 RL 物理清洗（Refiner + Progressive State Pool）→ Tracker BC+RL 蒸馏与 12 层 DiT Command Generator 分层闭环策略」将无标注人类视频转化为可零样本部署到 Unitree G1 的参考无关自主交互策略。

## 核心摘录

### 1) 原有策展索引定位
- **在 161 篇地图中的位置：** 03 视觉感知驱动的人形移动操作，编号 **076/161**；原始汇总归档见 [wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md](../blogs/wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md) 与 [wechat_humanoid_loco_manip_161_2026-06-26.md](../raw/wechat_humanoid_loco_manip_161_2026-06-26.md)。

### 2) 三阶段架构与方法细节（§3 & Appendix A–B）
- **Stage 1：自动化运动与接触先验抽取（§3.2）：** 用 `SAMBody` + 深度 ICP 提取人体运动 \(\hat{p}_{1:T}^R\)，`SAMObj` + 点云尺度对齐 + `FoundationPose` 估计物体 6D 轨迹 \(\hat{p}_{1:T}^O\)；用 VLM（Qwen3-VL）按任务部件逐帧判别直接物理接触标签 \(\hat{l}_t\)（强遮挡任务如 Kick Box 辅以物体速度阈值判别），时序滤波得噪声先验数据集 \(\mathcal{P}\)。
- **Stage 2：特权 RL Refiner 物理清洗（§3.3）：** 在统一模仿奖励 \(r = r_{track} + r_{int} + r_{reg}\)（Table 7）与多维域随机化/接触脉冲扰动下训练特权 RL 策略 \(\pi_r\)；提出 **Progressive State Pool (PSPI)** 从已物理验证的成功中间状态而非噪声 \(\mathcal{P}\) 初始化环境；收集物理可行专家示范集 \(\mathcal{R}\) 及高层命令 \(c_t = [q_t^{cmd}, v_t^{cmd}, \omega_t^{cmd}, l_t]\)。
- **Stage 3：Tracker + Diffusion Command Generator 闭环蒸馏（§3.4 & Appendix A.2）：**
  - **Command Tracker \(\pi_t\)：** 先以 BC 模仿 \(\pi_r\)（前 `500` 步纯 BC，随后 Critic 预热并于 `2000` 步后转全 RL），仅依赖 5 帧本体历史、当前物体相对位姿与命令 \(c_t\)（`50 Hz`）。
  - **Command Generator \(\pi_g\)：** 在冻结 \(\pi_t\) 闭环采集的实际轨迹 \(\mathcal{D} = \{(\tilde{o}_t^O, c_t, g)\}\) 上训练 12 层 Diffusion Transformer（DiT，预测 `H=8` 步并执行 `A_a=4` 步，经线性插值由 `10 Hz` 对齐至 `50 Hz`），在真机实现参考轨迹无关的自主交互与跌落/推挤后恢复。

### 3) 定量实验与真机结果（§4, Table 1–3, Appendix D）
- **仿真对比（每任务 100 条训练视频、30 条测试视频，Table 1）：**
  - 测试集上 SUGAR（不依赖参考轨迹观测，仅可选目标 `g`）在 `Carry Box`（`69.6%`）、`Sit Chair`（`99.6%`）、`Pick Bottle`（`99.2%`）、`Stand Bottle`（`86.3%`）与 `Push Box`（`70.0%`）等全面领先仅回放参考轨迹的 `Resmimic`（`0%–20.6%`）与 `HDMI`（`0%–54.6%`）。
- **数据规模缩放（Table 2）：** 每任务视频由 `20 → 50 → 100` 条时，Test 成功率稳定上升（`Kick Box: 32.7% → 63.1% → 76.0%`，`Push Box: 35.0% → 52.3% → 70.0%`，`Carry Box: 33.5% → 61.0% → 69.6%`）。
- **真机评测（Unitree G1 + MoCap，每任务 10 次试验，Table 3）：** `Kick Box 7/10`、`Push Box 6/10`、`Carry Box 7/10`、`Sit Chair 9/10`、`Pick Bottle 9/10`、`Stand Bottle 8/10`，且展现推挤扰动恢复、失败重拾及跨尺寸/外观物体零样本泛化。
- **算力消耗（Appendix D）：** 单卡 RTX 5090 上每任务训练 Refiner 约 `20h`、Tracker 约 `20h`、Command Generator 约 `5h`。

## 对 wiki 的映射

- [paper-loco-manip-161-076-sugar](../../wiki/entities/paper-loco-manip-161-076-sugar.md)
