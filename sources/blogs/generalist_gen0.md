# GEN-0 / Embodied Foundation Models That Scale with Physical Interaction（Generalist AI）

> 来源归档（blog / Generalist AI 官方）

- **标题：** GEN-0 / Embodied Foundation Models That Scale with Physical Interaction
- **类型：** blog（官方栏目标注 Research）
- **作者 / 组织：** Generalist Team / Generalist AI
- **原始链接：** <https://generalistai.com/blog/gen-0>
- **发表日期：** 2025-11-04（页面标注 November 4, 2025；Citation 写 Nov 2025）
- **入库日期：** 2026-10-09
- **抓取方式：** curl 抓官方页 HTML，去标签后逐段核对（2026-10-09）
- **一句话说明：** Generalist 发布 **GEN-0** 具身基础模型系列：在 **27 万+ 小时** 真实世界操作数据（每周新增 1 万+ 小时）上直接预训练，报告 **7B 附近的"相变"**（1B 出现 ossification、7B+ 能吸收数据并以少量后训练迁移）、预训练数据量与下游后训练误差之间的 **幂律 scaling law**，提出 **Harmonic Reasoning**（异步、连续时间的感知 / 动作 token 流"边想边做"），并已扩展到 **10B+**。

## 开源 / 项目页核查（步骤 2.5）

| 项 | 结论（截至 2026-10-09） |
|----|-------------------------|
| 本篇博客 / 项目页 | **无**独立项目页或技术报告；官方入口为博客正文 |
| 论文 | **无** arXiv 论文；正文 arXiv 链接均为所引参考文献（Kaplan 2020、Hernandez 2021、Springer 2025、Black 2025 等） |
| 代码 / 权重 | **未见公开**：正文无 GitHub / Hugging Face 链接；Hugging Face 检索 `gen-0` 无 Generalist 相关模型。GitHub 组织页在本环境代理下返回 403，**未能直接核验** |
| 数据集 | **未公开**：in-house 数据集（27 万+ 小时），仅展示内部检索工具视频与 Figure 5 规模对比 |
| 可信度边界 | 产业官方博客，非 peer-reviewed；全部定量为 **公司自报**，曲线未给原始数值（Table 1 除外） |

## 核心摘录（归纳，非全文）

### 定位与六条主张（博文 Introduction）

1. **Surpassing the Intelligence Threshold：** 高数据量下在 **7B** 观察到相变——小模型出现 **ossification**（"骨化"，无法再吸收新信息），大模型持续提升；已扩展至 **10B+**，新任务适应所需后训练越来越少。
2. **Scaling Laws：** 更多预训练数据与算力 **一致且可预测地** 提升多任务下游后训练表现。
3. **Harmonic Reasoning：** 物理世界"physics doesn't stop"，不能像聊天模型那样先想后答；在 **异步、连续时间** 的感知 token 与动作 token 流之间形成"和声"式交互，从而不依赖 **System1-System2 双系统**（引 Figure Helix）或 **推理时引导**（引 Black et al. 2025 实时 action chunking）即可扩到很大模型。
4. **Cross-Embodiment：** 架构按设计跨本体；已在 **6DoF、7DoF、16+DoF 半人形** 机器人上测试。
5. **No Longer Limited By Data：** in-house 数据集 **270,000+ 小时** 真实多样操作数据，**每周增长 10,000 小时** 且在加速。
6. **The Science of Pretraining：** 不同来源（如 data foundry 合作方）的数据配比会得到特性不同的 GEN-0 模型。

### 演示任务

- **Build a camera kit：** 长程灵巧任务——放清洁布入盒、折纸托、取相机并从塑料袋中抽出、装盒、合盖（插入小折舌）、丢弃塑料袋；模型 **无显式子任务概念**，在单一 harmonic reasoning 流中完成。

### 模型规模相变（Figure 1）

| 规模 | 博文描述 |
|------|----------|
| 1B | 预训练中难以吸收复杂多样的 sensorimotor 数据，权重随时间无法吸收新信息（早期、明显 ossification） |
| 6B | 开始从预训练获益，展现较强多任务能力 |
| 7B+ | 能内化大规模机器人预训练数据，**仅几千步后训练** 即迁移到下游任务 |

- 指标：完全 held-out（零样本）长程下游任务上的 **next-action 验证预测误差**；x 轴为以 GEN-0 7B 归一化为 1.0 的预训练算力。
- 作者称据其所知这是 **机器人领域首次观察到 ossification**；LLM 文献中 ossification 发生在 O(10M) 参数量级，而此处在 O(1B)，作者将其与 **Moravec 悖论** 联系（物理常识可能有更高的算力"激活阈值"）。脚注说明：LLM 文献中该词指 pretrain→finetune 设置，本文是在 **纯预训练阶段的零样本泛化** 上观察到类似现象。

### 预训练 → 后训练 scaling（Figure 2–4）

- 用不同预训练数据子集的 checkpoint，在 **16 个任务集** 上做多任务语言条件 SFT；预训练越多，所有任务的验证损失与 next-action 误差越低。任务含灵巧（搭 Lego）、行业流程（快餐打包）、泛化（"_ anything" 类任务）。
- **真机盲测 A/B：** 仅用 **5.6 小时（1%）** 任务数据后训练时，更多预训练数据带来更高闭环成功率；最高成功率（**部分情况峰值可达 99%**）出现在"完整预训练 + 全部 **550+ 小时** 任务后训练数据"组合。预训练与后训练数据 **无重叠**（不同人员、完全不同环境采集）。
- **幂律形式：** 固定下游数据与微调预算，预训练集大小 \(D\) 与下游验证误差满足 \(L(D) = (D_c / D)^{\alpha_D}\)。可用于回答"达到某误差需要多少预训练数据""更多预训练数据能替代多少后训练数据"。
- 例：**Clothes Handling**（分拣、理顺、扣扣子、挂衣，真实工作场所）——可预测 **10 亿条动作轨迹** 下的模型表现（博文未给具体预测值）。
- 适用行业（作者称所有测过的任务均成立）：服装、制造、物流、汽车、电子。

### 数据与基础设施（Robotics is No Longer Limited By Data）

- **270,000 小时** 真实操作轨迹，采自全球 **数千个** 家庭、仓库与工作场所；**每周 10,000+ 小时** 新数据，由数千台采集设备与机器人组成的全球网络支撑。
- Figure 5：称训练数据比截至 2025-11 的若干最大机器人数据集 **多数个数量级**（图中对比对象未在正文列名）。
- 内部检索工具：语言标签嵌入的 t-SNE 地图，文本检索最近邻区域并随机抽样视频（演示仅覆盖 **<1%** 预训练数据，含"数百万种"活动）。
- 基础设施：自研硬件、dataloader 与网络（含铺设专用互联网线路）；多云合同、自研上传机；**O(10K) 核** 持续多模态处理；压缩 **数十 PB** 数据；训练时每天可吸收 **6.85 年** 的真实操作经验。

### 预训练科学（Table 1）

- 大规模消融结论：**数据质量与多样性比纯数量更重要**；不同数据配比得到不同预训练特性。
- 8 种预训练数据集（多家 data foundry 合作方 × 采集类别）→ 在 **10 个长程任务集**（分 dexterity / applications / generalization 三组）上微调后比较。
- 类别定义：**Class 1** 特定任务数据，**Class 3** "do-anything" 类数据，**Class 2** 介于其间。
- 指标：验证 MSE 与 **reverse KL**（以策略样本构造单位方差高斯混合密度做 Monte-Carlo 估计，衡量 mode-seeking）。
- 经验：**低预测误差 + 低 reverse KL** 的模型更适合 SFT 后训练；**高预测误差 + 低 reverse KL** 的模型分布更多峰，可能更利于后训练 RL。多种采集策略并行使其能持续 A/B 哪类数据对预训练最有益。
- Table 1 数值差异很小（预测误差约 0.0030–0.0034，reverse KL 约 0.0018–0.0026），博文未给误差棒。

## 对 wiki 的映射

- [generalist-gen0](../../wiki/entities/generalist-gen0.md) — 本篇升格实体页
- [generalist-ai-robotics](../../wiki/entities/generalist-ai-robotics.md) — 公司入口页 GEN 系列脉络
- [embodied-scaling-laws](../../wiki/concepts/embodied-scaling-laws.md) — 预训练数据 → 下游误差幂律、模型规模相变
- [data-flywheel](../../wiki/concepts/data-flywheel.md) — 每周 1 万小时的数据运营与 data foundry A/B
- [foundation-policy](../../wiki/concepts/foundation-policy.md) — 大规模预训练操作基座
- [hub-cross-embodiment](../../wiki/overview/hub-cross-embodiment.md) — 6/7/16+DoF 跨本体主张

## 可信度与使用边界

- **官方博客、全部自报**；无第三方基准、无可复现配方，曲线图未给原始数据点。
- "首次在机器人中观察到 ossification""数个数量级更多数据"等为 **作者立场**。
- Harmonic Reasoning 只给概念描述，**未披露架构、tokenization 与训练目标细节**。
- Table 1 指标差异很小且无方差，跨数据源排序结论需谨慎引用。

## Citation

```bibtex
@article{generalist2025gen0,
  author = {Generalist Team},
  title = {GEN-0: Embodied Foundation Models That Scale with Physical Interaction},
  journal = {Generalist AI Blog},
  year = {2025},
  note = {https://generalistai.com/blog/gen-0}
}
```
