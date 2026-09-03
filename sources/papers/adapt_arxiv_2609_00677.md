# ADAPT: Agile Diffusion Action Priors for Robust and Steerable Online Text-Driven Humanoid Control（arXiv:2609.00677）

> 来源归档（ingest）

- **标题：** ADAPT: Agile Diffusion Action Priors for Robust and Steerable Online Text-Driven Humanoid Control
- **短名：** ADAPT（Agile Diffusion Action Priors）
- **类型：** paper / humanoid / text-driven-control / diffusion-policy / unitree-g1
- **arXiv：** <https://arxiv.org/abs/2609.00677>
- **HTML：** <https://arxiv.org/html/2609.00677>
- **PDF：** <https://arxiv.org/pdf/2609.00677>
- **项目页：** <https://wuyan01.github.io/ADAPT-project/> — 归档见 [`sources/sites/adapt-project.md`](../sites/adapt-project.md)
- **作者：** Yan Wu、Chenhao Li、Kaifeng Zhao\*、Gen Li\*、Marco Hutter、Siyu Tang（\* equal contribution）
- **机构：** 苏黎世联邦理工（ETH Zürich）
- **入库日期：** 2026-09-03
- **一句话说明：** 端到端文本条件扩散技能先验 + 下肢残差 RL + 噪声转向下游适应；G1 上 50 Hz 在线换 prompt。**不是** 网球线 AdaPT（arXiv:2608.20087）。

## 开源状态（步骤 2.5，2026-09-03）

| 资源 | 状态 | 说明 |
|------|------|------|
| 项目页 | **已发布** | 摘要、方法四段、技能/换 prompt/风格到达视频；页脚 BibTeX 仍是 UniPhys / NaP-Control，**无 ADAPT 本篇 citation 块** |
| GitHub | **未列** | 项目页与 arXiv 均无仓库 URL |
| 权重 / 数据 | **未发布** | 无 HF / 训练轨迹下载 |
| 论文承诺 | 未写 "code will be released" | 以项目页实际链接为准 |

**结论：确认未开源（截至 2026-09-03）。** 勿与 [noitom-robotics/AdaPT](https://github.com/noitom-robotics/AdaPT)（人形网球 AdaPT，arXiv:2608.20087）混淆。

## 核心摘录（面向 wiki 编译）

### 摘录 1：问题与主张（§1）

- 交互式语言控制是**闭环动力学问题**，不是「先生成运动学再跟踪」。
- 两阶段（SONIC / Kimodo / TextOP）在快速换令时参考常变得动力学不可行。
- 端到端（LangWBC / SENTINEL）多用 **clip 级 caption**，训练见不到片段内切换边界。
- ADAPT：BABEL **帧级**标注 + AMASS→GMR 重定向 + 跟踪 rollout 得到 \((o_t,a_t,\ell_t)\)；扩散策略直接出关节动作；冻结先验上叠 **下肢残差 RL**；同一先验用 **噪声转向** 做目标到达并保留文本风格。

**对 wiki 的映射：** [paper-adapt-text-driven-humanoid](../../wiki/entities/paper-adapt-text-driven-humanoid.md)；对照 [LangWBC](../../wiki/entities/paper-bfm-37-langwbc.md)、[TextOp](../../wiki/entities/paper-loco-manip-161-022-textop.md)、[UniPhys](../../wiki/entities/paper-bfm-40-uniphys.md)

### 摘录 2：扩散先验与残差 / 转向（§3）

- clip \(T=20\)、历史 \(H=5\)；8 层因果 Transformer（512-d / 8 头）；冻结 CLIP；独立逐帧噪声（Diffusion Forcing / UniPhys）；历史里 **根线速度置零**。
- 推理：2 步 DDIM + CFG 2.5；执行首动作；约 **2 ms**，支撑 **50 Hz**。
- 残差：\(a_t=a_t^{\mathrm{diff}}+\alpha\Delta a_t^{\mathrm{res}}\)；只改下肢；\(\alpha\) 从 0 热到 0.05；self-tracking 奖励贴近扩散预测下一状态；训练每 5–10 s 随机换 prompt。
- 转向：PPO 输出扩散初始噪声 \(z_t\)，冻结先验按文本风格到达目标后 stand。

**对 wiki 的映射：** [paper-adapt-text-driven-humanoid](../../wiki/entities/paper-adapt-text-driven-humanoid.md)、[Unitree G1](../../wiki/entities/unitree-g1.md)

### 摘录 3：评测数字（§4 / Table 1–3 / Table 2 / Table S11）

交互控制（2048×20 s，130 条命令，每 5–10 s 换 prompt；语义指标只在未摔倒回合上算）：

| 方法 | Success | R@1 | 备注 |
|------|---------|-----|------|
| DART+重定向+跟踪 | 0.764 | 39.05% | 两阶段，离线 lookahead |
| Offline TextOp | 0.522 | 45.52% | 两阶段上界 |
| LangWBC | 0.923 | 40.89% | 端到端 CVAE |
| Ours w/o residual | 0.804 | **59.50%** | 纯 BC 语义更好 |
| **Ours** | **0.984** | 44.60% | 残差换成功率、略降对齐 |

真机（5 trial）：Walk 5/5、Jog 5/5、Jump 3/5、Kick 4/5。失败集中在长单腿支撑与反复高跳。

目标到达（200 rollout / 风格）：转向成功率 **97.1%**、摔倒 **2.9%**；随机噪声 **18.0% / 34.7%**。未见过的 “jog”：**95.5% / 4.5%**。

残差消融：去掉空间约束 Success 0.997 但 R@1 掉到 26.19%（语义崩塌）。2 步 DDIM 是延迟–质量折中（1 步 Success 0.706；5 步 0.792 / 4 ms）。

**对 wiki 的映射：** [paper-adapt-text-driven-humanoid](../../wiki/entities/paper-adapt-text-driven-humanoid.md)

## 当前提炼状态

- [x] 项目页已打开：无 GitHub / HF
- [x] wiki 映射：`wiki/entities/paper-adapt-text-driven-humanoid.md` 新建
