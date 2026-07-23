# Agentic Real2Sim: Physics-based World Modeling with Vision-Language Agents（arXiv:2607.19190）

> 来源归档（ingest）

- **标题：** Agentic Real2Sim: Physics-based World Modeling with Vision-Language Agents
- **类型：** paper / agentic real2sim / physical digital twin / VLM agents / MuJoCo
- **arXiv：** <https://arxiv.org/abs/2607.19190>（PDF：<https://arxiv.org/pdf/2607.19190.pdf>）
- **项目页：** <https://agentic-real2sim.github.io/>
- **代码：** 项目页标注 **Code (coming soon)**（截至 2026-07-23 无公开 GitHub 训练/转换仓）
- **作者：** Guanxiong Chen、Qianjun Xia、Jiawei Peng、Heng Zhang、Bole Ma、Justin Qian、Ziyi Jiao、Bingyang Zhou、Luoxin Ye、Kaifeng Zhang、Kunyi Wang、Weijia Zeng、Yunuo Chen、Pengzhi Yang、Ziqiu Zeng、Huamin Wang、Chao Liu、Alan Yuille、Fan Shi、Changxi Zheng、Yunzhu Li、Chenfanfu Jiang、Peter Yichen Chen 等
- **机构：** 英属哥伦比亚大学（UBC）、约翰霍普金斯大学（Johns Hopkins）、埃尔朗根-纽伦堡大学 / NHR@FAU（FAU）、哥伦比亚大学（Columbia）、加州大学洛杉矶分校（UCLA）、新加坡国立大学（NUS）、凌迪科技（Style3D）
- **入库日期：** 2026-07-23
- **一句话说明：** 用 **VLM agent 编排**把真机交互录像（主攻 DROID）转成 **可回放的 MuJoCo episode twin**：视觉处理 → 物理先验 → 场景装配 → simulator-in-the-loop 抓取优化；同一 episode 合约可扩展到 **PhysTwin 可变形** 与 **BFM-Zero 人形运动**；开源 31B VLM 后端在 DROID-100 上与前沿闭源模型相近的回放成功率，成本约 GPT-5.4 的 **3%**。

## 开源状态（项目页核查，2026-07-23）

- **宣称将开源 / 待发布：** 项目页头部 **Paper / arXiv** 已挂，**Code (coming soon)**；页面提供浏览器内 OpenUSD / three.js 孪生预览与 BibTeX，**未列 GitHub / HF 训练仓**。复现入口以项目页后续更新为准。

## 摘要级要点

- **瓶颈：** Real2Sim 不止视觉重建——还需几何、物性、相机/机器人位姿、轨迹与可运行仿真装配；现状依赖手工调视觉基础模型、修 mesh、对齐坐标与脆弱工作流粘合。
- **贡献：** （1）DROID → MuJoCo episodic twin 的 agentic 管线；（2）可替换 VLM 后端（Gemma 4 31B / Qwen 3.6 35B / GPT-5.4 / Claude Haiku 4.5）；（3）同一合约适配可变形与人形。
- **指标：** DROID-100 上 Gemma 4 31B：**48/100** 回放成功、模型账单 **$2.62**；GPT-5.4 约 **31.4×** 成本仅换到相近成功率。绝对成功率仍 <50%，瓶颈更偏上游感知/仿真组件。
- **下游意图：** 用对齐孪生做策略学习与评测（文中明确 aim，实验主线仍是转换成功率）。

## 核心论文摘录（MVP）

### 1) Episode twin 合约

- **链接：** §3；Eq. (1)
- **摘录要点：** \(\mathcal{T}=(\mathcal{O},\mathcal{A},\mathcal{G},\mathcal{S}_{1:T},\Theta,\mathcal{B},\mathcal{M})\)——观测、执行器、几何、时序状态、物性参数、仿真后端与评测痕迹。目标是跨域共享阶段、回放环与 critic，仅技能/工具调用按域分化。
- **对 wiki 的映射：**
  - [Agentic Real2Sim](../../wiki/entities/paper-agentic-real2sim.md) — 核心产物定义。
  - [Sim2Real](../../wiki/concepts/sim2real.md) — Real2Sim 资产侧。

### 2) 四代理流水线（刚性 DROID）

- **链接：** §3.1–3.2；Fig. 1；Tab. 1
- **摘录要点：** **视觉处理**（SAM 3 分割 + SAM 3D mesh + FoundationStereo 深度 + FoundationPose 跟踪 + 掩码/跟踪 critic）；**物理先验**（材料/质量等）；**场景准备**（标定、基座位姿、地面参考 → MuJoCo）；**抓取优化**（物体位移 sweep 或 LLM 辅助精修）。VLM 只做有界 schema 决策，几何/物理由确定性工具执行。
- **对 wiki 的映射：**
  - [Agentic Real2Sim](../../wiki/entities/paper-agentic-real2sim.md) — 工程流水线。
  - [Articraft](../../wiki/entities/articraft.md) — 同属 agentic VLM 资产生成谱系，对象粒度不同。

### 3) 可变形 / 人形适配器

- **链接：** §3.3；§4.4
- **摘录要点：** 可变形沿 PhysTwin/EMPM，用人形沿 BFM-Zero 运动上下文；复用文件夹合约与回放/修复接口，状态变量按域替换。评测以定性对照为主。
- **对 wiki 的映射：**
  - [BFM-Zero](../../wiki/entities/paper-bfm-zero.md) — 人形适配器语境。
  - [CRISP](../../wiki/methods/crisp-real2sim.md) — 人形 Real2Sim 互补（接触平面原语 vs episode twin）。

### 4) 回放成功判据与多 VLM 成本

- **链接：** §4.1–4.3；Fig. 3
- **摘录要点：** 三角色 VLM 裁判对 start/mid/end 关键帧打分（目标物体身份、最终位置、动作相似度、夹爪位置）；≥8/10 记成功。四后端成功率 37–48%，成本差一个数量级以上。
- **对 wiki 的映射：**
  - [SimFoundry](../../wiki/entities/paper-simfoundry-real2sim-scene-generation.md) — 同属「真机→可仿真」但评测主线不同（策略相关 vs 回放成功）。
  - [embodied-eval-benchmark-selection-loop](../../wiki/queries/embodied-eval-benchmark-selection-loop.md) — 评测选型。

## BibTeX

```bibtex
@misc{chen2026agenticreal2sim,
  author       = {Guanxiong Chen and Qianjun Xia and Jiawei Peng and Heng Zhang and
                  Bole Ma and Justin Qian and Ziyi Jiao and Bingyang Zhou and
                  Luoxin Ye and Kaifeng Zhang and Kunyi Wang and Weijia Zeng and
                  Yunuo Chen and Pengzhi Yang and Ziqiu Zeng and Huamin Wang and
                  Chao Liu and Alan Yuille and Fan Shi and Changxi Zheng and
                  Yunzhu Li and Chenfanfu Jiang and Peter Yichen Chen},
  title        = {{Agentic Real2Sim}: Physics-based World Modeling with Vision-Language Agents},
  year         = {2026},
  eprint       = {2607.19190},
  archivePrefix= {arXiv},
  primaryClass = {cs.RO},
  url          = {https://arxiv.org/abs/2607.19190}
}
```

## 对 wiki 的映射

- 主实体页：[`wiki/entities/paper-agentic-real2sim.md`](../../wiki/entities/paper-agentic-real2sim.md)
- 项目页：[`sources/sites/agentic-real2sim-github-io.md`](../sites/agentic-real2sim-github-io.md)
