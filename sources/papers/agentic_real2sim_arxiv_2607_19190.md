# Agentic Real2Sim: Physics-based World Modeling with Vision-Language Agents（arXiv:2607.19190）

> 来源归档（ingest）

- **标题：** Agentic Real2Sim: Physics-based World Modeling with Vision-Language Agents
- **类型：** paper / agentic real2sim / physical digital twin / VLM agents / MuJoCo
- **arXiv：** <https://arxiv.org/abs/2607.19190>（PDF：<https://arxiv.org/pdf/2607.19190.pdf>）
- **项目页：** <https://agentic-real2sim.github.io/>
- **代码：** <https://github.com/agentic-real2sim/agentic_real2sim>（2026-11-16 核查已开放）
- **作者（v5 全 27 人）：** Guanxiong Chen、Qianjun Xia、Jiawei Peng、Heng Zhang、Pengyu Jing、Bole Ma、Justin Qian、Yixian Cheng、Ziyi Jiao、Bingyang Zhou、Yiduo Qu、Luoxin Ye、Kaifeng Zhang、Kunyi Wang、Weijia Zeng、Yunuo Chen、Pengzhi Yang、Ziqiu Zeng、Siyuan Luo、Huamin Wang、Chao Liu、Alan Yuille、Fan Shi、Changxi Zheng、Yunzhu Li、Chenfanfu Jiang、Peter Yichen Chen
- **机构：** 英属哥伦比亚大学（UBC）、约翰霍普金斯大学（Johns Hopkins）、埃尔朗根-纽伦堡大学 / NHR@FAU（FAU）、哥伦比亚大学（Columbia）、加州大学洛杉矶分校（UCLA）、新加坡国立大学（NUS）、凌迪科技（Style3D）
- **入库日期：** 2026-07-23；更新核查：2026-11-16（论文 v5：2026-10-11）
- **核对范围：** arXiv:2607.19190v5 全文 8 页（方法、实验、评测、结论与局限）；官方项目页；GitHub README 和 `ar2s/run_pipeline.py`（源码 880389b）。
- **通读：** 已完成
- **一句话说明：** 用 **VLM agent 编排**把真机交互录像（主攻 DROID）转成 **可回放的 MuJoCo episode twin**：视觉处理 → 物理先验 → 场景装配 → simulator-in-the-loop 抓取优化；同一 episode 合约可扩展到 **PhysTwin 可变形** 与 **BFM-Zero 人形运动**；开源 31B VLM 后端在 DROID-100 上与前沿闭源模型相近的回放成功率，成本约 GPT-5.4 的 **3%**。

## 开源状态（2026-11-16 核查）

- **已公开：** 官方 Code 链接对应 [agentic-real2sim/agentic_real2sim](https://github.com/agentic-real2sim/agentic_real2sim)，公开六阶段管线与命令 `python -m ar2s.run_pipeline`。
- **源码版本：** [880389b](https://github.com/agentic-real2sim/agentic_real2sim/tree/880389bda289d2e78154c73ae1b6d23b7a650557)，静态核对 README、`ar2s/run_pipeline.py`。
- **复现边界：** Linux/NVIDIA GPU、Python 3.12+、多个 Conda 环境、约 80 GB 工作空间；部分视觉权重需单独获取和授权。本次未运行端到端转换。
- **源码归档：** [sources/repos/agentic-real2sim.md](../repos/agentic-real2sim.md)。

## 摘要级要点

- **瓶颈：** Real2Sim 不止视觉重建——还需几何、物性、相机/机器人位姿、轨迹与可运行仿真装配；现状依赖手工调视觉基础模型、修 mesh、对齐坐标与脆弱工作流粘合。
- **贡献：** （1）DROID → MuJoCo episodic twin 的 agentic 管线；（2）可替换 VLM 后端（Gemma 4 31B / Qwen 3.6 35B / GPT-5.4 / Claude Haiku 4.5）；（3）同一合约适配可变形与人形。
- **指标：** v5 的 **DROID-500** 主实验中，Gemma 4 31B **242/500（48.4%）成功**、28/500 部分、230/500 失败；四模型对照另取 DROID-100 子集，Gemma **48/100**、模型账单 **$2.62**，GPT-5.4 成本约 **31.4×**。
- **下游实证：** 基于转换 episode 进行物体姿态、抓取、背景及视角扩增，微调 DROID 预训练 π0.5；另在 **6 项 pick-and-place** 任务中，用仅从真实数据微调的 π0 比较 sim/real 闭环成功率，报告 **Pearson r=0.920，p=0.009**（小样本任务相关性，不是普适保证）。

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
- **摘录要点：** v5 三独立 VLM 裁判对 start/middle1/middle2/end 关键帧打分，任一裁判最佳候选 ≥8/10 判回放成功；四后端在相同 DROID-100 子集上分别 **48/45/43/37**，与 500 条主实验分开计量。
- **对 wiki 的映射：**
  - [SimFoundry](../../wiki/entities/paper-simfoundry-real2sim-scene-generation.md) — 同属「真机→可仿真」但评测主线不同（策略相关 vs 回放成功）。
  - [embodied-eval-benchmark-selection-loop](../../wiki/queries/embodied-eval-benchmark-selection-loop.md) — 评测选型。

### 5) v5 新增：DROID-500 与下游策略评测

- **论文位置：** §IV-B、§IV-E、Fig. 6–7。DROID-500 上成功 242、部分 28、失败 230；349 条可评估 episode 平均 PSNR **14.069**、SSIM **0.650**、ADD-S **107 mm**。
- **跨硬件：** AIRBOT Play + ZED 2i 采集数据经 DROID-shaped 输入复用转换框架；真机取木勺操作与 MuJoCo twin 相对应。
- **物理与策略：** 姿态随机化、重采样抓取点、3DGS 背景替换和视角扰动后重新仿真与渲染；微调 π0.5。六任务 sim/real π0 评测 r=0.920（p=0.009），须与回放成功率分开理解。
- **局限（§V）：** 主线偏刚体，物体级系统辨识和物理参数直接验证尚未支持；可变形/Unitree G1 人形以定性展示为主。
- **对应知识节点：** [Agentic Real2Sim](../../wiki/entities/paper-agentic-real2sim.md)。

### 6) 官方运行入口静态核查

- [`ar2s/run_pipeline.py`](https://github.com/agentic-real2sim/agentic_real2sim/blob/880389bda289d2e78154c73ae1b6d23b7a650557/ar2s/run_pipeline.py) 固定阶段顺序：`visual_processing → geometry_prior → scene_view_repair → physical_prior → scene_prep → grasp_optimization`；VLM 自主决策位于子阶段。
- README 给出 `--raw-data`、`--agent-config`、`--artifact-root`、`--start-from`、`--until`，主要产物在 `sysid_inputs/` 与 `artifacts/sweep-*/`，尚未在本环境实际执行。

## BibTeX

```bibtex
@misc{chen2026agenticreal2sim,
  author       = {Guanxiong Chen and Qianjun Xia and Jiawei Peng and Heng Zhang and
                  Pengyu Jing and Bole Ma and Justin Qian and Yixian Cheng and
                  Ziyi Jiao and Bingyang Zhou and Yiduo Qu and Luoxin Ye and
                  Kaifeng Zhang and Kunyi Wang and Weijia Zeng and Yunuo Chen and
                  Pengzhi Yang and Ziqiu Zeng and Siyuan Luo and Huamin Wang and
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
- 官方源码：[`sources/repos/agentic-real2sim.md`](../repos/agentic-real2sim.md)
