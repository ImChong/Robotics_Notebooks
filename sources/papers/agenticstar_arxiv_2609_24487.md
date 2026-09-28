# AgentSTAR: Agentic Shape Tracking and Reconstruction from Monocular Videos（arXiv:2609.24487）

> 来源归档（ingest）

- **标题：** AgentSTAR: Agentic Shape Tracking and Reconstruction from Monocular Videos
- **类型：** paper / monocular video / articulated object / analysis-by-synthesis / VLM agent / shape tracking
- **arXiv：** <https://arxiv.org/abs/2609.24487>（PDF：<https://arxiv.org/pdf/2609.24487.pdf>）
- **项目页：** <https://agenticstar.github.io/>
- **代码：** <https://github.com/makezur/agenticSTAR>（MIT；**已开源** harness + 示例 capture）
- **作者：** Kirill Mazur、Nikita Karaev、Matthew Chang、Jitendra Malik、Nur Muhammad “Mahi” Shafiullah
- **机构：** 亚马逊 FAR（Amazon FAR, Frontier AI and Robotics）；加州大学伯克利分校（UC Berkeley，Jitendra Malik）
- **入库日期：** 2026-09-28
- **一句话说明：** 从 casual **单目视频** 用 **VLM 编码 agent + render-and-compare** 联合恢复 **Blender 程序化物体模型（几何+关节）** 与 **广义位姿序列**；数值侧以 **mask IoU** 与 **有界 pose sweep** 精修；在 **ARCTIC** 上 3D EPE **5.59 cm**、Chamfer **3.36 cm**，**HOT3D** 刚性跟踪平移中位 **2.42 cm**，**iTACO** 运动学显著优于对照。

## 开源状态（项目页 + GitHub 核查，2026-09-28）

- **已开源：** 项目页链到 [makezur/agenticSTAR](https://github.com/makezur/agenticSTAR)；README 给出 `install.sh`、`tools/run_kf.sh`（Claude Code / Codex agent）、Bubblewrap 沙箱、示例 `examples/garden_shears/`。
- **运行边界：** 需 Linux x86_64 + NVIDIA GPU、API key（Anthropic/OpenAI）；单次 supervised run **数小时 + 大量 token**；论文主结果用 **`--enable mechanism`**（默认关闭）与 GPT-5.6-Sol **`critic` 分支**；新模型 README 建议 mechanism off、可无 critic。
- **上游依赖：** 相机来自 [Pi3X](https://github.com/yyfz/Pi3)（`tools/make_capture.py`）；mask 任意分割器（论文语境含 SAM3）。

## 摘要级要点

- **范式：** **自上而下 analysis-by-synthesis**——先推断结构化 3D 模型（代码 + 关节），再用同一 agent 在时间上优化 **generalised pose**（6-DoF 基座 + 关节角），而非先稠密点轨迹再事后拟合 articulation。
- **Agent 环：** 每轮 **改 shape（scene.py 代码 diff）** 或 **改 pose（VLM 指定搜索区间 → 数值优化 top-K → VLM 目视选优）**；序列级 **temporal diagnostic** 报告不连续，由 agent 决定是否平滑（非固定时序先验）。
- **打分：** 主目标为渲染 silhouette 与物体 mask 的 **IoU**（手部 mask 剔除）；可选 depth 混合项（默认不用 depth）。
- **输出：** `mesh/object.glb`（命名部件）+ `mesh/pose.json`（逐帧位姿与关节状态）。

## 核心论文摘录（MVP）

### 1) 任务与表示

- **链接：** §3；Eq. (1)–(2)
- **摘录要点：** 输入关键帧 RGB、已知相机内外参（外部 SLAM / 前馈重建如 Pi3X）、物体 mask（+ 可选 hand mask）。canonical 模型 \(\mathcal{O}\) 为 **Python/Blender 原语代码**，共享全序列；每帧 \(\mathcal{T}_i=(T_i, j_{i,1},\ldots)\)。
- **对 wiki 的映射：**
  - [AgentSTAR](../../wiki/entities/paper-agenticstar.md) — 任务定义与 pose 参数化。
  - [Articraft](../../wiki/entities/articraft.md) — 同属 **代码化可关节几何**，但 AgentSTAR 主攻 **视频跟踪** 而非静态资产生成。

### 2) Agentic render-and-compare 环

- **链接：** §3.1–3.2；Fig. 2
- **摘录要点：** **Shape step：** coding agent 在单一 `scene.py` 写几何与关节，diagnostic render 自检。**Pose step：** 冻结 shape，agent 给出 interpretable 搜索盒（yaw/铰链区间等），harness **sweep/apply** 数值打分，VLM 在 top 候选中选视觉正确者（不必是 IoU 最高）。
- **对 wiki 的映射：**
  - [AgentSTAR](../../wiki/entities/paper-agenticstar.md) — 流程总览与时序图。
  - [Agentic Real2Sim](../../wiki/entities/paper-agentic-real2sim.md) — 另一 VLM agent + 工具 harness 的 Real2Sim 叙事（episode twin vs 单物体 track）。

### 3) ARCTIC / HOT3D / iTACO 评测

- **链接：** §4.2–4.3；Tab. 1–4
- **摘录要点：** **ARCTIC**（s1 ego，24 seq）：3D EPE **5.59 cm** vs V-DPM **7.65**；Chamfer **3.36 cm** vs **4.72**。**HOT3D**（93 seq，15 keyframes）：平移 mean/median **3.04 / 2.42 cm**，旋转 **37.6 / 26.3°**，优于 FoundationPose*+VGGT-Ω、SAM3D-Tracker 等。**iTACO**：几何 CD 与 SOTA 竞争，**revolute/prismatic 轴与状态** 全面领先 Articulate-Anything / Robot See Robot Do / iTACO。
- **对 wiki 的映射：**
  - [AgentSTAR](../../wiki/entities/paper-agenticstar.md) — 指标表与读数。
  - [Macrodata Egocentric Hand-Action](../../wiki/methods/macrodata-egocentric-hand-action.md) — 同用 HOT3D 但任务为 **手轨迹** 而非物体 6-DoF。

### 4) Harness 消融

- **链接：** Tab. 2；§4.4
- **摘录要点：** 无 harness **11.26 cm** EPE；仅 IoU 工具 **151.46 cm**（flat silhouette 投机）；GT mesh + 无 VLM pose 引导 **14.60 cm**；无 temporal **6.15 cm**；完整系统 GPT-5.6-Sol **5.59 cm**，Fable 5 **4.95 cm**。
- **对 wiki 的映射：**
  - [AgentSTAR](../../wiki/entities/paper-agenticstar.md) — 结论与工程边界。

## BibTeX

```bibtex
@misc{mazur2026agenticstar,
  title         = {{AgentSTAR}: Agentic Shape Tracking and Reconstruction from Monocular Videos},
  author        = {Mazur, Kirill and Karaev, Nikita and Chang, Matthew and Malik, Jitendra and Shafiullah, Nur Muhammad},
  year          = {2026},
  eprint        = {2609.24487},
  archivePrefix = {arXiv},
  url           = {https://arxiv.org/abs/2609.24487}
}
```

## 对 wiki 的映射

- 主实体页：[wiki/entities/paper-agenticstar.md](../../wiki/entities/paper-agenticstar.md)
