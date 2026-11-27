# Embodied Agents Take Control: Minimal-Interface Zero-Shot Agents Rival Industrial-Scale Policies in Vision-and-Language Navigation

- **作者：** Jian Zhou*、Xunyi Zhao*、Gengze Zhou、Zerui Li、Sihao Lin、Jiajun Liu、Qi Wu（*同等贡献）
- **机构：** 原文 arXiv HTML 与官方项目页未展开作者上标 `1–4` 对应的机构全称（仅标明作者署名及上标序号）
- **版本 / 载体：** arXiv:2607.26148v4 [cs.RO]（2026-07-28 提交 v1，2026-10-17 修订 v4；CC BY 4.0）
- **原文：** [arXiv HTML v4](https://arxiv.org/html/2607.26148)、[arXiv 摘要页](https://arxiv.org/abs/2607.26148)
- **项目页 / 代码：** <https://jianzhou0420.github.io/mip/>（项目页 / 排行榜）；<https://github.com/jianzhou0420/MIP>（**已开源**：MIT License，含官方运行入口 `runner.py`、各工具臂与提示词配置、`splits/` 100-episode 子集及轨迹浏览器 `ui.server`；配套仿真封装 <https://github.com/jianzhou0420/EmbodiedScore-envs>；真机 Unitree Go2 部署控制端代码未见于公开目录树）
- **核对范围：** 全文通读 arXiv HTML v4（`paper_text.txt` 第 1–3405 行），覆盖 §1–5、Table 1–14、Figure 1–13 及 Appendix A–F（含 ObjectNav 假阴性审计、Opus 5.5 18 例失败全表、Unitree Go2 31 次真机探针逐条清单、AutoEA 五项系统要求、控制权×能力来源二维分类法、冻结配置、推理强度消融、成本核算、全部工具描述与提示词差异）。
- **通读：** 已完成

## 原文核对笔记

### 1. 核心命题与二维分类法（§1, §3, Appendix D–E）

1. **Agentic Embodied Control（智能体具身控制）：** 将具身导航与问答决策权从外部固定循环/工作流交还给通用多模态大模型——模型通过极简 MCP 工具接口（如仅 `observe()` 与 `step(actions)`）自主决定何时观测、移动、原地环视、纠错或终止，使推理与控制保持对齐。
2. **控制权 × 能力来源二维分类（Appendix E）：**
   - **Control Authority（控制权主轴）：** 区分 `Embodied policy`（单模型逐步映射，无多组件编排，如 NaVid、StreamVLN）、`Workflow`（人工编写固定流水线或固定速率慢-快双脑交接，如 MapGPT、SmartWay、O2C-Nav、ABot-N1、InternVLA-N1/DualVLN）与 `Agentic`（模型在运行时动态决定调用什么工具、何时重规划与何时停止，如 NavGPT、AgenticNav 与本文 Minimal-Interface Probe [MIP]）。
   - **Capability Source（能力来源次轴）：** 区分 `Trained`（针对导航优化权重）与 `Zero-shot`（冻结通用模型，不作导航权重训练）。
3. **与长期自主系统（AutoEA）的界限（Appendix D）：** 作者明确指出 MIP 仍是**回合制（episodic）**探针，尚未满足自主具身智能体（AutoEA）的五项系统级要求（跨任务无重置持续服务、持久化情景状态、闭环监控恢复、资源有界自维护、安全适应与求助）。

### 2. 接口臂设计与实验协议（§3, §4, Appendix F）

- **基准与固定 100 回合子集：** 覆盖 3 类任务共 6 个基准——R2R-CE（沿用 OpenNav 的 `rand100` val-unseen）、RxR-CE（英文 val-unseen 确定性哈希采样 `mip100`）、HM3D-ObjectNav v1/v2（`mip100`）、MP3D-ObjectNav（`mip100`）及 HM-EQA（`mip100`）。
- **冻结运行约束（Table 11）：** 512×512 RGB（HFOV 90°）、每回合最多 200 次模型调用、500 步 Habitat 底层步数上限（`step` 单次最多 50 个原语）、2400 s 超时、每回合全新会话无跨回合记忆；ObjectNav 与 HM-EQA 额外开放相机俯仰动作 `4/5`（$\pm 30^\circ$，限幅 $\pm 60^\circ$），HM-EQA 用 `answer("A"|"B"|"C"|"D")` 替代 `STOP`。
- **五种接口臂（§3.2, Table 14）：** `Minimal`、`Waypoint`（深度驱动航点预测器 + 4 视角条带 + 最多 30 次 `goto`）、`Hybrid`（原语 + 可选航点）、`SLAM`（ORB-SLAM3 RGB-D 后端提供 `get_pose` / `get_map` / `get_trajectory` / `look_around` / `mark` / `preview_path` 六项免费工具）与 `Hybrid + SLAM`。

### 3. 核心实验结论与量化数据（§4.1–§4.6, Table 1–13）

- **Minimal 接口主表（Table 1，100 回合）：** 最强冻结模型无需地图与航点器即可与工业级训练策略持平或超出其已发表全量/子集指标（非严格同配）：
  - R2R-CE：`fable-5.1` 达 **83.0 ± 1.7** SR、`opus-5.5` **82.3 ± 3.5** SR（人类测试者 94）；
  - RxR-CE：`gpt-6-astra` 达 **89** SR（人类 83）；
  - HM3D v1 / v2 / MP3D（1 m SR / $\text{SR}_{\text{vis}}$）：`gpt-6-astra` 达 **81/94**、**91/95**、**66/89**，`opus-5.5` 达 **80/92**、**88/92**、**69/88**；
  - HM-EQA：`opus-5.5` 达 **81.0%** Acc（人类 87%）。
- **工具增强效应（Table 2–3）：** 强制 `Waypoint` 显著拉升弱模型（`qwen3.5-4b` 5 → 43 SR、`qwen3.5-9b` 7 → 44 SR），但强模型几乎无增益甚至略降；改为可选 `Hybrid` 后，`opus-5.5` 升至 **85.7 ± 1.5** SR 且仅用 0.59× 调用次数与 0.88× 成本；叠加 `Hybrid + SLAM` 在所有更强模型上取得最高 SR（`opus-5.5` **88.0 ± 1.0** SR、NE 2.09 m；`fable-5.1` **86** SR）。
- **控制循环归属与脚手架复杂度（Table 4）：** 在 R2R-CE 上运行相同强模型时，固定工作流大幅退化（`opus-5.5` 从 82.3 降至 SmartWay **65** / O2C-Nav **69**；`fable-5.1` 从 83.0 降至 **62** / **62**）；而极简开源脚手架 `mini-swe-agent` 与厂商 SDK（Claude Agent SDK / Codex CLI）表现接近（`fable-5.1` 84 vs 83.0，`opus-5.5` 80 vs 82.3）。
- **失败归因与真机局限（§4.5–4.6, Appendix A–C）：** ObjectNav 的 63 例失败中 39 例为目标可见但未命中标注实例或视角（假阴性）；R2R-CE `opus-5.5` 18 例失败主因为“贪心绑定无回溯”（11/18）和“`step()` 无碰撞/位移反馈”（10/18 出现静默卡死）；Unitree Go2 31 次定性真机试验证实短时逻辑/视觉定位可行，但受限于机身后部体积感知缺失、执行器实际位移与指令偏差及跨视角计数/参照系漂移。

对 wiki 的映射：[Embodied Agents Take Control (MIP) 论文实体](../../wiki/entities/paper-agentic-2607-26148.md)。
