# RoboFoundry: System-as-Policy Evolution for Self-Learning Embodied Agents（arXiv:2609.32862）

> 来源归档（ingest）

- **标题：** RoboFoundry: System-as-Policy Evolution for Self-Learning Embodied Agents
- **类型：** paper / agentic / embodied-agent / system-evolution / memory / skill-hierarchy
- **arXiv abs：** <https://arxiv.org/abs/2609.32862>
- **PDF：** <https://arxiv.org/pdf/2609.32862>
- **项目页：** <https://jingsongliang.com/robofoundry/> — 归档见 [`sources/sites/robofoundry-project.md`](../sites/robofoundry-project.md)
- **Hugging Face Papers：** <https://huggingface.co/papers/2609.32862>
- **机构：** 南洋理工大学（NTU）；北京航空航天大学（Beihang）；新加坡国立大学（NUS）；云蝶科技（Cloud Butterfly Technology）；上海交通大学（SJTU）
- **作者（摘要页）：** Jingsong Liang*、Shuhao Liao*、Shizhe Zhang、Diyuan Hou、Yuxin Cai、Xinjian Deng、Chengyang He、Wenhui Huang、Runjia Tan、Zhidong Wang、Lan Yu、Xuesong Tian、Guillaume Sartoretti、Jie Luo、Yao Mu、Wenjun Wu‡、Wanhua Li‡、Chen Lv‡（* 同等贡献；‡ 通讯）
- **入库日期：** 2026-09-30
- **一句话说明：** 将 **支撑具身 Agent 的整个系统**（context + skill + 语义/执行绑定）视为可演化的 **System-as-Policy**；通过 **Act–Reflect–Repair–Promote** 把执行轨迹转为经 held-in / held-out 校验的系统修订，并在 EmbodiedBench、RoboMemArena、LIBERO-PRO 与多真机任务上报告 SOTA 级增益。

## 开源状态（项目页核查，2026-09-30）

- **代码：** 项目页 **未列出** GitHub / Hugging Face 代码仓或权重下载入口（仅有论文、视频与 benchmark 结果面板）。
- **判定：** **截至入库日未开源**；后续若项目页挂链，应补 `sources/repos/` 与 wiki「工程实践」。

## 摘要级要点

- **问题：** 现有工作多优化 agent 栈中的 **单一组件**（harness、记忆、技能库、code-as-policy）；交互本身若不转为 **持久、经校验的系统变更**，则难以 **自进化**。
- **范式：** **Self-Evolving System-as-Policy** — 冻结 foundation model \(M\)，演化支撑系统 \(H(M)=H_g+H_t\)，通过 **filesystem 读写**（cat/grep/add/modify/delete）修订 context 与 skill 面。
- **双演化面：** **Context**（episode 内 active context + 跨 episode 文件系统记忆；SAVE/RETRIEVE/UTILIZE）；**Skill**（原子技能、组合、failure-conditioned recovery tree）。
- **语义接口：** **BIND_s**（具身不变语义）与 **BIND_e**（具身相关执行：VLA、coding agent、CuRobo 等）分离，支持 **跨机器人 zero-shot 迁移**。
- **外环：** 任务级 \(H_t\) 修复 → 经 trace 归因与 held-out 检验后 **promote** 到通用 \(H_g\)。

## 核心摘录（面向 wiki 编译）

### 1) EmbodiedBench（项目页，四套件算术平均 Avg.）

| 配置 | Avg. | EB-ALFRED | EB-Habitat | EB-Navigation | EB-Manipulation |
|------|------|-----------|------------|---------------|-----------------|
| RoboFoundry (GPT-6 Astra) | **78.0** (+6.1 pp) | 90.0 | 86.7 | 80.2 | 54.9 |
| RoboFoundry (GPT-5.5) | **72.7** (+15.8 pp) | 84.0 | 88.7 | 72.0 | 45.9 |
| RoboFoundry (Qwen3.7-Plus) | **70.3** (+10.4 pp) | 81.3 | 80.3 | 71.9 | 47.7 |
| GPT-5.5 基线 | 56.9 | 76.7 | 64.0 | 54.7 | 32.1 |

### 2) RoboMemArena（TSR / CSR %）

| Method | Overall |
|--------|---------|
| **RoboFoundry** | **53.5 / 72.8** |
| PrediMem | 38.5 / 55.2 |
| MemER | 27.3 / 49.1 |

### 3) LIBERO-PRO（position / task SR %，节选）

| Method | Object | Goal | Spatial |
|--------|--------|------|---------|
| **RoboFoundry** | 96.0 / **98.0** | 88.0 / 86.0 | 92.0 / 91.5 |
| Harness VLA (CC) | 94.0 / 80.0 | 88.0 / 90.0 | 87.0 / 87.5 |
| CaP-Agent0 | 21.8 / 18.2 | 25.6 / 16.8 | 11.8 / 14.0 |

### 4) 对 wiki 的映射

| 主题 | 目标页 |
|------|--------|
| 论文实体 | `wiki/entities/paper-robofoundry.md` |
| Agentic VLA 层 | `wiki/methods/vla.md` |
| 冻结 VLA harness 对照 | `wiki/entities/paper-harness-vla.md` |
| Skill contract 对照 | `wiki/entities/paper-embodiedskills.md` |

## 推荐继续阅读

- [RoboFoundry 项目页](https://jingsongliang.com/robofoundry/) — 视频、全表与真机片段索引
- [arXiv:2609.32862](https://arxiv.org/abs/2609.32862) — System-as-Policy 形式化与 Algorithm 1
- [Harness VLA（arXiv:2607.08448）](../../wiki/entities/paper-harness-vla.md) — 冻结 VLA + 记忆编排（LIBERO-PRO 同榜对照）
