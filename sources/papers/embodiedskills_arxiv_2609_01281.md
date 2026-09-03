# EmbodiedSkills: A Unified Framework for Orchestrating, Training, and Deploying VLA Agents（arXiv:2609.01281）

> 来源归档（ingest）

- **标题：** EmbodiedSkills: A Unified Framework for Orchestrating, Training, and Deploying VLA Agents
- **类型：** paper / vla / agentic / manipulation / closed-loop / skill-orchestration
- **arXiv abs：** <https://arxiv.org/abs/2609.01281>
- **PDF：** <https://arxiv.org/pdf/2609.01281>
- **代码：** <https://github.com/DCDmllm/EmbodiedSkills> — 归档见 [`sources/repos/embodiedskills.md`](../repos/embodiedskills.md)
- **机构：** 浙江大学（ZJU，主团队）；南京航空航天大学；康奈尔大学；云深处科技（Deep Robotics）；新加坡国立大学（NUS）等
- **作者（摘要页）：** Wei Wang、Wenqiao Zhang*、Yutong Lin、Yuqian Yuan、Tianwei Lin、Jinhao Mao、Zhenxuan Fan、Mingjian Gao、Yang Dai、Wentong Li、Zheqi Lv、Zheng Dong、Yingjie Niu、Jiaqi Zhu、Jun Xiao、Chao Li、Yueting Zhuang（* 同等贡献；Wenqiao Zhang 通讯）
- **入库日期：** 2026-09-03
- **一句话说明：** 将长时程 VLA **agent** 建模为 **guarded AgentLoop**：可执行 skill contract（preflight → bounded VLA chunk → verify → recover），policy 提案与 runtime 强制分离；实例化 Qwen3-VL + OpenPI/π₀.₅，RoboTwin 2.0 **86.20%**、LIBERO **97.40%**；RMBench 记忆任务 **12.5%**。

## 开源状态（GitHub 核查，2026-09-03）

- **已开源：** [`DCDmllm/EmbodiedSkills`](https://github.com/DCDmllm/EmbodiedSkills) 发布 AgentLoop、三 benchmark 适配器、OpenPI 集成、子任务 π₀.₅ 数据构建与 Qwen3-VL LoRA 训练脚本。
- **部分 / 外部：** 预训练 checkpoint、大规模轨迹与历史 RL 实验 **不在** 源码树；需按 README 采集或训练。
- **LICENSE：** 仓库 API 返回 `license: null`；部署前请核对仓库许可文件。

## 摘要级要点

- **问题：** 端到端 VLA 擅长短时动作预测，但长时程操作需要 **感知、规划、执行、验证、恢复** 的闭环；模型生成的 skill 决策不保证 **当前状态可执行** 或 **物理结果达标**。
- **方法：** **EmbodiedSkills** — 每个 skill 是带 **typed I/O、prerequisites、post 更新与 fail 映射** 的可执行契约；**AgentLoop** 在六阶段语义空间（Observe / Plan / Preflight / Execute / Verify / Recover）上运行，**非固定流水线**，可由策略重入各阶段。
- **Policy–runtime 分离：** 高层策略输出 `RunSkill` / `AdvanceStage` / `FinishRun`；runtime 用 guard 检查 phase、artifact freshness、action validity；blocked 决策写入 trace 而非静默执行。
- **模块化训练：** 统一 trajectory schema 记录规划、执行、验证与恢复事件；planner / verifier / 低层 VLA 可 **独立 SFT 或替换**，无需改 AgentLoop。
- **实例化：** Qwen3-VL agent 组件 + **OpenPI/π₀.₅** 低层；RoboTwin 2.0 50 任务均值 **86.20%**（高于 LingBot-VA 报道 π₀.₅ **82.74%**）；LIBERO 四套件均值 **97.40%**（OpenPI 参考 **96.85%**）；RMBench 四记忆任务 **12.5%**（X-VLA **7.3%**）。

## 核心摘录（面向 wiki 编译）

### 1) 六阶段 AgentLoop（Table 1）

| Phase | 角色 |
|-------|------|
| Observe | 采集当前视觉与任务证据 |
| Plan | 语义子目标与执行预算 |
| Preflight | 就绪检查（上下文、观测、机器人状态、VLA 可用性） |
| Execute | 有界 action chunk（低层 VLA） |
| Verify | 用执行后观测判断 Advance / Continue / Reobserve / Replan / Recover |
| Recover | 失败后修订子目标或计划并安全重入 |

### 2) RoboTwin 2.0 消融（README / 论文对照）

| 设定 | 平均 SR |
|------|---------|
| 完整 AgentLoop | **86.20%** |
| 去掉中间 verification | 48.2% |
| 全任务指令替代语义子任务 | 34.4% |
| 每子任务仅 1 个 action chunk | 19.5% |

### 3) 对 wiki 的映射

| 主题 | 目标页 |
|------|--------|
| 论文实体 | `wiki/entities/paper-embodiedskills.md` |
| VLA agent 层 | `wiki/methods/vla.md` |
| 冻结 VLA 编排对照 | `wiki/entities/paper-harness-vla.md` |
| RoboTwin 基准 | `wiki/entities/robotwin.md` |

## 推荐继续阅读

- [EmbodiedSkills GitHub](https://github.com/DCDmllm/EmbodiedSkills) — 安装、`embodiedskills-run`、训练文档
- [arXiv:2609.01281](https://arxiv.org/abs/2609.01281) — skill contract 与 guarded transition 形式化
- [Harness VLA（arXiv:2607.08448）](../../wiki/entities/paper-harness-vla.md) — 冻结 VLA 原语 + 记忆编排对照
