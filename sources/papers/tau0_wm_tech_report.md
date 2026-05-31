# τ₀-World Model（统一视频–动作世界模型技术报告）

> 来源归档（ingest）

- **标题：** τ₀-World Model: A Unified Video-Action World Model for Robotic Manipulation
- **类型：** paper（技术报告 / 项目 PDF，截至 ingest 未见 arXiv 条目）
- **原始链接：**
  - <https://finch-static.agibot.com/VAM/blog/tau_0_wm.pdf>
  - <https://finch.agibot.com/research/tau0-wm>
- **代码：** <https://github.com/sii-research/tau-0-wm>
- **权重：** <https://huggingface.co/sii-research/tau-0-wm>
- **发布日期：** 2026-05-31（项目页 / GitHub News）
- **入库日期：** 2026-05-31
- **一句话说明：** **5B** Joint **Video-Action Model（VAM）**：在 **Wan-2.2** 级视频扩散骨干上共享表征，联合训练 **未来多视角 latent** 与 **连续 action chunk**；异构 **~2.73 万小时** 数据用 **模态掩码** 分监督；推理端 **策略 + 动作条件仿真器** 与 **Re-denoising Consistency Score** 驱动的 **propose–evaluate–revise** 测试时计算。

## 核心摘录（MVP）

### 1) 异构数据与「各监督各所能」

- **摘录要点：** 机器人遥操作提供 **动作接地** 但场景窄；UMI 扩展行为与环境多样性；自我中心人视频提供 **物体运动、接触、空间结构** 但无机器人动作。统一 formulation 下：**有视频则监督未来观测；有机器人控制则监督动作；有进度/失败标签则监督任务进度**；缺失相机或模态 **mask 掉**。
- **对 wiki 的映射：**
  - [τ₀-World Model](../../wiki/entities/tau0-world-model.md) — 数据与掩码训练节。
  - [robot-world-models-training-loop-taxonomy](../../wiki/overview/robot-world-models-training-loop-taxonomy.md) — ① 策略内预测 + ③ 可控视频生成的交界。

### 2) VAM：共享视频扩散 + 动作 cross-attention

- **摘录要点：** 输入 **多视角观测、语言指令、机器人状态**；**视频 分支** 建模时序场景动力学；**动作 分支** 通过 **逐层 cross-attention** 读取视频中间表征，输出 **可执行 action chunk**。未来预测成为 **控制相关** 训练目标，而非独立辅助 loss。
- **对 wiki 的映射：**
  - [World Action Models](../../wiki/concepts/world-action-models.md) — Joint WAM 族。
  - [mimic-video](../../wiki/methods/mimic-video.md) — 对照「冻结骨干 + 边际动作头」vs **端到端联合 + 测试时仿真**。

### 3) 动作条件视频仿真器与任务进度

- **摘录要点：** 给定当前观测、指令与 **候选 action chunk**，模型预测 **多视角未来** 与 **稠密 task-progress 轨迹**（子任务进度标签 + 失败数据）；用于 **视觉合理性 + 任务推进** 双维度评估，减少接触丰富任务上的真机试错。
- **对 wiki 的映射：**
  - [Video-as-Simulation](../../wiki/concepts/video-as-simulation.md) — 评估型 rollout。
  - [GE-Sim 2.0](../../wiki/entities/ge-sim-2.md) — 同系「世界裁判 / 进度信号」叙事对照（GE-Sim 用独立 World Judge VLM）。

### 4) 测试时：Propose → Evaluate → Revise

- **摘录要点：** 策略先 **采样多个 action chunk**，用 **Re-denoising Consistency Score** 与所学动作分布的一致性排序；若无一候选达标，则 **仿真候选未来 → 选最优 rollout → 条件化第二次动作预测**。把额外算力花在 **执行前** 的动作筛选与修正。
- **对 wiki 的映射：**
  - [τ₀-World Model](../../wiki/entities/tau0-world-model.md) — 推理闭环 Mermaid。
  - [Generative World Models](../../wiki/methods/generative-world-models.md) — 「反应式策略 → 预测式操纵」讨论。

### 5) 真机与开源状态

- **摘录要点：** 四类 **预训练未见** 真机任务上 **平均成功率领先**；Faucet 等严格对齐任务全体仍难但 τ₀-WM 更稳。开源：**VAM 权重**（HF）、**策略部署 server**；**Simulator 权重** 与 **测试时计算** 代码 **即将发布**（README 口径）。
- **对 wiki 的映射：**
  - [Manipulation](../../wiki/tasks/manipulation.md) — 精细操作与测试时搜索语境。

## 当前提炼状态

- [x] 项目页 + GitHub README + HF 元数据已对齐摘录
- [x] wiki 映射：`wiki/entities/tau0-world-model.md` 新建，并与 WAM / mimic-video / GE-Sim 2.0 交叉引用
- [ ] 待 arXiv 发布后补 abs 链接并同步 frontmatter `arxiv` 字段
