# MemCompiler：状态条件记忆编译

> 来源归档（ingest）

- **标题：** MemCompiler: Compile, Don't Inject — State-Conditioned Memory for Embodied Agents
- **类型：** paper
- **原始链接：** <https://arxiv.org/abs/2605.07594>
- **版本：** arXiv:2605.07594v2（2026-05-14）
- **作者：** Xin Ding、Xinrui Wang、Yifan Yang、Hao Wu、Shiqi Jiang、Qianxi Zhang、Liang Mi、Hanxin Zhu、Kun Li、Yunxin Liu、Zhibo Chen、Ting Cao
- **机构：** 中国科学技术大学、华中科技大学、Microsoft Research、南京大学、清华大学 AIR
- **项目页：** <https://air-embodied-brain.github.io/MemCompiler/>
- **代码：** <https://github.com/air-embodied-brain/MemCompiler>
- **模型页：** <https://huggingface.co/xinrui-23/thor-5.0-16-Qwen2.5-14B-Instruct-Qwen2.5-7B-Instruct>
- **开放声明：** 论文称 code and data available；项目页提供代码、模型入口。模型页当前无 model card；数据获取/许可与权重适用范围需按各自仓库核实。
- **入库日期：** 2026-10-11
- **一句话说明：** 每个执行步依据 Brief State 编译当前有用的记忆，借助文本指导与 Soft-Mem 潜在通道减少静态上下文错位及注意力稀释。

## 核心摘录（MVP）

### 1) 从「存什么」转向「何时、如何送达」

- **摘录要点：** 论文将 episode 开始时一次性注入候选记忆、并在执行中保持不变的做法称为 Ahead-of-time Monolithic Memory Injection（AMMI）。随着子目标完成、观察与环境信念变化，静态上下文与当前需求错位；轻量 Executor 可能因注意力稀释而低于无记忆基线。
- **对 wiki 的映射：** [MemCompiler 实体页](../../wiki/entities/paper-memcompiler-state-conditioned-memory.md)；[Spatial Memory Agent](../../wiki/entities/paper-spatial-memory-agent.md)。

### 2) State-Conditioned Memory Compilation

- **摘录要点：** State-Conditioned Memory Compilation（SCMC）将记忆库视为源材料，在 episode 开始时基于目标检索候选池；此后 Memory Compiler 在每一步读取当前观察与 Brief State，并按实时状态挑选、编译适用条目。Brief State 结构化压缩目标、历史动作/观察、子目标进展和环境信念，可持续更新。四类输出：EXPERIENCE（给经验）、BRIEF（更新状态）、HYBRID（两者都有）、NOACTION（当前无适用记忆）。
- **对 wiki 的映射：** [MemCompiler 实体页](../../wiki/entities/paper-memcompiler-state-conditioned-memory.md)；[Zeva](../../wiki/entities/paper-zeva.md)（自身交互记忆路线对照）。

### 3) 文本 + Soft-Mem 双通道

- **摘录要点：** Memory Compiler 同时生成可读文本指导和注入 Executor embedding 的 Soft-Mem 潜在 token；后者传递难以文字表达的空间/感知信息，并通过正交约束避免退化为文本的重复。论文配置 16 个 soft prompt tokens。
- **对 wiki 的映射：** [MemCompiler 实体页](../../wiki/entities/paper-memcompiler-state-conditioned-memory.md)。

### 4) 训练、评测与效率

- **摘录要点：** Compiler 与 Soft-Mem 先经监督微调，再以任务成功的二值奖励做 GRPO；RL 阶段冻结 Executor。论文在 ALFWorld、EmbodiedBench（EB-ALFRED、EB-Habitat）和 ScienceWorld 上评测，覆盖四个开源及两个闭源骨干。相对无记忆开源 Executor，报告最高 +110%（EB-ALFRED）与 +129%（ScienceWorld）；ALFWorld 中 SCMC 成功率 35.03%→82.16%，Executor 输入 token 2481.1→1003.2、延迟 0.30s→0.12s。摘要将延迟概括为约降低 60%。
- **对 wiki 的映射：** [MemCompiler 实体页](../../wiki/entities/paper-memcompiler-state-conditioned-memory.md)。

## 实现入口与开放边界

- 官方 README 给出 `bash run_alfworld.sh`、`bash run_scienceworld.sh`，以及 `tasks/run_manager_executor*.py` 入口；已训练模型评测入口为 `tasks/eval_with_trained_model.py`。
- 仓库说明 Manager–Executor 架构、Insight / Query / Interaction Graph 记忆组织。
- 复现依赖 ALFWorld / ScienceWorld 环境、数据、LLM API key 或本地模型；README 要求按配置提供数据与模型路径。
- 项目页所列 Hugging Face 模型条目当前无 model card，不能据此推断论文权重、数据和许可均已明确可复用。

## 当前提炼状态

- [x] 核验 arXiv v2、官方项目页与官方 GitHub README
- [x] wiki 映射：[paper-memcompiler-state-conditioned-memory.md](../../wiki/entities/paper-memcompiler-state-conditioned-memory.md)
