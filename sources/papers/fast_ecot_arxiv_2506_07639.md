# Fast ECoT（思维复用加速）

> 来源归档（ingest）

- **标题：** Fast ECoT: Efficient Embodied Chain-of-Thought via Thoughts Reuse
- **类型：** paper
- **原始链接：** <https://arxiv.org/abs/2506.07639>
- **机构：** 伦敦大学学院（UCL）；弗莱堡大学（University of Freiburg）；思科研究（Cisco Research）
- **作者：** Zhekai Duan、Yuan Zhang、Shikai Geng、Gaowen Liu、Joschka Boedecker、Chris Xiaoxuan Lu
- **代码：** <https://github.com/kevinDuan1/Fast-ECoT>
- **入库日期：** 2026-09-09
- **一句话说明：** 推理时加速 ECoT：跨时间步缓存复用高层推理、并行生成模块化推理步、异步调度解耦推理与动作解码；无需改模型或重训，LIBERO + 真机最高 **7.5×** 降延迟。

## 核心摘录（MVP）

### 1) 问题：ECoT 自回归延迟阻塞实时部署

- **摘录要点：** ECoT 提升 VLA 性能与可解释性，但逐步自回归生成推理 token 导致推理延迟过高。Fast ECoT 利用 ECoT **结构化且跨步重复** 的特性做 **纯推理时** 加速。
- **对 wiki 的映射：**
  - [Fast ECoT](../../wiki/entities/paper-fast-ecot.md) — 加速框架。
  - [ECoT](../../wiki/entities/paper-ecot.md) — 被加速的奠基范式。

### 2) 三件套：复用 / 并行 / 异步

- **摘录要点：**
  - **Thought reuse**：缓存上一时刻高层推理（task / plan / subtask）；Bridge V2 上 plan 模块平均更新率仅 **8.4%**。
  - **Parallel generation**：模块化推理步 + continuous batching，共享历史前缀并行生成。
  - **Async scheduler**：动作解码优先，推理在后台异步刷新缓存。
- **对 wiki 的映射：**
  - [Fast ECoT](../../wiki/entities/paper-fast-ecot.md) — 方法细节。
  - [VLA](../../wiki/methods/vla.md) — 部署延迟轴。

### 3) 评测

- **摘录要点：** LIBERO 四套件（Spatial / Object / Goal / Long）仿真 + Franka 真机操作；相对原生 ECoT **最高 7.5×** 延迟下降，成功率与推理忠实度 **持平或提升**；基于 `Embodied-CoT/ecot-openvla-*` 检查点，可用 vLLM。
- **对 wiki 的映射：**
  - [Fast ECoT](../../wiki/entities/paper-fast-ecot.md) — 工程读法。

### 4) 开源状态（截至 2026-09-09）

- **摘录要点：** **已开源** MIT。`kevinDuan1/Fast-ECoT` 基于 ECoT + OpenVLA；`vla-scripts/deploy.py`、`--async_engine` / `--use_vllm` 评测开关；**无独立项目页**，以 GitHub README 为入口。
- **对 wiki 的映射：**
  - [fast-ecot 仓库](../repos/fast-ecot.md)

## 当前提炼状态

- [x] arXiv / README 已对齐摘录
- [x] 仓库已交叉核查（**已开源**；无独立项目页）
- [x] wiki 映射：`wiki/entities/paper-fast-ecot.md` 新建
