# AtomicVLA（原子技能 VLA）

> 来源归档（ingest）

- **标题：** AtomicVLA: Unlocking the Potential of Atomic Skill Learning in Robots
- **类型：** paper
- **原始链接：** <https://arxiv.org/abs/2603.07648>
- **HTML：** <https://arxiv.org/html/2603.07648v1>
- **项目页：** <https://zhanglk9.github.io/atomicvla-web/>
- **代码：** <https://github.com/zhanglk9/AtomicVLA>
- **权重：** <https://huggingface.co/likui/AtomicVLA-libero>
- **机构：** 中山大学（SYSU）、鹏城实验室（PCL）、引望智能科技（Yinwang Intelligent Technology Co. Ltd.）
- **会议：** CVPR 2026
- **入库日期：** 2026-09-18
- **一句话说明：** 统一规划–执行 VLA：联合生成任务级计划、原子技能抽象与细粒度动作；用 Skill-Guided MoE（SG-MoE）建可扩展原子技能库，routing encoder 支持持续扩技能；基于 openpi 栈开源 LIBERO 训练/推理。

## 核心摘录（策展）

### 1) 问题：单体动作解码器难以扩展长程与终身技能

- **摘录要点：** 现有 VLA 多在聚合数据上训练 **单体 action decoder**，长程多步任务与 **持续技能获取** 泛化差；真实任务需要跨多个原子技能组合，而非单次动作拟合。
- **对 wiki 的映射：**
  - [AtomicVLA](../../wiki/entities/paper-atomicvla.md) — 问题设定。
  - [VLA](../../wiki/methods/vla.md) — 技能分解与 MoE 谱系。

### 2) 方法：统一规划–执行 + SG-MoE + routing encoder

- **摘录要点：** **AtomicVLA** 在同一框架内联合输出 **task-level plan**、**atomic skill abstraction** 与 **fine-grained actions**。通过 **Skill-Guided Mixture-of-Experts（SG-MoE）** 构建可扩展 **原子技能库**，各 expert 专精通用且精确的原子技能；**flexible routing encoder** 为新技能自动分配 dedicated expert，支持 **continual learning**。训练需 **结构化 reasoning annotation**（episode 级 JSON：帧段、primary_action_verb、chain_of_thought 等；README 示例 pick/place 分段）。
- **对 wiki 的映射：**
  - [AtomicVLA](../../wiki/entities/paper-atomicvla.md) — 流程图与时序图。
  - [π₀](../../wiki/entities/paper-pi0.md) — openpi 基座对照。

### 3) 仿真评测（摘要口径）

- **摘录要点：** 相对 **π₀**：LIBERO **+2.4%**、LIBERO-LONG **+10%**；CALVIN 平均任务长度相对 **π₀ / π₀.₅** 分别 **+0.22 / +0.25**。
- **对 wiki 的映射：**
  - [LIBERO benchmark](../../wiki/entities/libero-benchmark.md)
  - [CALVIN benchmark](../../wiki/entities/calvin-benchmark.md)

### 4) 真机长程与持续学习

- **摘录要点：** 真机长程任务相对基线 **+18.3%**；持续学习设定 **+21%**（摘要口径）。
- **对 wiki 的映射：**
  - [AtomicVLA](../../wiki/entities/paper-atomicvla.md) — 真机读法。
  - [OrthoSkillVLA](../../wiki/entities/paper-orthoskillvla.md) — 另一 VLA 持续技能学习对照。

### 5) 开源状态（截至 2026-09-18，项目页 + GitHub 核查）

- **摘录要点：** **已开源** MIT，基于 [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi)；`uv` 环境、`scripts/compute_norm_stats.py` + `scripts/train.py Atomic_libero` 训练 LIBERO；`scripts/serve_policy.py` openpi 式 policy server + 硬件 client 部署；HF 发布 [likui/AtomicVLA-libero](https://huggingface.co/likui/AtomicVLA-libero)。致谢 InternVideo 与 OneTwoVLA。
- **对 wiki 的映射：**
  - [atomicvla 仓库](../repos/atomicvla.md)
  - [atomicvla 项目页](../sites/atomicvla-zhanglk9-github-io.md)

## 当前提炼状态

- [x] arXiv / 项目页 / GitHub / HF 已交叉核查
- [x] wiki 映射：`wiki/entities/paper-atomicvla.md`
