# Telekinesis Agentic OS — RLbotics 文档

> 来源归档

- **标题：** RLBotics – Reinforcement Learning Skills
- **类型：** 官方文档（Telekinesis Agentic Skill Library）
- **链接：** https://docs.telekinesis.ai/skills/rlbotics/overview.html
- **上级入口：** https://docs.telekinesis.ai/
- **机构：** Telekinesis GmbH
- **入库日期：** 2026-09-10
- **一句话说明：** Telekinesis Agentic OS 中 RL 技能模块的产品说明：何时使用、提供哪些 sim 训练 / sim2sim / sim2real 能力，以及 Gymnasium / mjlab / Isaac Lab 分后端教程入口。
- **代码：** https://github.com/telekinesis-ai/telekinesis-rlbotics（**已开源**，Apache 2.0）
- **PyPI：** https://pypi.org/project/telekinesis-rlbotics/
- **沉淀到 wiki：** [telekinesis-rlbotics](../../wiki/entities/telekinesis-rlbotics.md)
- **交叉归档：** [telekinesis-rlbotics.md](../repos/telekinesis-rlbotics.md)

---

## 文档摘要（overview，2026-09）

**RLBotics** 是 Telekinesis Agentic OS 面向机器人 **学习型控制** 的模块：轻量、GPU 加速的 PyTorch 库，经 **单一 pipeline** 在 Gymnasium、mjlab、Isaac Lab 上训练，导出 ONNX，部署侧 **仅依赖 NumPy**（推理用 onnxruntime）。换任务 = 换 YAML 配置路径。

### 适用场景（When to Use）

- 仿真中训练 locomotion / manipulation / control 策略
- 仿真内调试、压测与验证策略
- **Sim2Sim**：同一策略跨仿真器迁移
- **Sim2Real**：仿真到真机，统一接口
- 与感知、规划并行的 Physical AI 管线中嵌入学习控制

### 能力模块（What It Provides）

- 常见算法与多后端下的 RL 策略训练
- 仿真中运行与调试策略
- Sim2Sim / Sim2Real 部署技能
- 教程：Gymnasium PPO → TensorBoard → ONNX → NumPy 真机推理闭环

### 许可与分发

文档声明 **Apache 2.0 开源**；包名 `telekinesis-rlbotics`（PyPI），源码 GitHub 互链。

---

## 对 wiki 的映射

- [telekinesis-rlbotics](../../wiki/entities/telekinesis-rlbotics.md)
- [telekinesis-rlbotics 仓库摘录](../repos/telekinesis-rlbotics.md)
