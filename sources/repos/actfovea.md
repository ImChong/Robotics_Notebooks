# ActFovea（VLA 运行时防护参考实现）

> 来源归档

- **标题：** ActFovea — Runtime Safeguarding for VLA Policies
- **类型：** repo
- **来源：** 论文作者（GitHub: `SunnyYWD`）
- **链接：** <https://github.com/SunnyYWD/ActFovea>
- **论文：** <https://arxiv.org/abs/2607.29169>（Submitted 2026-07-31）
- **许可：** Apache-2.0（含 Gemma 许可条款，随 π₀ / PaliGemma 主干传递）
- **入库日期：** 2026-08-04
- **一句话说明：** 在 **openpi** 基础上加一层运行时防护：把冻结的 π₀ 策略包一层 `DefensePolicy`，在推理时做威胁检测（观测新鲜度 / 几何一致性 / 视觉相关性）与恢复，并在观测不可信时执行安全 hold 或受控失败。
- **沉淀到 wiki：** [`wiki/entities/paper-actfovea.md`](../../wiki/entities/paper-actfovea.md)

---

## 核心定位

**训练自由（training-free）、无新增可学习参数**的推理期包装器。仓库不发布新权重——π₀ checkpoint 走官方渠道（`OPENPI_CHECKPOINT_DIR`），仓库贡献在防护逻辑与四条件评测 harness。

---

## 仓库入口（README / 目录）

| 组件 | 路径 / 命令 | 说明 |
|------|------------|------|
| 防护逻辑 | `src/openpi/defense/` | 威胁检测（freshness / 几何 / 视觉相关性）与恢复策略 |
| 策略包装 | `src/openpi/policies/defense_policy.py` | 运行时拦截 observation→action 接口 |
| 主干实现 | `src/openpi/models_pytorch/` | PyTorch 版 π₀ |
| 基线服务端 | `scripts/serve_policy.py`（:8000） | 未防护 π₀ |
| ActFovea 服务端 | 同脚本 + `--defense-enable`（:8003） | 开启防护 |
| 评测 | `scripts/eval_libero_task_matrix.py` | 四条件（干净 / 叠加 / 延迟 / 漂移 / 重放）任务矩阵 |
| 评测 harness | `examples/libero/main.py` | LIBERO 闭环 rollout |
| 批量脚本 | `scripts/run_runtime_baseline_*`、`scripts/run_patch_ablation_*` | 基线与消融批跑 |
| 单测 | `pytest` | 覆盖防护逻辑与评测组件 |

---

## 环境要求

- Ubuntu 22.04+、Python 3.11+
- NVIDIA GPU ≥ 12 GB VRAM，PyTorch + CUDA
- 依赖用 `uv` 管理：`uv sync --frozen`
- 需初始化 submodule（含 LIBERO 基准），并打一个 PyTorch transformers 补丁，再装 LIBERO 运行时依赖
- checkpoint 目录经 `OPENPI_CHECKPOINT_DIR` 指定

---

## 复现最短路径

1. `git submodule update --init --recursive` → `uv sync --frozen` → 打补丁 → 装 LIBERO。
2. 设 `OPENPI_CHECKPOINT_DIR`，起**两个**服务端：基线（8000）与 `--defense-enable`（8003）。
3. `scripts/eval_libero_task_matrix.py` 跑四条件矩阵，对齐论文 Table 1–3（40 任务 × 50 episodes = 2000 episodes / 组合）。

---

## 与仓库内实体的关系

| 关联 | 说明 |
|------|------|
| [paper-actfovea](../../wiki/entities/paper-actfovea.md) | 论文实体与方法/结论 |
| [libero-benchmark](../../wiki/entities/libero-benchmark.md) | 全部闭环评测所在基准 |
| [paper-pi05-open-world-vla](../../wiki/entities/paper-pi05-open-world-vla.md) | 同族 π 系列 VLA（本仓被防护的是 π₀） |
| [safety-filter](../../wiki/concepts/safety-filter.md) | 概念对照：ActFovea 是「感知侧一致性过滤」，不是控制侧 CBF/安全集 |
