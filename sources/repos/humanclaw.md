# HumanCLAW GitHub 仓库

> 来源归档（ingest）

- **项目名称：** HumanCLAW
- **GitHub 地址：** <https://github.com/Human-CLAW/HumanCLAW>
- **许可证：** Apache 2.0
- **核心功能：** HumanCLAW-Bench 评测 harness：冻结 VLM 闭环决策 + DiT 全身技能运动生成 + Habitat Half-Physics 执行；含 metric / video / motion 训练工具。
- **入库日期：** 2026-09-18

## 仓库结构（README / docs/ARCHITECTURE.md 对齐）

| 路径 | 作用 |
|------|------|
| `src/humanclaw_bench/main.py` | CLI：`humanclaw-bench run/render/...` |
| `src/humanclaw_bench/evaluation/evaluator.py` | 单回合 rollout 主循环 |
| `src/humanclaw_bench/agent/planner.py` | PSVEgoAgent：planner + optional verifier |
| `src/humanclaw_bench/motion/runner.py` | MotionSkillRunner：DiT + per-skill ControlNet |
| `src/humanclaw_bench/envs/find_nav_interact_env.py` | find–navigate–interact 任务语义 |
| `src/humanclaw_bench/envs/half_physics/` | Half-Physics Bullet 控制器 |
| `src/humanclaw_bench/vlm/` | OpenAI-compatible / queue 模型适配 |
| `src/humanclaw_bench/evaluation/metrics/` | FindSR / NavSR / InteractSR 等指标 |
| `resources/benchmark/` | 固定 1,218-episode split 与 val100 索引 |
| `resources/hssd/` | HumanClaw 场景与物体配置 |
| `configs/models/` | VLM 接口 JSON 模板 |
| `patches/habitat-sim/` | 必需 Habitat-Sim patch |

## 关键复现路径

1. Python 3.10+ venv + CUDA PyTorch + patched Habitat-Sim（Bullet）
2. 授权 HSSD-Hab val + HF motion weights +（可选）gated HSSD supplement
3. `cp configs/models/vllm_openai_compatible.json my_model.json` 并填 VLM endpoint
4. Smoke：`humanclaw-bench run --episodes one --model-config my_model.json --gpus auto --output outputs/smoke`
5. 子集 / 全量：`--episodes val100` 或 `--episodes fullval`；加 `--metrics` / `--video` 按需

## 关联 Wiki 页面

- [HumanCLAW 论文实体](../../wiki/entities/paper-humanclaw.md)
- [HumanCLAW 项目页](../sites/human-claw-github-io.md)
- [VLA](../../wiki/methods/vla.md)
