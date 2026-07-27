# minecraft-zzz/KineBench

> 来源归档

- **标题：** KineBench（官方实现）
- **类型：** repo
- **组织 / 作者：** minecraft-zzz（TeleAI 等相关作者线）
- **代码：** <https://github.com/minecraft-zzz/KineBench>
- **许可证：** **MIT**
- **论文：** <https://arxiv.org/abs/2607.19876>
- **入库日期：** 2026-07-27
- **一句话说明：** **IDM-free** 具身世界模型闭环评测：视频 → YOLO/MoGe/FoundationPose 抽 6D EEF → pyroki 规划/可控动作 → **ManiSkill3** 执行；提供 `local_smoke` 与 DashScope Wan 生成两条路径。

## 开源核查（2026-07-27）

- GitHub API：`license.spdx_id = MIT`；默认分支 `main`。
- 顶层 README 仅标题；可运行性以包内脚本与 `examples/README.md` 为准。
- **已开源** 可跑通管道（smoke）与完整评测配置骨架；权重/CAD 需自备。

## 入口速查（对齐仓库树）

| 路径 / 命令 | 作用 |
|-------------|------|
| `kinebench/` | 公共 API：`KineBenchEvaluator`、`load_config` |
| `kinebench/perception/` | YOLO 掩码、MoGe 深度、FoundationPose 6D |
| `kinebench/planning/` | `extractor`、`pyroki`、gripper / transforms |
| `kinebench/envs/runtime.py` | ManiSkill3 运行时（`pd_ee_pose`） |
| `kinebench/generation/` | `local` npy 视频 / DashScope Wan I2V |
| `configs/eval/local_smoke.yaml` | 合成视频管道冒烟 |
| `configs/eval/maniskill_wan26.yaml` | Wan2.6-i2v + 全感知/规划路径 |
| `python scripts/run_eval.py --config …` | 主评测入口 |
| `scripts/prepare_third_party.py` | 拉齐 FoundationPose / pyroki 等第三方 |
| `scripts/analyze_results.py` | 结果汇总 |

## 最短复现路径

1. 安装依赖并准备 ManiSkill3 / third_party（见 `prepare_third_party.py`）。
2. 按 `examples/README.md` 生成 `examples/local_video.npy`。
3. `python scripts/run_eval.py --config configs/eval/local_smoke.yaml`。
4. 完整评测：填写 `maniskill_wan26.yaml` 中 CAD / YOLO / MoGe / FoundationPose / pyroki 路径后跑 Wan 生成链路。

## 与本仓库知识的关系

| 主题 | 关系 |
|------|------|
| [KineBench](../../wiki/entities/paper-kinebench.md) | 实体归纳：IDM-free 接地、四套件、SPARC/Manipulability |
| [EWMBench](../../wiki/entities/ewmbench.md) | 同属具身 WM 评测；KineBench 强调执行闭环而非仅像素守恒 |
| [物理保真度输出轴](../../wiki/overview/world-model-physics-fidelity-outputs.md) | 「可执行性」测试优先序的工程落地 |
| [Masked Visual Actions](../../wiki/entities/paper-masked-visual-actions.md) | 对照：仍依赖 IDM 从合成视频抽低维动作 |

## 对 wiki 的映射

- 论文摘录：[`sources/papers/kinebench_arxiv_2607_19876.md`](../papers/kinebench_arxiv_2607_19876.md)
- 沉淀 **[`wiki/entities/paper-kinebench.md`](../../wiki/entities/paper-kinebench.md)**
