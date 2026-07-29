# SoftVTBench（TuojingAI/SoftVTBench）

> 来源归档

- **标题：** SoftVTBench — Safety-Aware Visuo-Tactile Benchmark
- **类型：** repo / benchmark / Isaac Lab / OpenPI
- **组织：** 拓境智能等（见论文作者）
- **代码：** <https://github.com/TuojingAI/SoftVTBench>
- **项目页：** <https://softvtbench.github.io/>
- **论文：** <https://arxiv.org/abs/2607.04234>
- **数据集：** <https://huggingface.co/datasets/Arthur12137/SoftVTBench> · <https://www.modelscope.cn/datasets/Arthur12137/SoftVTBench>
- **License：** Apache-2.0
- **入库日期：** 2026-07-29
- **一句话说明：** SoftVTBench 官方实现：Isaac Sim/Lab 视触觉仿真扩展、π₀.₅ 训练转换管线、闭环评测脚本与安全阈值配置；数据与 USD 外置 HF/ModelScope。
- **沉淀到 wiki：** [SoftVTBench（论文实体）](../../wiki/entities/paper-softvtbench.md)

## 开源状态（核查，2026-07-29）

| 资产 | 状态 |
|------|------|
| 训练 / 闭环评测代码 | **已开源**（`openpi/scripts/train_softvtbench.sh`、`evaluate_softvtbench.sh`） |
| Isaac Lab 扩展 | **已开源**（`SoftVTBench/source/tac_manip`） |
| 演示数据 + eval USD | **已开源**（HF / ModelScope；约 1,628 episodes） |
| Franka/GelSight 运行时资产 | **上游开源**（`china-sae-robotics/Tactile_Manipulation_Dataset`） |
| SoftVTBench 参考 checkpoint | **计划外发**（README：Planned） |
| License | Apache-2.0 |

## 仓库导航（对齐时序图节点）

| 路径 | 作用 |
|------|------|
| `environment.yml` / `requirements.txt` | `softvtbench-eval`（Py3.10 + Isaac Sim 4.5） |
| `openpi/upstream/` + `uv.lock` | `softvtbench-openpi`（Py3.11，π₀.₅ 训练/服务） |
| `tools/doctor.py` | 训练/评测前只读预检 |
| `openpi/scripts/train_softvtbench.sh` | convert → stats → LoRA train（`SUITE` / `MODALITY` / `PHASE`） |
| `openpi/scripts/evaluate_softvtbench.sh` | 闭环评测；软体套件出 Goal / Safety / 形变摘要 |
| `configs/safety_thresholds.json` | 物体特异形变安全阈值 |
| `SoftVTBench/source/tac_manip/` | Isaac Lab 扩展：GelSight/TacEx、任务、录制 |
| `SoftVTBench/benchmarks/common/metrics.py` | Goal / Safety 等评测指标 |
| `SoftVTBench/benchmarks/common/closedloop_policy_inference.py` | 闭环策略推理 |

## 最短复现路径（README）

1. 双环境：`conda` → `softvtbench-eval`；`uv sync` → OpenPI venv。
2. `hf download Arthur12137/SoftVTBench` + 触觉运行时资产 symlink。
3. `tools/doctor.py --mode train|eval …`
4. `SUITE=object-soft MODALITY=tactile PHASE=all bash openpi/scripts/train_softvtbench.sh`
5. `bash openpi/scripts/evaluate_softvtbench.sh`（默认每任务 N=50）

## 与本仓库知识的关系

| 主题 | 关系 |
|------|------|
| [视触觉融合](../../wiki/concepts/visuo-tactile-fusion.md) | VO vs VT 在软体安全约束下的消融 |
| [Tactile Sensing](../../wiki/concepts/tactile-sensing.md) | GelSight Mini RGB + marker motion |
| [接触丰富操作](../../wiki/concepts/contact-rich-manipulation.md) | 过程级物理安全，而非终端位姿成功 |
| [TacO](../../wiki/entities/paper-taco-tactile-sensor-benchmark.md) | 互补：传感器硬件选型 vs 形变安全指标 |
| [VLA / π₀.₅](../../wiki/methods/vla.md) | 基线为 OpenPI π₀.₅ LoRA |

## 为何值得保留

- 把「可变形操作成功」拆成 **Goal / Safety** 可复现协议，可直接服务评测选型。
- 模块边界清晰，适合写 wiki **源码运行时序图** 与双环境复现 checklist。
