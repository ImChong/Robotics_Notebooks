# SAGE（Sim2Real Actuator Gap Estimator）

> 来源归档（仓库 README 与公开文档要点摘录，非全文镜像）

- **标题：** SAGE: Sim2Real Actuator Gap Estimator
- **类型：** repo
- **组织：** isaac-sim2real（GitHub）
- **链接：** https://github.com/isaac-sim2real/sage
- **许可：** Apache-2.0
- **入库日期：** 2026-05-17
- **一句话说明：** 在 Isaac Sim / Isaac Lab 中重放关节轨迹并与真机日志对齐，用多指标统计与可视化量化「仿真–真机」执行器层差距，并导出成对数据供 gap 补偿类模型训练。
- **沉淀到 wiki：** [wiki/entities/sage-sim2real-actuator-gap-estimator.md](../../wiki/entities/sage-sim2real-actuator-gap-estimator.md)

---

## 依赖与运行面（README 声明）

- Ubuntu 22.04 LTS，Linux x86_64，NVIDIA GPU
- Python 3.10；Isaac Sim **5.0.0**；Isaac Lab **2.2.0**
- 可选 Docker 镜像（仓库内 `Dockerfile`），避免本机逐项安装

---

## 能力边界（作者在 README 中的定位）

1. **仿真重放**：`scripts/run_simulation.py` 在 Isaac 中按 motion file 重放，写出 `output/sim/...`（控制指令与电机状态等 CSV）。
2. **真机采集**：`scripts/run_real.py` 统一入口，厂商差异下沉到各子文档（Unitree G1/H1-2、Realman WR75S、LeRobot SO-101 等）。
3. **对比分析**：`scripts/run_analysis.py` 生成关节级 RMSE、MAPE、相关系数、余弦相似度等，并出图（位置 / 速度 / 力矩等）。
4. **批处理与数据发布**：提供 **OSMO** 工作流（`osmo_workflow.yaml`）一键提交仿真+分析；处理后 sim–real 成对序列可进一步整理为 **NPZ** 等训练格式（README 描述「gap compensation models」用途）。

---

## 数据与配置契约（摘要）

| 环节 | 约定路径 / 要点 |
|------|----------------|
| Motion 输入 | `motion_files/{robot}/{source}/`；首行为关节名，后续行为弧度角 |
| 仿真输出 | `output/sim/{robot}/{source}/{motion}/`：`control.csv`、`state_motor.csv`、`joint_list.txt` |
| 真机输出 | `output/real/...`：除关节状态外可有 `state_base.csv`、`event.csv`；时间戳单位为 **微秒**（与仿真秒级不同） |
| 控制器对齐 | `sage/assets.py` 中 per-robot `default_kp`、`default_kd`、`default_control_freq` 应与真机采集时一致，否则 README 明确会 **夸大测得 gap** |
| 参数优先级 | CLI > `assets.py` 默认值；二者皆缺则报错 |

**README 列出的运动来源示例：** AMASS 重定向轨迹（并指向 [AMASS 数据集](https://amass.is.tue.mpg.de/) 与 [Human2Humanoid](https://github.com/LeCAR-Lab/human2humanoid?tab=readme-ov-file#motion-retargeting) 作为重定向参考）。

---

## 新平台接入（README 结构提示）

- **新人形（仿真）**：USD 资产、`configs/*_joints.yaml`、`*_valid_joints.txt`，并扩展 `joint_motion_gap/simulation.py` 等（详见仓库 `docs/NEW_ROBOT.md`）。
- **新真机**：ROS 传输层模板 + 仿 `unitree_configs.py` 的配置结构，并在 `scripts/run_real.py` 注册任务路径。

---

## 公开数据集线索（README）

- 处理后 Unitree / RealMan 成对数据：北大网盘链接（README 内；下载与条款以网盘页为准）。
- Unitree 侧：AMASS 上身动作、0–3 kg 负载变体、步态组合与部分全身扩展等数据划分说明。
- RealMan：四臂 × 四负载，用于跨机台泛化实验的叙事。

---

## 引用（仓库建议 BibTeX）

```bibtex
@misc{sage-2025,
  title={SAGE: Sim2Real Actuator Gap Estimator},
  author={SAGE Team},
  year={2025},
  url={https://github.com/isaac-sim2real/sage}
}
```

---

## 与本仓库其他资料的关系

| 资料 | 关系 |
|------|------|
| [isaac_gym_isaac_lab.md](isaac_gym_isaac_lab.md) | SAGE 管线绑定 Isaac Sim / Isaac Lab 版本 |
| [wiki/entities/amass.md](../../wiki/entities/amass.md) | 常见上游人体运动档案；SAGE 以 AMASS 重定向为示例 motion source |
| [wiki/concepts/sim2real.md](../../wiki/concepts/sim2real.md) | Sim2Real 总览；执行器层 gap 属 domain gap 子问题 |
| [wiki/methods/actuator-network.md](../../wiki/methods/actuator-network.md) | 数据驱动执行器建模；SAGE 产出可用于标定或学习残差前的定量画像 |
