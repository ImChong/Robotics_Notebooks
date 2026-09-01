# ctu-vras/mobile-paradigm-model

> 来源归档（ingest · 代码仓库）

- **名称：** mobile-paradigm-model
- **组织：** [ctu-vras](https://github.com/ctu-vras)（CTU FEE，Cybernetics / VRAS）
- **URL：** <https://github.com/ctu-vras/mobile-paradigm-model>
- **许可证：** 仓库未在 README 明确 SPDX；以 GitHub 页面为准
- **关联论文：** [arXiv:2504.17939](https://arxiv.org/abs/2504.17939) — *A computational model of infant sensorimotor exploration in the mobile paradigm*（IEEE TCDS 2026）；与 [Science Robotics 2026 真机论文](../papers/robot_in_crib_sensorimotor_contingency_scirobotics_2026.md) 同课题组、同范式，**仿真专用**。
- **开放程度：** **已开源**（仿真计算模型；非 iCub 真机 YARP 栈）
- **入库日期：** 2026-09-01

## 仓库结构

| 路径 | 说明 |
|------|------|
| `main.py` | 主入口；`runExperiment(name, params)` 配置实验 |
| `README.md` | 依赖与参数说明 |

## 运行要点（README）

- Python **3.9.15**；PyTorch。
- 示例：`runExperiment("standard", [1,1,1,True,600,0.3,0.3,False,0.1])`
  - 参数依次为：fatigue / curiosity / prediction 权重、隐层、运动指令数、baseline、噪声、均匀分布、新奇阈值。
- CLI：`python main.py --name myOwnExperiment`（需开启文件内 console 参数开关）。

## 与 Science Robotics 真机论文边界

- 本仓复现 **虚拟肢体 mobile paradigm** 与 ablation；**不包含** iCub 真机摇篮装置、Gazebo 七条件批量实验、MLP 外部观察者。
- 读 Science Robotics 论文时：**理论机制（预测+好奇心）可在此仓试验**；**四策略行为光谱与 MLP 指标需读正式论文**。

## 关联

- 论文（仿真）：arXiv:2504.17939
- 论文（真机）：[`robot_in_crib_sensorimotor_contingency_scirobotics_2026.md`](../papers/robot_in_crib_sensorimotor_contingency_scirobotics_2026.md)
- Wiki：[`wiki/entities/paper-robot-in-crib-sensorimotor-contingency.md`](../../wiki/entities/paper-robot-in-crib-sensorimotor-contingency.md)
