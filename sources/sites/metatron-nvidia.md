# METATRON：官方项目页与 Hugging Face 模型卡归档

- **类型：** NVIDIA 官方项目页 + Hugging Face 模型卡
- **项目页：** <https://research.nvidia.com/labs/par/projects/metatron/>
- **论文：** <https://arxiv.org/abs/2610.38027> （v1，2026-10-27）
- **框架代码：** <https://github.com/NVlabs/ProtoMotions>
- **模型：** <https://huggingface.co/nvidia/Metatron-v1>
- **官方运行文档：** <https://nvlabs.github.io/ProtoMotions/getting_started/pretrained_models.html>
- **核查日期：** 2026-10-31（2026-11-14 完成全文与模型卡复核）
- **通读：** 已完成
- **核对范围：** NVIDIA METATRON 官方项目页内容、`nvidia/Metatron-v1` 模型卡全文（`README.md`，166 行）以及 `NVlabs/ProtoMotions` 中 `demos/metatron/README.md` 浏览器演示说明。

## 已核查的项目页与演示口径

- 先以完整参考动作训练跟踪器，再通过掩码目标的 on-policy 蒸馏得到生成式基础策略；部署时冻结权重，有状态技能代码逐周期构造结构化运动 prompt，策略输出关节目标。**提示是结构化运动目标，不是运行时自然语言指令。**
- 编码智能体在离线仿真中编写并迭代技能代码；运行时仅由普通确定性技能程序根据状态更新 prompt，LLM 不处于控制回路中。
- G1 的跨栏、跌倒恢复、保龄球、网球与踢球展示属于 MuJoCo 仿真例子（跨栏位置由任务代码给出，保龄球由场景代码定时释放，网球为定时落球而非视觉追踪），**不能直接视作真机统计成功率**；SOMA 演示使用独立策略。
- 实机 Unitree G1 转向与抗推步态演示证明了同一冻结权重可跨技能复用，但项目页部分演示的步态美观度与稳定性改进行为属于定性展示。

## Hugging Face 模型卡核查（`nvidia/Metatron-v1`）

### 1. 许可分离与数据边界

- **模型权重与导出制品：** 受 **NVIDIA Open Model License** 约束（具体条款见模型仓库 `LICENSE.pdf`）。
- **框架代码：** `NVlabs/ProtoMotions` 源码采用 **Apache-2.0** 许可。
- **数据集边界：** 模型卡与代码仓库均**不随包分发原始训练动作片段数据集**（MoCap / SAMP 重定向动作文件）。

### 2. 公开发布的三类推理制品

| 制品路径 | 规模/规格 | 运行宿主与用途 | 兼容性注意 |
|---|---|---|---|
| `G1/inference_last.ckpt` | 459,455,983 字节（epoch-6410） | 原生 PyTorch 检查点（29 关节 Unitree G1，含 SAMP 交互能力学生策略），用于 ProtoMotions MuJoCo/Isaac 推理与导出 | 仅包含 G1 检查点；**未公开发布 SOMA 预训练权重** |
| `demo/model.onnx` | 单机器人 ONNX（FP32） | 浏览器 WASM/WebGPU 演示导出件，39 路输入合同（含 8 步历史状态、9 步原始动作历史、`scene_boxes` 与 `recovery_enabled`） | 仅支持单环境实时推理 |
| `racer/network.onnx` | 6 机器人批量 ONNX | 面向多机器人并行竞速演示导出 | batch 与输入契约不同于 `demo/model.onnx`，**二者不可直接互换** |

### 3. 浏览器演示源码状态

- `NVlabs/ProtoMotions` 的 `demos/metatron/` 仅随仓库提供预编译演示包 `demo.zip` 与资源清单 `bundle.json`，用于部署或本地试玩浏览器交互演示（运行期从 Hugging Face 匿名拉取外部 FP32 权重）；**TypeScript 前端演示工程源码未开源**。
