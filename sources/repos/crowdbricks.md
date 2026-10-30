# CrowdBricks — 原始资料归档

- **来源**：https://github.com/patch-gg/crowdbricks
- **类型**：repo；Patch 的独立第三方 MotionBricks 批处理运行时
- **归档日期**：2026-10-30
- **核对版本**：[bf8f682](https://github.com/patch-gg/crowdbricks/tree/bf8f682affe1e2637ba64fbe333aa51cf772ee85)
- **核对范围**：仓库 README 全文；`examples/horde.py`、`crowdbricks/crowd.py`、`batch.py`、`gaits.py`、`obstacles.py` 与 `benchmarks/throughput.py` 的相关运行路径。
- **开放状态**：代码 Apache-2.0；需另行取得 [NVIDIA MotionBricks](https://github.com/NVlabs/motionbricks) 及资产。模型与样例动作分别适用 NVIDIA Open Model Agreement 和 NVIDIA Sample Data Evaluation License 等各自条款。

## 资料要点

- 为上游 SOMA ONNX 模型构建带 batch 维度的 TensorRT plan，并对编码、解码、端点修正等 Warp kernel 做批处理；每条请求有独立目标和 token 数量约束。
- `Crowd.step` 聚合临近转场、紧急或转向重规划请求；默认最多提前 0.15 秒，交互序列使用精确转场。GPU motion bank 保存生成的动作，采样结果返回 CPU。
- README 的 RTX 4090 FP16 测量：50 次中位数，48 请求 wall 4.66 ms，10,307 generations/s；上游单请求 wall 2.57 ms，389 generations/s。基准为站立到跑步，非普适帧率或 48 个角色各自每秒 1 万次。
- 仓库自述 batch 1、FP32 根位置误差 11 微米，关节达到 0.1 度比较精度；FP16 中离散 token 选择可能变化，不能理解成所有批量与精度均完全一致。

## 对 wiki 的映射

- [CrowdBricks](../../wiki/entities/crowdbricks.md)：机制、运行时序、性能口径与限制。
- [MotionBricks](../../wiki/entities/paper-motionbricks.md)：上游论文和模型。
