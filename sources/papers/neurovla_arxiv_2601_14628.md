# A Brain-inspired Embodied Intelligence for Fluid and Fast Reflexive Robotics Control（NeuroVLA，arXiv:2601.14628）

> 来源归档（ingest）

- **标题：** A Brain-inspired Embodied Intelligence for Fluid and Fast Reflexive Robotics Control
- **缩写 / 框架：** **NeuroVLA**（Neuromorphic Vision-Language-Action）
- **类型：** paper / vla / neuromorphic / brain-inspired / safety-reflex
- **arXiv：** <https://arxiv.org/abs/2601.14628>（Submitted 2026-01-21）
- **代码（维护仓）：** <https://github.com/AlphaBrainGroup/AlphaBrain>（MIT）— 归档见 [`sources/repos/alphabrain.md`](../repos/alphabrain.md)
- **引用仓：** <https://github.com/guoweiyu/NeuroVLA>（README 指向 AlphaBrain 为可复现实现）
- **模型权重：** <https://huggingface.co/AlphaBrainGroup>（如 `neurovla-libero-all4suite`）
- **文档：** <https://alphabraingroup.github.io/AlphaBrain/>
- **作者：** Weiyu Guo, He Zhang, Pengteng Li, Tiefu Cai, Ziyang Chen, Yandong Guo, Xiao He, Yongkui Yang, Ying Sun, Hui Xiong
- **机构：** HKUST、SIAT、AI² Robotics 等（论文作者单位）
- **入库日期：** 2026-09-16
- **触发 ingest：** [深蓝具身智能三层控制架构长文](../blogs/wechat_shenlan_embodied_three_layer_control_2026-09-16.md) 以 NeuroVLA 为跨层案例
- **一句话说明：** 模仿皮层-小脑-脊髓组织：**Qwen-VL + 层-wise Q-Former** 规划语义意图，**自适应小脑** 用高频传感反馈稳定运动，**LIF 脉冲脊髓** 事件驱动生成动作；真机部署报告抖动抑制、~0.4 W 神经形态处理器功耗与 **<20 ms** 安全反射。

## 开源状态（步骤 2.5）

- **仓库核查（2026-09-16）：** [AlphaBrainGroup/AlphaBrain](https://github.com/AlphaBrainGroup/AlphaBrain) 提供 NeuroVLA 训练/评测脚本（`scripts/run_brain_inspired_scripts/`）、quickstart（`docs/quickstart/neurovla.md`）、部署 `server_policy.py` 与 HF 权重；[guoweiyu/NeuroVLA](https://github.com/guoweiyu/NeuroVLA) 为论文配套说明页，明确复现走 AlphaBrain。
- **结论：** **已开源**（框架 + 权重 + LIBERO 评测管线）；自研 FPGA 神经形态处理器为论文硬件，读者以仿真/标准 GPU 复现为主。

## 摘录 1：三层映射（Abstract + Method 归纳）

- **皮层：** 预训练 VLM（Qwen-VL 系）处理 RGB + 语言；**Q-Former** 从选定中间层蒸馏 **Semantic Latent Intention** \(z_{sem}\)。
- **小脑：** 自适应模块用高频本体/触觉反馈稳定运动（文内类比小脑协调）。
- **脊髓：** **SNN + LIF** 残差脉冲结构，膜电位跨步保留 → 隐式时序记忆；静止时神经元近静默，事件驱动省电。
- **涌现行为（论文声称）：** 臂抖抑制、碰撞 <20 ms 撤退反射、长时序任务优于无状态 MLP 头。

**对 wiki 的映射：** 升格 [`wiki/entities/paper-neurovla.md`](../../wiki/entities/paper-neurovla.md)；链 [`wiki/concepts/embodied-three-layer-control-architecture.md`](../../wiki/concepts/embodied-three-layer-control-architecture.md)。

## 摘录 2：AlphaBrain 工程入口（README 归纳）

| 组件 | 说明 |
|------|------|
| 骨干 VLM | `Qwen/Qwen2.5-VL-3B-Instruct`（HF 发布配置） |
| Q-Former | 层 36→37，`num_query_tokens=8`，`output_dim=768` |
| 动作头 | SNN + DiT 风格解码，`action_dim=7`，chunk 16 |
| 训练数据 | LIBERO 四套件混合 `libero_all` 等 |
| 评测 | `deployment/model_server/server_policy.py` + LIBERO eval |

**对 wiki 的映射：** 实体页 **源码运行时序图** 对齐 AlphaBrain README 入口。

## 摘录 3：与「三层架构」科普文的关系

[深蓝三层架构长文](../blogs/wechat_shenlan_embodied_three_layer_control_2026-09-16.md) 用 NeuroVLA 说明 **学习模型也可下沉到反射层**（脉冲脊髓），但强调反射仍须可预测；与经典 **纯阈值反射** 形成对照。

**对 wiki 的映射：** 概念页「工程实践」表引用本论文；勿把 NeuroVLA 当作所有产品的默认分层实现。
