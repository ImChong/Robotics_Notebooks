# New NVIDIA Cosmos 3 Open Models for Interactive World Simulation（Hugging Face 博文）

> 来源归档

- **类型：** 厂商官方博客（Hugging Face Blog，nvidia 组织）
- **标题：** New NVIDIA Cosmos 3 Open Models for Interactive World Simulation
- **链接：** <https://huggingface.co/blog/nvidia/cosmos-interactive-world-simulation>
- **发布日期：** 2026-11-05（页面 "Article Published November 5, 2026"）
- **作者（页面署名，均 nvidia）：** Pranjali Joshi、Rick Z、Yao Xu、Qianli Ma、Chen-Hsuan Lin、Ming-Yu Liu、Xiang Li、Zhengfei Kuang、Zhonghao Wang、Jiaxiang Tang
- **核查日期：** 2026-11-06（curl 取 HTML，用 `python3 -I` 去标签通读全文；HTTP 200）
- **口径提醒：** 全文属厂商自述；博文本身没有给出新的定量表，数字以模型卡 / 技术报告为准（见 [模型卡归档](../sites/hf_cosmos3_nano_sim_checkpoints_model_cards.md)）。
- **对应 wiki：** [Cosmos Interactive World Simulation](../../wiki/entities/paper-cosmos-interactive-world-simulation.md)

## 博文结构与要点（逐节）

1. **TL;DR：** Cosmos 3 家族新增四个开放权重 checkpoint——Transfer-Auto（多视角驾驶数据，变天气 / 光照 / 时段）、Sim-Auto（流式相机 + LiDAR 观测，驾驶仿真）、Sim-Transfer（仿真器控制 + 文本 → 照片级视频）、Sim-Bimanual（机器人或相机动作 → 视频）。四者均为 Cosmos 3 Nano 的后训练示例、16B 可训练参数、OpenMDW 1.1（允许商用与非商用，受许可条款约束）。
2. **Why：** 机器人需要能在变化场景与视角下回答“动作的后果”的环境；Sim-Bimanual 由动作预测视觉后果，Sim-Transfer 把物理仿真器的控制变成生成观测。驾驶需要外观与几何一致，相机与 LiDAR 同步生成；场景与信号灯控制可改路网与信号状态。作者承认问题仍难：下一步要度量模型在策略学习与评估上的忠实程度并认清局限。
3. **Transfer-Auto：** 双向（bidirectional）离线合成数据模型；最多 11 路相机（含鱼眼），支持 RGB–LiDAR 联合生成；每相机的 World Scenario Map（WSM）控制场景结构与交通运动，文本控制外观；可选首帧 RGB 锚定外观；联合 LiDAR 另需对齐的 HD-map range-map 控制。也是适配自有传感器配置的起点：先后训练它，再做因果训练 + 少步蒸馏得到流式模型（即 Sim-Auto 的开发路径）。
4. **Sim-Auto：** Transfer-Auto 的因果蒸馏流式变体；以少量去噪步生成相机视频与 LiDAR 的连续小块，使用生成历史与更新后的驾驶控制。闭环交互需要自备仿真运行时与 WSM 渲染器来按驾驶输入更新控制。
5. **Sim-Transfer：** 仿真器负责运动、接触与物理状态；模型用其空间控制（深度、分割、边缘、模糊）+ 文本渲染策略可读的彩色观测。循环：动作 → 仿真器更新 → 空间控制 → 生成 RGB → 下一步策略动作；物理与真值状态留在仿真器。
6. **Sim-Bimanual：** 模拟双臂机器人动作下环境的演化；**博文称支持 YAM 与 AgiBot 两种本体**，可由策略或人用摇杆操控；一条控制序列可在视角变化与操作间切换。混合动作示例为 59 个控制值（训练时补零到 64）；T 个转移需要 T+1 个观测；训练把原始采集转成模型动作表示，已发布推理动作已归一化。要求：Linux、NVIDIA GPU、Docker、NVIDIA Container Toolkit；已发布 vLLM-Omni 配置在单张 GB200 上测试，“并非最小显存规格”。
7. **流式生成原理：** MoT 主干；每个短片段由提示词、可用控制与已生成历史预测；时间因果注意力读历史与控制、屏蔽未来目标；KV cache 保留注意力信息。潜变量在首帧之后每个时间步对应 4 帧 RGB；Transfer 每块预测 1 个潜变量时间步，技术报告里优化后的 Bimanual 配置每块预测 2 个。
8. **四阶段训练配方：** Stage 0 双向适配；Stage 1 teacher forcing（以先前录制观测预测每块，遮掉干净答案与未来目标）；Stage 2 一致性蒸馏（冻结教师训练学生，每块 4 个去噪步）；Stage 3 self-forcing（在学生自身生成的历史上训练；冻结教师 + 辅助 score 模型提供分布匹配信号）。Stage 3 的理由：小误差会作为后续帧上下文累积，需让训练暴露于生成历史。
9. **自适应：** 后训练配方起于 Cosmos3-Nano 或兼容基座，覆盖全部四阶段；保留完整训练 checkpoint 以便恢复（导出的推理权重不含优化器状态）；建议在留出 episode 上评估，“短跑成功只证明执行通，不证明收敛”。
10. **更多资源：** 检查点集合 <https://huggingface.co/collections/nvidia/cosmos3-examples>、技术报告 PDF、项目页 demo、Cosmos Labs 直播回放、NVIDIA Developer 技术博客 <https://developer.nvidia.com/blog/build-world-models-for-interactive-simulation-with-nvidia-cosmos/>。

## 博文列出的入口（均指向 NVIDIA/cosmos-framework 的 `.agents/skills/`）

| checkpoint | 推理 skill | notebook | 后训练 skill |
|------------|-----------|----------|--------------|
| Transfer-Auto | `cosmos3-nano-transfer-auto/inference/SKILL.md` | `cookbooks/cosmos3/simulation/auto/run_auto_with_cosmos_framework.ipynb` | `cosmos3-nano-transfer-auto/post-training` |
| Sim-Auto | `cosmos3-nano-sim-auto/inference/SKILL.md` | `.../auto/run_sim-auto_with_cosmos_framework.ipynb` | `cosmos3-nano-sim-auto/post-training` |
| Sim-Transfer | `cosmos3-nano-sim-transfer/inference/SKILL.md` | `.../transfer/run_transfer_with_vllm_omni.ipynb` | `cosmos3-nano-sim-transfer/post-training/SKILL.md` |
| Sim-Bimanual | `cosmos3-nano-sim-bimanual/inference/SKILL.md` | `.../bimanual/run_bimanual_with_vllm_omni.ipynb` | `cosmos3-nano-sim-bimanual/post-training/SKILL.md` |

## 与技术报告归档的出入

- 博文把 **Transfer-Auto 与 Sim-Auto 写成两个不同 checkpoint**（双向离线 vs 因果蒸馏流式）；[技术报告归档](../papers/cosmos_interactive_world_simulation_tech_report_2026.md) 早先把二者当成“同一驾驶方向的两种命名”，已在该归档与 wiki 页更正。
- 博文称 Sim-Bimanual 支持 **YAM 与 AgiBot**；模型卡 Inputs 一栏只写 "Supported embodiment: AgiBot"，仓库文档与代码却含 YAM（详见模型卡归档）。
- 博文未列任何基准数字；Sim-Auto 被称为可“快速流式”，但模型卡吞吐表在多视角 + 720p 下低于 30 fps 播放帧率（见 wiki 页）。
