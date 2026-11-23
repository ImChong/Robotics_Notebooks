---
type: entity
tags: [repo, china-embodied-opensource, open-source, project]
status: complete
updated: 2026-11-21
code: https://github.com/sharinka0715/X-WAM
project: https://sharinka0715.github.io/X-WAM/
original:
  - ../../sources/repos/x-wam.md
related:
  - ../overview/china-domestic-embodied-opensource-76-companies-technology-map.md
  - ../entities/humanoid-motion-intelligence.md
  - ../queries/china-domestic-opensource-424-coverage.md
  - ./paper-rcl-2604-26694-unified-4d-world-action-modeling-from-video-prio.md
sources:
  - ../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md
  - ../../sources/repos/x-wam.md
summary: "小米集团开源项目 X-WAM（sharinka0715/X-WAM）：基于 Wan2.2-TI2V-5B 的统一 4D 世界-动作模型，通过轻量深度适配分支与异步噪声采样联合生成未来 RGB-D 视频与关节动作。"
institutions:
  - xiaomi-robotics
---

# X-WAM

## 一句话定义

**X-WAM** 是 [小米集团](https://github.com/XiaomiRobotics) 公开的 **世界模型** 开源项目：模型联合学习视频世界变化与机器人动作，在共享表征中支持跨本体操作和未来预测；项目用于检查世界模型输出怎样与动作头连接。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| SDK | Software Development Kit | 真机控制与状态读取接口 |
| RL | Reinforcement Learning | 强化学习训练与策略优化 |
| VLA | Vision-Language-Action | 视觉–语言–动作统一策略 |
| Sim2Real | Simulation to Real | 仿真策略迁移真机 |
| URDF | Unified Robot Description Format | 机器人描述与仿真资产 |

## 为什么重要

- 收录于 [国内具身智能开源全景（76 家 · 424 项）](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md) 的 **第一层** 分组。
- 与 [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md) 同源策展；本页为 **独立详情节点**，便于从公司清单跳到机制与入口说明。

## 核心原理

`X-WAM`（Unified 4D World-Action Modeling，`arXiv:2604.26694`，仓库 `sharinka0715/X-WAM`）在预训练视频扩散主干 `Wan2.2-TI2V-5B` 上以**单阶段端到端联合去噪**同时建模未来 RGB 视频、未来深度图（4D 几何表征）与机器人关节动作轨迹。

```mermaid
flowchart TB
  Obs["当前视角观测 I_0 + 深度 D_0 + 文本指令 + 机器人状态"] --> Tok["Wan2.2 VAE / UMT5-XXL 编码"]
  Tok --> RGB["RGB Latent Branch (Wan2.2-TI2V-5B)"]
  Tok --> Dep["Depth Branch (LDA 轻量深度适配 + 交错自注意力)"]
  RGB <--> Dep
  RGB --> Async["Asynchronous Noise Sampling (视频 K_v=50 步 vs 动作 K_a=10 步)"]
  Async --> Act["Action Branch (DiT Action Expert / MLP Head)"]
  Act --> Out["预测 4D 未来帧 (RGB+Depth) + 连续动作块"]
```

- **轻量深度适配（LDA, Lightweight Depth Adaptation）**：保留预训练 RGB 分支权重的跨通道对齐，在相邻层之间插入交错式自注意力与轻量深度通路，避免从头训练破坏先验视频生成质量。
- **异步噪声采样（Asynchronous Noise Sampling）**：为动作序列单独分配独立且更密/更快的扩散时间步调度（默认视频 50 步、动作 10 步），在单主干联合优化下兼顾视频预测质量与高频动作控制延迟。

## 工程实践

- **训练入口**：`scripts/train_sft.py` 通过 PyTorch Lightning + DeepSpeed 调度 `UnifiedWorldModel`（默认 `Wan2.2-TI2V-5B`，`bf16-mixed` 精度），支持 `robotwin` 与 `robocasa` 两套配置文件（`configs/`）。
- **推理与异步评估**：`evaluation/` 采用 `policy_server.py`（模型加载与推理）+ `policy_broker.py`（请求分发）+ `robocasa_client.py` / RoboTwin 客户端架构，隔离仿真环境依赖与 GPU 推理节点。
- **预训练权重与评测基准**：官方 Hugging Face `sharinka0715/X-WAM-checkpoints` 提供微调权重；论文报告 RoboCasaGR1Tabletop 24 任务平均成功率 **79.2%**，RoboTwin 2.0 清洁/随机化设定分别为 **89.8% / 90.7%**。

## 局限与风险

- **长时程物理一致性依赖视频先验**：闭环高频干扰下若视频分支预测偏移，跨模态注意力仍可能将伪影传导至动作预测；此外全量 5B DiT 训练与推理依赖多卡 A100/H100 级显存。
- **真机与仓库范围边界**：当前仓内主要提供 RoboTwin 2.0 / RoboCasa 仿真训练与评测脚本及对应 checkpoints，跨本体真机硬件控制驱动需按本体适配，并应结合 [X-WAM 论文页](./paper-rcl-2604-26694-unified-4d-world-action-modeling-from-video-prio.md) 核对最新消融数据。

## 源码运行时序图

入口（评测为三个进程）：`python evaluation/policy_broker.py --frontend_port 10086 --backend_port 10087`；`CUDA_VISIBLE_DEVICES=0 python evaluation/policy_server.py --exp_path checkpoints/robocasa_sft --wan_checkpoint_dir <wan22_5b 路径> --broker_port 10087 --denoise_steps 50 --action_denoise_steps 10`；环境端 `python evaluation/robocasa_client.py`（参数见 `evaluation/README.md`）。

```mermaid
sequenceDiagram
    autonumber
    participant PS as evaluation/policy_server.py 策略服务端
    participant M as runners/xwam_runner.py XWAMRunner
    participant B as evaluation/policy_broker.py 中介
    participant C as evaluation/robocasa_client.py 环境端
    participant RS as robosuite 与 robocasa 仿真环境

    PS->>M: XWAMRunner(config) 加载模型并载入 checkpoint
    PS->>B: DEALER 连接 backend 并发送 READY
    B->>B: 登记空闲服务端 available_servers
    C->>RS: robosuite.make 创建环境并 env.reset
    loop 每个 rollout 的每一步 直到成功或达到 TASK_MAX_STEPS
        C->>RS: render_obs 取多相机 RGB 与末端状态
        C->>B: socket.send 发送 pickle 的 data_batch 含 video proprios prompt cfg
        B->>B: 放入 pending_requests 并与空闲服务端 1:1 分派
        B->>PS: WORK client_id data_pkl
        PS->>PS: resize 与中心裁剪视频 归一化 proprio 按 step 推导 seed
        PS->>M: model.generate rgb proprio prompt seeds cfg
        M-->>PS: xt_actions 与 xt_proprios
        PS->>PS: 反归一化得到 actions 与 proprios
        PS->>B: RESULT client_id result_pkl
        PS->>B: READY 重新登记
        B->>C: 按 client_id 转发 result_pkl
        C->>C: 取前 action_length 步 逐步组装 pad_action
        loop 每个动作步
            C->>RS: env.step pad_action
            C->>RS: _check_success 判断是否成功
        end
    end
```

图依据：sharinka0715/X-WAM@72cfb86，实际读过 `README.md`、`evaluation/README.md`（前 80 行，启动命令段）、`evaluation/policy_broker.py`、`evaluation/policy_server.py`、`evaluation/robocasa_client.py`（`main` 与 `render_obs` 相关段）。`runners/xwam_runner.py` 内部未读，`XWAMRunner.generate` 仅据 `policy_server.py` 中的调用名标注；RoboTwin 客户端与训练流程未画。未运行代码。

## 关联页面

- [X-WAM 论文页（Unified 4D World-Action Modeling）](./paper-rcl-2604-26694-unified-4d-world-action-modeling-from-video-prio.md)
- [国内具身开源全景技术地图](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md)
- [HMI 开源项目主表导读](../queries/hmi-opensource-projects-coverage.md)
- [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md)

## 参考来源

- [X-WAM 源码归档](../../sources/repos/x-wam.md)（`sharinka0715/X-WAM@72cfb86`，<https://github.com/sharinka0715/X-WAM>）

## 推荐继续阅读

- <https://github.com/sharinka0715/X-WAM>
- <https://sharinka0715.github.io/X-WAM/>

- [国内具身智能开源全景（微信公众号）](../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)
- 源码核对（2026-11-02）：sharinka0715/X-WAM@72cfb86，读过 README.md、evaluation/README.md（前 80 行）、evaluation/policy_broker.py、evaluation/policy_server.py、evaluation/robocasa_client.py；时序图依据以上文件，未运行代码。

## 推荐继续阅读

- [小米集团 官方组织](https://github.com/XiaomiRobotics)
