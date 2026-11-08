# Dreamscale Labs — 官方平台与 SDK 技术文档核查

- **类型：** 官方项目页 / 开发文档 / 商业云推理平台
- **主站：** https://www.dreamscalelabs.com/
- **文档索引：** https://docs.dreamscalelabs.com/llms.txt
- **Quick start：** https://docs.dreamscalelabs.com/
- **模型接口：** https://docs.dreamscalelabs.com/models
- **Python SDK：** https://docs.dreamscalelabs.com/python
- **SO-101 CLI：** https://docs.dreamscalelabs.com/robots/so101
- **双臂 YAM CLI：** https://docs.dreamscalelabs.com/robots/yam
- **核查日期：** 2026-11-08
- **核对范围：** 已阅读全文：官网首页、Quick start、Models、Python SDK、SO-101、Bimanual YAM 与 llms.txt；核对其公开命令、数据约定、动作执行方式、费率及安全提示。llms.txt 所列 Training、Deployments、SDK reference 等深层页面在本次访问中不可读取，未将其全文视作已审读。
- **通读：** 已完成
- **代码：** https://github.com/Dreamscale-Labs/dreamscale-hud （开放的是 HUD 适配/评测代码；不是 Dreamscale 云端推理基础设施全栈源码）
- **对应实体：** [Dreamscale Labs](../../wiki/entities/dreamscale-labs.md)

## 事实与定位

Dreamscale Labs 提供机器人模型托管推理、模型训练/微调入口、客户自有模型部署；用户选模型、连接机器人/仿真、输入任务指令。它是一种部署基础设施，不是“一个模型适配所有机器人”的通用权重。官方模型列表明确注明：每个共享模型绑定一个机器人类型及输入/输出合同。

## 公开模型与动作合同（2026-11-08 文档快照）

| 模型标识 | 对象 | 动作 chunk | 动作数据频率 |
|---|---|---|---|
| molmoact2-so101 | SO-101 | 30 × 6 | 30 Hz |
| flux-3-action-so101 | SO-101 | 42 × 6 | 30 Hz |
| pi0.5-so101 | SO-101 | 30 × 6 | 30 Hz |
| molmoact2-bimanual-yam / pi0.5-bimanualyam | 双臂 YAM | 30 × 14 | 30 Hz |
| molmoact2-droid | Franka / DROID | 15 × 8 | 15 Hz |
| pi0.5-libero | LIBERO 仿真 | 20 × 7 | 20 Hz |
| gr00t-n1.7-libero | LIBERO 仿真 | 16 × 7 | 20 Hz |
| xiaomi-robotics-1-robocasa365 | RoboCasa365 仿真 | 16 × 12 | 20 Hz |

以上是动作输出采样率和 chunk 大小，**不是单次云端推理频率或端到端闭环无阻塞保证**。可用模型受组织访问权限约束，应执行 `dreamscale models list`。

- SO-101：RGB side/wrist 两相机 + 6 关节位置（LeRobot 标定角度，夹爪 0–100）；输出六个**绝对目标**，不是力矩。
- 双臂 YAM：三相机 + 两侧关节/夹爪状态，14 个绝对目标（两侧各 6 关节 + 1 夹爪），关节弧度、夹爪 [0,1]。
- LIBERO：agent/wrist 图像 + 8 维状态；7 维 delta（位置三维、axis-angle 旋转三维、夹爪）——与 SO-101 绝对关节位置不同。
- Python 接口：`dreamscale.connect(model)` / `dreamscale.aconnect(...)`、`policy.predict(observation, instruction=...)`、`policy.next_action(...)`；后者参与按 tick 推进、动作块混合/断线处理。不能把 predict 直接当作已安全执行。
- 端侧安全：官方明确要求本地限制关节、速度、工作空间、实体急停。云推理不可替代安全控制器。

## 可复现最小入口

```bash
uv tool install dreamscale
dreamscale login
dreamscale sim libero --model pi0.5-libero
```

```bash
dreamscale robots so101 setup
dreamscale robots so101 calibrate
dreamscale doctor so101 --model molmoact2-so101
dreamscale run so101 --model molmoact2-so101
```

SDK 文档使用 `uv add dreamscale`，Python 3.11+；SO-101 指令是 `pick up the red block and put it in the box`。官方 quick start 的定价说明为 **连接期间 $5 / robot-hour，按秒计费**（2026-11-08 文档快照，实际以下单时页面为准）；不要把它推算成整机运营成本或全部私有模型的固定报价。

## 开源边界与注意

- 官网没有给出可核实的完整训练/云端 serving 后端仓库；公开的 `dreamscale-hud` 是服务于集成与评测的 MIT 许可代码，不代表云推理服务开放源代码。
- 部署选项所说“支持自有模型”和“支持训练”，**不等于**所列任意权重与硬件可无缝互换；须按机器人观测/动作合同以及权限校验。
- 这份归档主要承载已读的公开接口文档与官网；未读取的深层 Training/Deployments 页面不作逐项参数核验。

## 交叉引用

- [Dreamscale Labs 实体页](../../wiki/entities/dreamscale-labs.md)
- [官方博客/技术论证读书笔记](../blogs/dreamscale_cloud_robot_inference_2026.md)
- [官方 HUD 源码审读](../repos/dreamscale-hud.md)
