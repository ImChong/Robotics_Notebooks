---
type: entity
tags:
- repo
- galbot
- vla
- grasping
- sim2real
status: complete
updated: '2026-10-05'
related:
- ../overview/china-domestic-embodied-opensource-76-companies-technology-map.md
- ../entities/humanoid-motion-intelligence.md
- ../queries/china-domestic-opensource-424-coverage.md
- ./galbot-astrabrain.md
- ../methods/vla.md
- ../concepts/sim2real.md
sources:
- ../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md
- ../../sources/repos/graspvla.md
summary: GraspVLA 在 SynGrasp-1B 合成动作数据上预训练，以自回归感知与 flow-matching 动作生成联合支持开放词汇抓取和零样本 Sim2Real。
institutions:
- galbot
---

# GraspVLA：十亿级合成数据预训练的抓取基座

## 一句话定义

GraspVLA 在 SynGrasp-1B 合成动作数据上预训练，以自回归感知与 flow-matching 动作生成联合支持开放词汇抓取和零样本 Sim2Real。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
| --- | --- | --- |
| VLA | Vision-Language-Action | 视觉和语言条件下生成动作 |
| CoT | Chain of Thought | 串联感知与动作生成的思维链 |
| HF | Hugging Face | 官方模型与数据分发平台 |

## 为什么重要

- 展示大规模仿真数据能否预训练可真机迁移的抓取策略。
- 已有模型服务、训练数据和仿真/真机接口，可从原先的公司清单入口转为复现入口。

## 核心原理

**SynGrasp-1B** 是十亿帧合成抓取数据，官方 README 称覆盖 **240 类、超过一万物体**。模型将自回归感知任务与 flow-matching 动作生成置于统一 CoT，以联合利用动作数据与互联网语义；这使合成动作学习与开放词汇理解能共享表征。

这里的“零样本”指论文设定下的直接 Sim2Real 抓取迁移，不代表任意本体、任意相机和任意接触任务无需适配。

## 源码运行时序图

```mermaid
sequenceDiagram
    autonumber
    participant U as offline_test / 环境客户端
    participant S as vla_network.scripts.serve
    participant C as checkpoint + preprocessor
    participant R as 仿真 / 真机接口
    S->>C: 加载同一实验的配置、预处理与权重
    U->>S: 图像、指令、机器人状态
    S-->>U: 感知输出与动作预测
    U->>R: 转换并执行动作
    R-->>U: 下一帧观测
```

从 `offline_test` 验证服务协议，再接 README 的 simulation playground；真机控制需要独立适配。

## 工程实践

1. 锁定官方 `PKU-EPIC/GraspVLA` 的版本，按 `uv sync --locked` 安装。
2. 获取 HF `vegebirrd/GraspVLA` 权重；`config.json`、`preprocessor.npz` 与 checkpoint 必须来自同一实验目录。
3. 启动 `vla_network.scripts.serve`，用 `vla_network.scripts.offline_test` 验证请求/响应及感知框。
4. 先接 simulation playground，再查 real-world control interface；训练入口使用 SynGrasp-1B。
5. **开源核查（2026-10-05）**：推理、训练入口、权重和 SynGrasp-1B 已列出；数据发布新闻日期为 2026-08-19。

## 局限与风险

- 配置/归一化不匹配会让可运行的服务生成错误动作，须先做离线回放。
- 感知泛化与真机抓取成功率要分开；仿真 benchmark 不能替代硬件协议。
- 公开代码与模型可用不自动说明所有训练依赖或每个许可都相同，使用前检查相应仓库/资产说明。

## 关联页面

- [AstraBrain](./galbot-astrabrain.md)
- [VLA](../methods/vla.md)
- [Sim2Real](../concepts/sim2real.md)

## 参考来源

- [官方 GraspVLA 代码与资产核查](../../sources/repos/graspvla.md)

## 推荐继续阅读

- [官方仓库](https://github.com/PKU-EPIC/GraspVLA)
- [原始论文](https://arxiv.org/abs/2505.03233)
