# RLark（跨集群具身智能云原生平台）

> 来源归档

- **标题：** RLark — Cross-Cluster Embodied Intelligence Cloud-Native Platform
- **类型：** repo
- **组织：** [RLinf](https://github.com/RLinf) 生态（与 [RLinf 训练系统](rlinf.md) 同组织）
- **代码：** <https://github.com/RLinf/RLark>
- **文档：** <https://rlark.readthedocs.io/en/latest/>（中文：<https://rlark.readthedocs.io/zh-cn/latest/>）
- **Quick Start：** <https://rlark.readthedocs.io/en/latest/quickstart/>（仓库 [README Quick Start](https://github.com/RLinf/RLark#quick-start) 指向同一套已验证流程）
- **许可：** Apache-2.0
- **入库日期：** 2026-09-29
- **一句话说明：** **跨站点 GPU 集群与边缘设备** 的统一云原生编排：kcp 控制面 + Domain/Node/Job/Task CRD；跨集群 Pod 直连（TUN + gVisor netstack + SSH 隧道）；云侧 RL/LLM 训练到机械臂/相机/传感器边缘部署的 **声明式 Job** 抽象。
- **步骤 2.5（开源核查）：** **已开源**（2026-09 GitHub 复核）— 2026/09 公告开源；完整 Go 控制面/agent、React 控制台、embodied-runtime（ROS/相机）、Python/Go SDK 与 Read the Docs 指南；Quick Start 含 **一键 CLI**（控制面 + 双 kind 数据面 + 跨集群连通验证）与 **Web UI** 建集群/Domain/Job 流程。

## 与本仓库知识的关系

| 主题 | 关系 |
|------|------|
| [RLark 实体](../../wiki/entities/rlark.md) | wiki 主节点 |
| [RLinf](rlinf.md) | 同生态：**RLinf** 偏 **训练算法与离线/在线 RL 管线**；**RLark** 偏 **多集群/云–边资源编排与网络** |
| [APXinf-robo](apxinf-robo.md) | 同生态：训练/编排 → **端侧 π₀.₅ 推理** 的最后一环可经 RLark 调度边缘节点 |
| [RPent](rpent.md) | 同生态 agentic 运行时；大规模 harness 评测可部署在 RLark 管理的集群上 |
| [Genie Sim 3.0](../../wiki/entities/genie-sim-3.md) | 材料称 Genie Sim 可接 RLinf；RLark 可作为 **跨集群训练/部署控制面** 的工程选项 |
| [VLA 开源复现景观](../../wiki/overview/vla-open-source-repro-landscape-2025.md) | 「搭集群 RL 基建」索引可延伸到 **云–边编排** |

## 能力摘要（README / 文档）

- **Embodied AI Workload Orchestration：** 云 GPU 训练（RL/LLM）到边缘（机械臂、传感器、相机）全链路 **Job/Task** 声明式抽象。
- **Multi-Runtime Data Plane：** Kubernetes 统一管理云 GPU 与边缘；规划扩展 Docker / Raw 轻量边缘运行时。
- **Cross-Cluster：** **Domain**（虚拟网络域）+ **Node**（计算节点）CRD；控制面运行于 **kcp**。
- **Cross-Cluster Pod Networking：** TUN + gVisor netstack + SSH 隧道，**无需 NAT 打洞** 的 Pod–Pod 通信（云 GPU 与边缘机器人直连）。
- **安全：** 双层 X.509 + SSH 证书（Agent、Domain 转发、用户 SSH）。
- **可观测：** Prometheus、Pod 日志流、Web 管理 UI。
- **子模块：** `apps/embodied-runtime`（ROS/相机）、`apps/rlark-ui`、`sdks/embodied-runtime-python|go`、`proto/embodied-runtime`。

## 为何值得保留

- 填补 **「RLinf 算法栈 → 多站点/云–边统一调度」** 的公开入口，避免把 RLinf 与 RLark 混为同一仓库。
- Quick Start 在 Read the Docs 上 **可复现验证**（kind 双数据面 + 跨集群连通），适合作为集群选型前的 POC 文档锚点。
