# SwarmNxt 项目站（EPFL LIS）

> 来源归档

- **标题：** SwarmNxt Documentation
- **类型：** site / project-page / documentation
- **链接：** https://lis-epfl.github.io/swarm-nxt/
- **代码：** https://github.com/lis-epfl/swarm-nxt
- **论文：** [arXiv:2609.11382](https://arxiv.org/abs/2609.11382)
- **入库日期：** 2026-09-16
- **一句话说明：** EPFL LIS 托管的 SwarmNxt 官方文档站：IT 基础设施、OmniNxt 机体组装、软件常见任务与飞行检查单。

## 源码开放核查（步骤 2.5）

| 类别 | 状态 | 说明 |
|------|------|------|
| 平台代码 | **已开源** | Footer / 首页链至 [GitHub](https://github.com/lis-epfl/swarm-nxt) |
| 硬件 BOM + 教程 | **已开源** | 文档站 `drone-setup.md` + YouTube 组装视频 |
| OmniNxt 设计 | **已开源** | 链至 [HKUST-Aerial-Robotics/OmniNxt](https://github.com/HKUST-Aerial-Robotics/OmniNxt) |
| 预训练权重 | N/A | 平台论文，无 NN 权重发布 |

## 站点结构（策展）

1. **IT Infrastructure Setup** — Host PC、网络、路由器、动捕接口
2. **Drone Setup** — 逐机硬件组装与 Orin 软件安装
3. **Flying** — 预飞 Ansible、仪表盘 `:8080`、单机 arm/takeoff/land 验证
4. **Software Common Tasks** — SSH、Wi-Fi、ROS 包管理、VNC

## 对 wiki 的映射

- [paper-swarmnxt](../../wiki/entities/paper-swarmnxt.md)
- [sources/repos/swarm_nxt.md](../repos/swarm_nxt.md)
