---
type: entity
tags: [repo, china-embodied-opensource, open-source, project]
status: draft
updated: 2026-09-06
related:
  - ../overview/china-domestic-embodied-opensource-76-companies-technology-map.md
  - ../entities/humanoid-motion-intelligence.md
  - ../queries/china-domestic-opensource-424-coverage.md
sources:
  - ../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md
  - ../../sources/repos/fastumi_data_platform_web.md
summary: "鹿明机器人 开源项目 FastUMI_Data_Platform_Web（数据集/Benchmark）：提供FastUMI Pro数据管理Web平台，用于查看、组织和管理采集任务与数据。…"
institutions:
  - lumos
---

# FastUMI_Data_Platform_Web

## 一句话定义

**FastUMI_Data_Platform_Web** 是 [鹿明机器人](https://github.com/LumosRobot) 公开的 **数据集/Benchmark** 开源项目：提供FastUMI Pro数据管理Web平台，用于查看、组织和管理采集任务与数据。

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

| 字段 | 内容 |
|------|------|
| 机构 | 鹿明机器人 |
| 类别 | 数据集/Benchmark |
| 官方组织 | https://github.com/LumosRobot |

### 架构流程图

依据 [LumosRobot/FastUMI_Data_Platform_Web](https://github.com/LumosRobot/FastUMI_Data_Platform_Web) 仓库 README 与顶层文件结构：

```mermaid
flowchart LR
  DEV["FastUMI Pro 采集设备 (视觉 / 触觉 / 力数据)"] --> SDK["FastUMI_Hardware_SDK (ROS1)"]
  SDK --> BACK["后端二进制服务 (deploy.sh 部署 / backend_process_clean.sh 清理)"]
  BACK --> WEB["fastumi_data_platform_frontend Web 平台 (localhost:8000)"]
  WEB --> COL["data collection (数据采集)"]
  COL --> REP["data replay (实时回放与复核)"]
  REP --> QA["内置质量评估筛选"]
  QA --> OPEN["data open (导出与管理用于模型训练)"]
```

## 工程实践

1. 从官方 GitHub/Gitee 组织检索 `FastUMI_Data_Platform_Web` 仓库并核对 README 许可与依赖。
2. 对照本库 [424 项覆盖索引](../queries/china-domestic-opensource-424-coverage.md) 查看同公司其它入口是否共用训练/部署链路。
3. 若与既有方法页（如 RL 框架、VLA、SDK）主题相同，优先读关联页中的「开源入口」小节，避免重复维护平行叙事。

## 源码运行时序图

以下时序图基于仓库内实际脚本 `deploy.sh` 与 `backend_process_clean.sh` 编写；该仓库前端为 `fastumi_data_platform_frontend/` 预构建静态产物，后端 `fastumi_data_platform_backend` 为外部下载的二进制程序，因此时序仅描述部署与进程清理脚本的真实执行过程：

```mermaid
sequenceDiagram
  participant User as 操作者
  participant Deploy as deploy.sh
  participant Nginx as Nginx (dataplatform.conf)
  participant Back as fastumi_data_platform_backend
  participant Clean as backend_process_clean.sh

  User->>Deploy: bash deploy.sh
  Deploy->>Deploy: 检查 ./fastumi_data_platform_frontend 目录是否存在
  Deploy->>Nginx: apt install nginx / systemctl start 与 enable nginx
  Deploy->>Nginx: 生成 /etc/nginx/conf.d/dataplatform.conf (listen 8000, /data/ -> $HOME/fastumi/preview/)
  Deploy->>Nginx: nginx -s reload 并复制前端至 /var/www/html/fastumi_data_platform_frontend
  Deploy-->>User: 输出 http://localhost:8000 (user/user123) 并等待确认后端二进制文件
  User->>Deploy: 按 Enter 确认 ./fastumi_data_platform_backend 就绪
  Deploy->>Back: chmod +x 并前台运行 ./fastumi_data_platform_backend

  Note over User,Clean: 端口 8887 占用冲突或需清理后端残留进程时
  User->>Clean: bash backend_process_clean.sh
  Clean->>Clean: ss -anpltr | grep -w 8887 提取占用 PIDs
  alt 找到占用 PID 且用户确认 y/Y
    loop 遍历每个 PID
      Clean->>Back: kill PID (SIGTERM) -> sleep 1
      opt ps -p PID 仍存在
        Clean->>Back: kill -9 PID (SIGKILL)
      end
    end
    Clean->>Clean: ss -anpltr | grep -w 8887 复检端口
  else 未找到进程或用户取消
    Clean-->>User: 输出提示并退出 (exit 0 或 exit 1)
  end
```

## 局限与风险

- 公众号清单为 **策展快照**（2026-09-06）；仓库更名、归档或许可证变化须回官方组织页核实。
- **开源状态**：以仓库 README 与 release 为准（入库日按文章描述归纳，未逐仓 clone 验证）。

## 关联页面

- [国内具身开源全景技术地图](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md)
- [HMI 开源项目主表导读](../queries/hmi-opensource-projects-coverage.md)
- [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md)

- [机器人学习数据管线](../methods/robot-learning-data-pipelines.md)：了解采集、同步与数据质量检查。
- [机器人软件系统集成](../methods/robot-software-integration.md)：对照通信、驱动与模块接入。

## 参考来源

- [FastUMI_Data_Platform_Web 源码归档](../../sources/repos/fastumi_data_platform_web.md)（<https://github.com/LumosRobot/FastUMI_Data_Platform_Web>）

- [国内具身智能开源全景（微信公众号）](../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)

## 推荐继续阅读

- [鹿明机器人 官方组织](https://github.com/LumosRobot)
