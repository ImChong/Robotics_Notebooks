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
  - ../../sources/repos/fastumi_camera.md
summary: "鹿明机器人 开源项目 FastUMI_Camera（数据集/Benchmark）：提供FastUMI板载相机工具，承担第一视角视频采集与设备接入。…"
institutions:
  - lumos
---

# FastUMI_Camera

## 一句话定义

**FastUMI_Camera** 是 [鹿明机器人](https://github.com/LumosRobot) 公开的 **数据集/Benchmark** 开源项目：提供FastUMI板载相机工具，承担第一视角视频采集与设备接入。

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

## 工程实践

1. 从官方 GitHub/Gitee 组织检索 `FastUMI_Camera` 仓库并核对 README 许可与依赖。
2. 对照本库 [424 项覆盖索引](../queries/china-domestic-opensource-424-coverage.md) 查看同公司其它入口是否共用训练/部署链路。
3. 若与既有方法页（如 RL 框架、VLA、SDK）主题相同，优先读关联页中的「开源入口」小节，避免重复维护平行叙事。

## 局限与风险

- 公众号清单为 **策展快照**（2026-09-06）；仓库更名、归档或许可证变化须回官方组织页核实。
- **开源状态**：以仓库 README 与 release 为准（入库日按文章描述归纳，未逐仓 clone 验证）。

## 源码运行时序图

入口：`cd FastUMI_Camera_Steam && python readimage.py`（视频流采集）；标定部分是预编译可执行文件 `./demo-api` 与 `./pipe_srv`（源码不在仓库内，下图只画出 README / readme.txt 描述的使用顺序，不含其内部实现）。

```mermaid
sequenceDiagram
    autonumber
    participant U as 用户
    participant RI as readimage.py
    participant CV as cv2.VideoCapture(V4L2)
    participant DEV as /dev/video0（FastUMI 相机，需 xvsdk 驱动）
    participant PS as pipe_srv（二进制）
    participant DA as demo-api（二进制）

    U->>RI: python readimage.py
    RI->>CV: VideoCapture(DEV, CAP_V4L2)，失败则抛 RuntimeError
    RI->>CV: set FOURCC=YU12、CONVERT_RGB=0、1280x1280、FPS=100、BUFFERSIZE=1
    CV->>DEV: 协商 I420 格式与帧率
    loop grab_bgr() 直到按 q
        RI->>CV: cap.read()
        CV-->>RI: raw I420 帧
        RI->>RI: reshape(H*3//2, W) + cvtColor(COLOR_YUV2BGR_I420)
        RI->>U: cv2.imshow 显示
    end
    RI->>CV: cap.release() + destroyAllWindows()
    Note over U,DA: 标定流程（按 README / readme.txt，二进制内部未读）
    U->>DA: ./demo-api（先启动）
    U->>PS: ./pipe_srv
    U->>PS: 输入 1-0-37
    PS->>DA: 经管道转发命令
    DA-->>U: 终端打印 RGB 标定参数
```

图依据：LumosRobot/FastUMI_Camera@16afe0f，读过 README.md、FastUMI_Camera_Steam/readimage.py、FastUMI_Camera_Calibration-master/readme.txt；`demo-api`、`pipe_srv` 为二进制，未反编译，「命令经管道转发」为 README 描述。

## 关联页面

- [国内具身开源全景技术地图](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md)
- [HMI 开源项目主表导读](../queries/hmi-opensource-projects-coverage.md)
- [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md)

## 参考来源

- [FastUMI_Camera 源码归档](../../sources/repos/fastumi_camera.md)（<https://github.com/LumosRobot/FastUMI_Camera>）

- [国内具身智能开源全景（微信公众号）](../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)
- 源码核对（2026-11-02）：LumosRobot/FastUMI_Camera@16afe0f，读过 README.md、FastUMI_Camera_Steam/readimage.py、FastUMI_Camera_Calibration-master/readme.txt；时序图依据以上文件，未运行代码。

## 推荐继续阅读

- [鹿明机器人 官方组织](https://github.com/LumosRobot)
