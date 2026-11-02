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
  - ../../sources/repos/geoflowslam.md
summary: "地平线 开源项目 GeoFlowSlam（感知/导航）：感知/定位/建图模块：为导航与控制提供环境状态…"
institutions:
  - horizon-robotics
---

# GeoFlowSlam

## 一句话定义

**GeoFlowSlam** 是 [地平线](https://github.com/HorizonRobotics) 公开的 **感知/导航** 开源项目：感知/定位/建图模块：为导航与控制提供环境状态

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| SDK | Software Development Kit | 真机控制与状态读取接口 |
| RL | Reinforcement Learning | 强化学习训练与策略优化 |
| VLA | Vision-Language-Action | 视觉–语言–动作统一策略 |
| Sim2Real | Simulation to Real | 仿真策略迁移真机 |
| URDF | Unified Robot Description Format | 机器人描述与仿真资产 |

## 为什么重要

- 收录于 [国内具身智能开源全景（76 家 · 424 项）](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md) 的 **第四层** 分组。
- 与 [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md) 同源策展；本页为 **独立详情节点**，便于从公司清单跳到机制与入口说明。

## 核心原理

| 字段 | 内容 |
|------|------|
| 机构 | 地平线 |
| 类别 | 感知/导航 |
| 官方组织 | https://github.com/HorizonRobotics |

## 工程实践

1. 从官方 GitHub/Gitee 组织检索 `GeoFlowSlam` 仓库并核对 README 许可与依赖。
2. 对照本库 [424 项覆盖索引](../queries/china-domestic-opensource-424-coverage.md) 查看同公司其它入口是否共用训练/部署链路。
3. 若与既有方法页（如 RL 框架、VLA、SDK）主题相同，优先读关联页中的「开源入口」小节，避免重复维护平行叙事。

## 局限与风险

- 公众号清单为 **策展快照**（2026-09-06）；仓库更名、归档或许可证变化须回官方组织页核实。
- **开源状态**：以仓库 README 与 release 为准（入库日按文章描述归纳，未逐仓 clone 验证）。

## 源码运行时序图

入口：`python script/run_orbslam/RGBD-Inertial/script/run_rgbd_vi_g1.py`（README 写作 `script/run_orbslam/run_rgbd_vi_g1.py`，仓库内实际在 `RGBD-Inertial/script/` 下），其内部 `os.system` 调用 `Examples/RGB-D-Inertial/rgbd_inertial Vocabulary/ORBvoc.txt <config.yaml> <dataset_path> <dataset_path>/associate.txt`；先行数据准备：`bash script/tools/parse_d435i.sh`（README，该脚本未读）。

```mermaid
sequenceDiagram
    participant U as 用户 / run_rgbd_vi_g1.py
    participant EX as Examples/RGB-D-Inertial/<br/>rgbd_inertial.cc main()
    participant SYS as src/System.cc<br/>ORB_SLAM3::System
    participant TR as src/Tracking.cc<br/>Tracking
    participant LM as src/LocalMapping.cc<br/>LocalMapping::Run（线程）
    participant LC as src/LoopClosing.cc<br/>LoopClosing::Run（线程）
    participant LI as LidarMapping（可选）

    U->>EX: os.system(rgbd_inertial vocab yaml dataset associate.txt)
    EX->>EX: 读 yaml；LoadImages(associate.txt)；LoadIMU(imu/imu.txt)
    EX->>EX: 若 UseRobotOdom：LoadOdom(imu/odom.txt)
    EX->>SYS: System SLAM(vocab, yaml, IMU_RGBD, viewer, ...)
    SYS->>TR: new Tracking（词典、Atlas、KeyFrameDatabase、Settings）
    SYS->>LM: new LocalMapping；起线程 LocalMapping::Run
    SYS->>LC: new LoopClosing；起线程 LoopClosing::Run
    opt 配置了激光点云建图
        SYS->>LI: new LidarMapping（SetPointCloudMapper 给 Tracking/LocalMapping/LoopClosing）
    end
    loop 每帧图像
        EX->>EX: 取上一帧到当前帧之间的 IMU 与 odom 测量
        EX->>SYS: TrackRGBD(imRGB, imD, t, vImuMeas, vOdomMeas)（定义 ENABLE_ASYNC 时为 TrackRGBDAsync）
        SYS->>TR: GrabImuData / GrabOdomData
        SYS->>TR: GrabImageRGBD(im, depth, t)
        TR->>TR: 构造 Frame（ORB 提取）→ Track()
        TR->>TR: PreintegrateIMU；状态预测：PredictStateIMU（+ PredictStateOdom）或光流 EstimatePoseByOF
        TR->>TR: TrackWithMotionModel / TrackWithMotionModelICP（GICP）/ TrackReferenceKeyFrame / Relocalization
        TR->>TR: TrackLocalMap；NeedNewKeyFrame → CreateNewKeyFrame
        TR->>LM: InsertKeyFrame
        TR-->>SYS: Tcw、mState
    end
    EX->>SYS: Shutdown(dataset_path)
    EX-->>U: 输出 median / mean tracking time、tracking lost count
```

图依据：HorizonRobotics/GeoFlowSlam@40c89ab，读过 README.md、script/run_orbslam/RGBD-Inertial/script/run_rgbd_vi_g1.py（末段调用部分）、script/run_orbslam/RGBD-Inertial/config/g1_op_icp_lidar_outdoor.yaml（开关项）、Examples/RGB-D-Inertial/rgbd_inertial.cc、src/System.cc（构造与 TrackRGBD）、src/Tracking.cc（GrabImageRGBD、Track 的分支调用、PredictStateOdom、TrackWithMotionModelICP 开头）。

## 关联页面

- [国内具身开源全景技术地图](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md)
- [HMI 开源项目主表导读](../queries/hmi-opensource-projects-coverage.md)
- [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md)

## 参考来源

- [GeoFlowSlam 源码归档](../../sources/repos/geoflowslam.md)（<https://github.com/HorizonRobotics/GeoFlowSlam>）

- [国内具身智能开源全景（微信公众号）](../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)
- 源码核对（2026-11-02）：HorizonRobotics/GeoFlowSlam@40c89ab，读过 README.md、script/run_orbslam/RGBD-Inertial/script/run_rgbd_vi_g1.py（调用段）、config/g1_op_icp_lidar_outdoor.yaml（开关项）、Examples/RGB-D-Inertial/rgbd_inertial.cc、src/System.cc（构造与 TrackRGBD）、src/Tracking.cc（局部）；LocalMapping.cc / LoopClosing.cc / Optimizer.cc / 因子图部分未读；时序图依据以上文件，未运行代码。

## 推荐继续阅读

- [地平线 官方组织](https://github.com/HorizonRobotics)
