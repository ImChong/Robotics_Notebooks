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
  - ../../sources/repos/rdk_model_zoo.md
summary: "地瓜机器人 开源项目 rdk_model_zoo（SDK/驱动）：提供在RDK平台部署的模型示例与转换入口，覆盖机器人感知和边缘推理任务。…"
institutions:
  - d-robotics
---

# rdk_model_zoo

## 一句话定义

**rdk_model_zoo** 是 [地瓜机器人](https://github.com/D-Robotics) 公开的 **SDK/驱动** 开源项目：提供在RDK平台部署的模型示例与转换入口，覆盖机器人感知和边缘推理任务。

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
| 机构 | 地瓜机器人 |
| 类别 | SDK/驱动 |
| 官方组织 | https://github.com/D-Robotics |

## 工程实践

1. 从官方 GitHub/Gitee 组织检索 `rdk_model_zoo` 仓库并核对 README 许可与依赖。
2. 对照本库 [424 项覆盖索引](../queries/china-domestic-opensource-424-coverage.md) 查看同公司其它入口是否共用训练/部署链路。
3. 若与既有方法页（如 RL 框架、VLA、SDK）主题相同，优先读关联页中的「开源入口」小节，避免重复维护平行叙事。

## 局限与风险

- 公众号清单为 **策展快照**（2026-09-06）；仓库更名、归档或许可证变化须回官方组织页核实。
- **开源状态**：以仓库 README 与 release 为准（入库日按文章描述归纳，未逐仓 clone 验证）。

## 源码运行时序图

本仓库是多个独立样例的合集（`samples/vision`、`samples/vla`、`samples/robotics` 等），没有统一的单一入口；下图只取其中一个代表样例 `samples/vision/ultralytics_yolo` 的 Python 运行路径，其余样例的调用链未逐个核对，不能由此图推出。

入口：`python samples/vision/ultralytics_yolo/runtime/python/main.py --platform x5 --family yolov8 --task detect --model-path <模型> --test-img <图片>`（模型先由 `model/download_model.sh` 下载）

```mermaid
sequenceDiagram
    participant U as 用户命令行
    participant M as main.py main()
    participant C as cli.py
    participant D as detect.py YoloDetect
    participant B as backend.py ModelRunner
    participant RT as hbm_runtime HB_HBMRuntime（板端镜像提供）

    U->>M: --platform --family --task --model-path --test-img
    M->>C: build_parser().parse_args()
    M->>C: resolve_platform(resolve_target(platform))
    M->>C: describe_plan(profile, args)
    alt --list-models / --dry-run
        M-->>U: 只打印，不加载模型
    else 实际推理
        M->>C: ensure_model(plan)（缺失时下载）
        M->>C: load_labels(args, task)
        M->>C: prepare_runtime_model(profile, args) 返回 Model 类与 Config
        M->>D: Model(config) 构造
        D->>B: build_runner(config) 即 ModelRunner.from_selection
        B->>RT: RuntimeSession.load() 加载 .bin / .hbm
        B->>B: 读取模型元数据并绑定输入输出
        M->>D: set_scheduling_params(priority, bpu_cores)
        M->>D: predict(image)
        D->>D: preprocess：resize_with_transform，bgr_to_nv12_planes，打包 NV12 张量
        D->>B: infer，即 ModelRunner.__call__(prepared_input)
        B->>RT: model.run(prepared_input)
        RT-->>B: 原始输出，read_raw_outputs 按角色整理
        D->>D: postprocess：decode_dfl，inverse_boxes 还原原图坐标
        M->>C: present_result(args, image, result, labels)
    end
```

图依据：D-Robotics/rdk_model_zoo@7a97c17，读过 `README.md`、`samples/vision/ultralytics_yolo/README.md`、`runtime/python/main.py`、`runtime/python/cli.py`（`prepare_runtime_model` 及入口函数签名）、`runtime/python/detect.py`（`YoloDetect`）、`runtime/python/backend.py`（`ModelRunner`、`build_runner`）；`RuntimeSession`（`utils/py_utils/runtime`）与 `hbm_runtime` 内部未读；`present_result` 内部、YOLO26/分割/姿态等分支未读，图中仅按 `YoloDetect`（DFL 检测）画出。

## 关联页面

- [国内具身开源全景技术地图](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md)
- [HMI 开源项目主表导读](../queries/hmi-opensource-projects-coverage.md)
- [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md)

## 参考来源

- [rdk_model_zoo 源码归档](../../sources/repos/rdk_model_zoo.md)（<https://github.com/D-Robotics/rdk_model_zoo>）

- [国内具身智能开源全景（微信公众号）](../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)
- 源码核对（2026-11-02）：D-Robotics/rdk_model_zoo@7a97c17，读过 README.md、samples/vision/ultralytics_yolo/README.md、samples/vision/ultralytics_yolo/runtime/python/{main,cli,detect,backend}.py（部分）；时序图依据以上文件，未运行代码。

## 推荐继续阅读

- [地瓜机器人 官方组织](https://github.com/D-Robotics)
