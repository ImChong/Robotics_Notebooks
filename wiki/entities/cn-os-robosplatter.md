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
  - ../../sources/repos/robosplatter.md
summary: "地平线 开源项目 RoboSplatter（仿真环境）：官方仿真环境与模型接入：联调、策略回放与 Sim2Sim 验证…"
institutions:
  - horizon-robotics
---

# RoboSplatter

## 一句话定义

**RoboSplatter** 是 [地平线](https://github.com/HorizonRobotics) 公开的 **仿真环境** 开源项目：官方仿真环境与模型接入：联调、策略回放与 Sim2Sim 验证

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
| 类别 | 仿真环境 |
| 官方组织 | https://github.com/HorizonRobotics |

## 工程实践

1. 从官方 GitHub/Gitee 组织检索 `RoboSplatter` 仓库并核对 README 许可与依赖。
2. 对照本库 [424 项覆盖索引](../queries/china-domestic-opensource-424-coverage.md) 查看同公司其它入口是否共用训练/部署链路。
3. 若与既有方法页（如 RL 框架、VLA、SDK）主题相同，优先读关联页中的「开源入口」小节，避免重复维护平行叙事。

## 局限与风险

- 公众号清单为 **策展快照**（2026-09-06）；仓库更名、归档或许可证变化须回官方组织页核实。
- **开源状态**：以仓库 README 与 release 为准（入库日按文章描述归纳，未逐仓 clone 验证）。

## 源码运行时序图

入口：`render-cli --data_file robo_splatter/config/gs_data_basic.yaml --camera_extrinsic ... --camera_intrinsic ... --image_height 480 --image_width 640 --output_dir ./output/background`（`pyproject.toml` 中 `render-cli = robo_splatter.scripts.render_scene:entrypoint`）

```mermaid
sequenceDiagram
    participant U as 用户 render-cli
    participant E as scripts/render_scene.py entrypoint
    participant V as models/gaussians.py VanillaGaussians（背景）
    participant R as models/gaussians.py RigidsGaussians（前景实例）
    participant S as render/scenes.py Scene
    participant C as models/camera.py Camera
    participant G as gsplat.rendering.rasterization

    U->>E: parse_args（数据 yaml、相机位姿/内参、图像尺寸）
    E->>V: init_from_yaml(data_file)，读 background.model_path（.ply）
    E->>R: init_from_yaml(data_file)，读 foreground.instances（GSInstance）
    E->>S: Scene(RenderConfig, background_models, foreground_models, device)
    S->>S: 模型 .to(device)，set_eval
    loop 每个相机位姿 idx
        E->>C: Camera.init_from_pose_list([x,y,z,qx,qy,qz,qw], K, H, W)
        E->>S: scene.render(camera, instances_poses, scene_type, coord_system)
        S->>S: collect_gaussians(c2w, instances_pose, render_type)
        S->>R: get_gaussians(c2w, instances_pose)（_compute_transform 把实例位姿作用到前景高斯）
        S->>V: get_gaussians(c2w)
        S->>S: SCENE 模式下前景与背景高斯拼接为 GaussianData
        S->>G: rasterization(means, quats, scales, opacities, colors, viewmats, Ks, render_mode="RGB+ED")
        G-->>S: renders, alphas, info
        S-->>E: RenderResult(rgb, depth, opacity)
    end
    E->>E: 全局深度归一化并 applyColorMap，拼接 RGB 与深度图
    E->>E: cv2.imwrite(output_dir/image_XXXX.png)
    opt 指定 --gen_mp4_path
        E->>E: utils/helper.create_mp4_from_images(fps=10)
    end
```

图依据：HorizonRobotics/RoboSplatter@8f2c4a9，读过 README.md、pyproject.toml、robo_splatter/config/gs_data_basic.yaml、robo_splatter/scripts/render_scene.py、robo_splatter/render/scenes.py、robo_splatter/models/gaussians.py（init_from_yaml 与 RigidsGaussians.get_gaussians）、robo_splatter/models/camera.py（init_from_pose_list）。

## 关联页面

- [国内具身开源全景技术地图](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md)
- [HMI 开源项目主表导读](../queries/hmi-opensource-projects-coverage.md)
- [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md)

## 参考来源

- [RoboSplatter 源码归档](../../sources/repos/robosplatter.md)（<https://github.com/HorizonRobotics/RoboSplatter>）

- [国内具身智能开源全景（微信公众号）](../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)
- 源码核对（2026-11-02）：HorizonRobotics/RoboSplatter@8f2c4a9，读过 README.md、pyproject.toml、robo_splatter/config/gs_data_basic.yaml、robo_splatter/scripts/render_scene.py、robo_splatter/render/scenes.py、robo_splatter/models/{gaussians,camera}.py（部分）；时序图依据以上文件，未运行代码。

## 推荐继续阅读

- [地平线 官方组织](https://github.com/HorizonRobotics)
