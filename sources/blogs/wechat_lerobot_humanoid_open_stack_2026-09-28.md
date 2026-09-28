# LeRobot Humanoid 开源双足栈（微信公众号策展）

> 来源归档（blog / 微信公众号）

- **标题：** LeRobot Humanoid 开源硬件与软件栈资源索引（策展向）
- **类型：** blog
- **原始链接：** https://mp.weixin.qq.com/s/4NihwYhK9ajrpwRUwM82OA
- **入库日期：** 2026-09-28
- **抓取说明：** Cloud Agent 直链抓取遇微信「环境异常」验证页；正文要点以下方 **策展清单** 与 **官方 GitHub README**（2026-09-28 复核）为准。
- **一句话说明：** 汇总 Hugging Face **LeRobot Humanoid** 四仓分工（硬件 BOM/CAD、运行时 CAN+MuJoCo、MJLab 训练、MJWarp 参数辨识）及 LeRobot 文档、制造资源与 NVIDIA/Hub 生态入口。

## 核心代码仓库（四仓分工）

| 角色 | 仓库 | 要点 |
|------|------|------|
| **硬件** | [huggingface/lerobot-humanoid-hardware](https://github.com/huggingface/lerobot-humanoid-hardware) | 机械装配文档、制造文件、电子接线与连接器映射、BOM、电机预组装调试；STL 按左右腿与躯干子装配组织 |
| **运行时** | [huggingface/lerobot-humanoid-runtime](https://github.com/huggingface/lerobot-humanoid-runtime) | MuJoCo 仿真控制器、真机 CAN MIT + 安全机制、ONNX/Torch 策略推理、IMU 集成、LeRobot 集成控制器；电机扫描、交互归零、IMU 校准、关节方向验证、数据集生成等工具 |
| **训练** | [Virgileboat/lerobot-legged-zoo](https://github.com/Virgileboat/lerobot-legged-zoo) | LeRobot Humanoid 的 MJCF + **MJLab** 训练环境；平地/崎岖地形行走任务；`uv` 管理依赖，训练需 **CUDA GPU** |
| **参数辨识** | [Virgileboat/lerobot-humanoid-identification](https://github.com/Virgileboat/lerobot-humanoid-identification) | **MJWarp + CMA-ES** 关节级动力学辨识，缩小仿真与真机参数差距 |

## 官方文档与规范

- [LeRobot 文档总入口](https://huggingface.co/docs/lerobot)
- [安装](https://huggingface.co/docs/lerobot/installation)
- [硬件集成](https://huggingface.co/docs/lerobot/integrate_hardware)
- [LeRobotDataset v3](https://huggingface.co/docs/lerobot/lerobot-dataset-v3)

## 硬件与制造资源

- [Onshape CAD（双足平台）](https://cad.onshape.com/documents/fb645318a27646d1d8840be6/w/d1cae8805fb652b4d1614997/e/804a1da43f242001a05129b4)
- [3D 打印指南](https://github.com/huggingface/lerobot-humanoid-hardware/blob/main/docs/manufacturing/printing_guide.md)
- [电机调试脚本入口](https://github.com/huggingface/lerobot-humanoid-hardware/blob/main/hardware/config/commission_motor.py)
- [BOM 采购清单](https://github.com/huggingface/lerobot-humanoid-hardware/tree/main/hardware/bom)
- [RobStride 电机官网](https://www.robstride.com)

## AI 与仿真集成

- [NVIDIA × Hugging Face LeRobot 博客](https://blogs.nvidia.com/blog/hugging-face-lerobot-models-frameworks-open-robotics/)
- [Isaac GR00T 1.7](https://developer.nvidia.com/isaac/groot)
- [Isaac Teleop](https://nvidia.github.io/IsaacTeleop/)
- [HF Hub `lerobot` 组织](https://huggingface.co/lerobot)
- [LeRobot 数据集筛选](https://huggingface.co/datasets?other=LeRobot)

## 社区

- [LeRobot Discord](https://discord.gg/s3KuuzsPFb)
- [huggingface/lerobot 主仓](https://github.com/huggingface/lerobot)

## 开源核查（2026-09-28）

| 资产 | 结论 |
|------|------|
| 四仓 GitHub | **已开源**；hardware/runtime/identification 为 Apache-2.0；legged-zoo README 未标 license 字段（以仓内 LICENSE 为准） |
| 预训练行走策略 | legged-zoo README 写明 **no pretrained policies**；训练任务已提供，权重需自训 |
| CAD | Onshape 公开文档可浏览器查看/导出 |

## 对 wiki 的映射

- 升格 [LeRobot Humanoid](../../wiki/entities/lerobot-humanoid.md) 实体页
- 交叉更新 [LeRobot](../../wiki/entities/lerobot.md)、[开源人形硬件对比](../../wiki/entities/open-source-humanoid-hardware.md)、[人形机器人](../../wiki/entities/humanoid-robot.md)
- 仓库归档见 `sources/repos/lerobot_humanoid_*.md`、`sources/repos/lerobot_legged_zoo.md`
