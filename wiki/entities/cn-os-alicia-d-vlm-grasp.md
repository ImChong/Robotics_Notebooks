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
  - ../../sources/repos/alicia-d-vlm-grasp.md
summary: "玄雅科技 开源项目 Alicia-D-VLM-Grasp（VLA/操作模型）：Alicia-D 视觉语言模型抓取示例…"
institutions:
  - xuanya
---

# Alicia-D-VLM-Grasp

## 一句话定义

**Alicia-D-VLM-Grasp** 是 [玄雅科技](https://github.com/Synria-Robotics) 公开的 **VLA/操作模型** 开源项目：Alicia-D 视觉语言模型抓取示例

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| SDK | Software Development Kit | 真机控制与状态读取接口 |
| RL | Reinforcement Learning | 强化学习训练与策略优化 |
| VLA | Vision-Language-Action | 视觉–语言–动作统一策略 |
| Sim2Real | Simulation to Real | 仿真策略迁移真机 |
| URDF | Unified Robot Description Format | 机器人描述与仿真资产 |

## 为什么重要

- 收录于 [国内具身智能开源全景（76 家 · 424 项）](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md) 的 **第五层** 分组。
- 与 [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md) 同源策展；本页为 **独立详情节点**，便于从公司清单跳到机制与入口说明。

## 核心原理

| 字段 | 内容 |
|------|------|
| 机构 | 玄雅科技 |
| 类别 | VLA/操作模型 |
| 官方组织 | https://github.com/Synria-Robotics |

## 工程实践

1. 从官方 GitHub/Gitee 组织检索 `Alicia-D-VLM-Grasp` 仓库并核对 README 许可与依赖。
2. 对照本库 [424 项覆盖索引](../queries/china-domestic-opensource-424-coverage.md) 查看同公司其它入口是否共用训练/部署链路。
3. 若与既有方法页（如 RL 框架、VLA、SDK）主题相同，优先读关联页中的「开源入口」小节，避免重复维护平行叙事。

## 局限与风险

- 公众号清单为 **策展快照**（2026-09-06）；仓库更名、归档或许可证变化须回官方组织页核实。
- **开源状态**：以仓库 README 与 release 为准（入库日按文章描述归纳，未逐仓 clone 验证）。

## 源码运行时序图

入口：`python ./vlmgrasp/vlm_grasp.py`（需先按 README 配置百炼/DashScope 密钥，并启动 Alicia 的 ROS1 + MoveIt 环境与相机话题；`__main__` 中构造 `vlm_grasp()` 后调用 `run()`）。

```mermaid
sequenceDiagram
    participant M as vlm_grasp.py（vlm_grasp.run）
    participant AP as AudioPlayer.asr + speech_recog.Callback
    participant ASR as DashScope Recognition（paraformer-realtime-v2）
    participant TTS as tts.TextToSpeech（cosyvoice-v1）
    participant OD as object_detection.ObjectDetection
    participant VLM as 百炼 qwen-vl 接口（HTTP）
    participant CT as camera_trans.ObjectPoseTransformer
    participant MC as moveit_control.MoveItRobotController
    M->>AP: asr(execution_time=3.0)
    AP->>ASR: recognition.start()，send_audio_frame(麦克风 3200 帧块)
    ASR-->>AP: Callback.on_event：文本含 "hi, alicia" → wake_up=True
    loop while not rospy.is_shutdown()
        M->>AP: 读取 callback.wake_up
        opt 已唤醒
            M->>TTS: synthesize_and_play("I heard you...")
            M->>M: detect_object()：wait_for_message 取 /camera/color/image_raw、/camera/depth/image_raw、camera_info
            M->>OD: detect_green_cube(rgb)
            OD->>VLM: inference_with_api（base64 图像 + 提示词，要求输出 JSON 框）
            VLM-->>OD: label + bbox_2d → extract_center_coordinates
            M->>TTS: 播报检测到的物体数与标签
            M->>CT: pixel_to_camera_pose(center_coordinates, depth, camera_info)
            CT-->>M: camera_pose / world_pose（用 yaml 外参变换）
            M->>AP: asr(execution_time=6.0)，语音指令存入 latest_command
            M->>AP: callback.object_to_grasp(latest_object_pose)
            loop 每个选中物体
                M->>M: grasp(world_pose, label)
                M->>MC: gripper_control(0.0)，move_to_pose(预抓取位姿，z+0.05)
                M->>MC: move_to_pose(下压 0.05)，gripper_control(2.0)
                M->>MC: move_to_joint_state(place_pose)，gripper_control(0.0)
            end
            M->>M: callback.wake_up=False，播报 "Task complete"
        end
    end
```

图依据：Synria-Robotics/Alicia-D-VLM-Grasp@5350d37，读过 README_Ch.md、vlmgrasp/vlm_grasp.py（全文）、speech_recog.py（on_event、object_to_grasp 部分）、object_detection.py（detect_green_cube 及函数签名）、tts.py 与 camera_trans.py（仅函数签名/导入）；`moveit_control.MoveItRobotController` 来自仓库外的 `~/alicia_ws/src/alicia_duo_moveit/scripts`，未读其实现。

## 关联页面

- [国内具身开源全景技术地图](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md)
- [HMI 开源项目主表导读](../queries/hmi-opensource-projects-coverage.md)
- [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md)

## 参考来源

- [Alicia-D-VLM-Grasp 源码归档](../../sources/repos/alicia-d-vlm-grasp.md)（<https://github.com/Synria-Robotics/Alicia-D-VLM-Grasp>）

- [国内具身智能开源全景（微信公众号）](../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)
- 源码核对（2026-11-02）：Synria-Robotics/Alicia-D-VLM-Grasp@5350d37，读过 README_Ch.md、vlmgrasp/vlm_grasp.py、vlmgrasp/speech_recog.py（部分）、vlmgrasp/object_detection.py（部分）、vlmgrasp/tts.py 与 camera_trans.py（仅签名/导入）；MoveIt 控制类在仓库外未读；时序图依据以上文件，未运行代码。

## 推荐继续阅读

- [玄雅科技 官方组织](https://github.com/Synria-Robotics)
