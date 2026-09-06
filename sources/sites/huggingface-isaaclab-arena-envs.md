# Hugging Face — nvidia/isaaclab-arena-envs（LeRobot EnvHub）

- **标题：** IsaacLab-Arena Environments（EnvHub）
- **类型：** site / Hugging Face Hub 数据集与环境注册
- **链接：** https://huggingface.co/nvidia/isaaclab-arena-envs
- **组织：** NVIDIA (`nvidia`)
- **入库日期：** 2026-09-06
- **一句话说明：** Arena 官方 EnvHub 入口：把 GPU 加速的 Isaac Lab 仿真环境接到 LeRobot `lerobot-eval`，供社区发现、共享与评测通才策略。
- **代码：** 环境定义依赖 [IsaacLab-Arena](https://github.com/isaac-sim/IsaacLab-Arena)（已开源）；Hub 仓本身为环境注册与元数据
- **沉淀到 wiki：** 是 → [`wiki/entities/isaac-lab-arena.md`](../../wiki/entities/isaac-lab-arena.md)

---

## 示例环境（页面表格，2026-09-06）

| Environment ID | 描述 |
|----------------|------|
| `gr1_microwave` | 伸手打开微波炉 |
| `galileo_pnp` | 抓取物体并放到目标位 |
| `g1_locomanip_pnp` | G1 从货架取箱放到桌侧蓝桶（loco-manipulation） |
| `kitchen_pnp` | 厨房物体操作 |
| `press_button` | 定位并按压按钮 |

社区可基于 Arena 框架自建环境并发布到 Hub。

---

## `lerobot-eval` 快速命令（页面原文）

```bash
lerobot-eval \
    --policy.path=nvidia/smolvla-arena-gr1-microwave \
    --env.type=isaaclab_arena \
    --env.hub_path=nvidia/isaaclab-arena-envs \
    --rename_map='{"observation.images.robot_pov_cam_rgb": "observation.images.robot_pov_cam"}' \
    --policy.device=cuda \
    --env.environment=gr1_microwave \
    --env.embodiment=gr1_pink \
    --env.object=mustard_bottle \
    --env.headless=false \
    --env.enable_cameras=true \
    --env.video=true \
    --env.video_length=10 \
    --env.video_interval=15 \
    --env.state_keys=robot_joint_pos \
    --env.camera_keys=robot_pov_cam_rgb \
    --trust_remote_code=True \
    --eval.batch_size=1
```

---

## 相关链接

- [LeRobot](https://github.com/huggingface/lerobot)
- [EnvHub 文档](https://huggingface.co/docs/lerobot/envhub)
- [Isaac Lab](https://github.com/isaac-sim/IsaacLab)
- NVIDIA 集成博文：<https://huggingface.co/blog/nvidia/generalist-robotpolicy-eval-isaaclab-arena-lerobot>

---

## 对 wiki 的映射

- [Isaac Lab-Arena](../../wiki/entities/isaac-lab-arena.md)
- [LeRobot](../../wiki/entities/lerobot.md)
- [LW BENCHHUB TOUR](../../wiki/entities/lw-benchhub-tour.md)
