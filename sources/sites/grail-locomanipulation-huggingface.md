# PhysicalAI-Robotics-Locomanipulation-GRAIL（Hugging Face）

- **标题:** PhysicalAI-Robotics-Locomanipulation-GRAIL
- **类型:** dataset / huggingface
- **链接:** <https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-Locomanipulation-GRAIL>
- **论文:** [GRAIL（arXiv:2606.05160）](https://arxiv.org/abs/2606.05160)
- **项目页:** <https://research.nvidia.com/labs/dair/grail/>
- **代码:** <https://github.com/NVlabs/GRAIL>
- **文档:** <https://NVlabs.github.io/GRAIL/>
- **机构:** NVIDIA
- **收录日期:** 2026-06-30

## 一句话摘要

NVIDIA 在 Hugging Face 发布的 **GRAIL 全合成 G1 loco-manipulation 轨迹集**（约 **250 GB**）：每条运动含合成视频、SMPL-X+物体 6-DoF 的 4D HOI 重建、经 SONIC tracker 在 Isaac Lab 中物理可行的 **G1 机器人轨迹 + 物体轨迹**、元数据与 OpenUSD 物体资产；按 HOI 类别分目录，另附 GEM-SMPL / FoundationPose / SONIC 子模块 checkpoint。

## 规模与类别（官方统计）

| 类别 | 含义 | 3D 资产来源 | # 物体 | # 运动 | 序列时长 | 总帧数 |
|------|------|-------------|--------|--------|----------|--------|
| `pickup_table` | 桌面拾取 | RoboCasa 衍生 | 685 | 2,991 | 10 s | 747,750 |
| `pickup_ground` | 地面拾取 | RoboCasa 衍生 | 631 | 1,613 | 15 s | 611,625 |
| `sitting` | 坐姿交互 | Hunyuan3D 生成 | 189 | 1,748 | 5 s | 218,500 |
| `slope` | 坡道行走 | 程序化地形 | 200 | 1,880 | 10 s | 470,000 |
| `curb` | 路缘跨越 | 程序化地形 | 200 | 1,769 | 10 s | 442,250 |
| `stair` | 楼梯上下 | 合成+真实楼梯资产 | 4,952 | 12,188 | 10 s | 3,047,000 |

> 桌面/地面 **manipulation** 类别计划后续发布。

## 每条运动的模态

| 路径模式 | 内容 |
|----------|------|
| `video/<seq>.mp4` | 源合成 HOI 视频（24 fps） |
| `recon/<seq>.pkl` | 4D HOI 重建（SMPL-X + 物体 6-DoF） |
| `robot/<seq>.pkl` | post-RL **G1 轨迹**（29 body DOF + `hand_dof_pos`） |
| `objects/<seq>.pkl` | post-RL **物体 6-DoF**（xyz + quat） |
| `meta/<seq>.pkl` | 长度、接触标志、源 ID 等元数据 |
| `object_usd/<seq>.usd` | OpenUSD 物体资产（含 textures） |

- **发布轨迹采样率：** 25 Hz
- **机器人平台：** Unitree G1（29 body + 7×2 hand DOF）
- **重建人体模型：** SMPL-X（75 body + 45×2 hand DOF）

## 目录布局（摘要）

```
data/<hoi_category>/
├── video/ recon/ robot/ objects/ meta/ object_usd/
checkpoint/
├── GEM-SMPL/          # 人体姿态估计权重
├── FoundationPose/    # 物体 6-DoF 估计
└── SONIC/models/      # tracking checkpoint
```

## 许可要点

- **GRAIL 原创输出**（轨迹、4D 重建、程序化/Hunyuan3D 生成资产）：**Apache 2.0**
- **RoboCasa 衍生物体**（pickup 类）：**CC BY 4.0**，再分发需署名 RoboCasa Team
- **ComAsset 衍生**（advanced manipulation 计划类）：**ODC-By v1.0**
- **bundled checkpoint/**：NVIDIA Open Model License + 各上游第三方许可（HMR2、ViTPose、SMPL-X 等）

## 典型用途

1. **人形 tracker / IL 监督** — 用 `robot/` + `objects/` 作为已物理可行的 G1 参考轨迹
2. **4D HOI 重建研究** — 配对（合成视频, SMPL-X recon, 物体 pose, 重定向人形轨迹）
3. **Sim-to-real** — 作为可部署控制器的运动学参考，或残差策略的 kinematic target

## 对 Wiki 的映射

- [grail-locomanipulation-dataset](../../wiki/entities/grail-locomanipulation-dataset.md) — 数据集实体页
- [paper-grail](../../wiki/entities/paper-grail.md) — GRAIL 论文与生成管线

## 参考来源（原始）

- Hugging Face：<https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-Locomanipulation-GRAIL>
- 项目页：<https://research.nvidia.com/labs/dair/grail/>
