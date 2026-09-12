# UMR（官方实现）

> 来源归档

- **标题：** UMR — Unified Motion Retargeting for Humanoids with Learned Point Cloud Correspondence
- **类型：** repo（官方）
- **链接：** <https://github.com/hanyang9/UMR>
- **项目页：** <https://hanyang9.github.io/UMR/umr_project.html>
- **UMR Studio：** <https://hanyang9.github.io/UMR/umr_studio.html>
- **论文：** [arXiv:2609.02134](https://arxiv.org/abs/2609.02134)
- **机构：** HKUST-GZ；Noitom Robotics；Hanyang University；HKUST；HKU
- **入库日期：** 2026-09-12
- **一句话说明：** 官方仓：T-pose 点云对应学习 + 约束 Gauss-Newton QP 重定向；MuJoCo + Clarabel；多 motion source adapter 与批处理；`humanoid_retarget_pipeline*.py` 系列入口。
- **沉淀到 wiki：** 是 → [`wiki/entities/paper-umr-unified-motion-retargeting.md`](../../wiki/entities/paper-umr-unified-motion-retargeting.md)

## 摘录要点

### 安装与依赖

- Conda `python=3.12`；PyTorch 2.4.1（cu121）；`requirements-umr.txt`。
- SMPL-X 模型需自行下载至 `smpl/SMPLX_NEUTRAL.pkl` 或 `.npz`（及 male/female 按需）。
- NR FBX/BVH 路径需 Node.js ≥ 18（仅 FBX mesh 解析）。

### 主要入口脚本

| 脚本 | 用途 |
|------|------|
| `scripts/humanoid_retarget_pipeline.py` | SMPL-X / LAFAN1 默认 quick start |
| `scripts/humanoid_retarget_pipeline_character.py` | MimicKit humanoid character |
| `scripts/humanoid_retarget_pipeline_hsi_hoi.py` | GRAIL / OmniContact / OMOMO 交互 |
| `scripts/humanoid_retarget_pipeline_nr.py` | NR FBX/BVH |
| `scripts/humanoid_retarget_pipeline_adapt.py` | AdaPT body+racket |
| `scripts/humanoid_retarget_pipeline_batch.py` | 批处理 + DP warm start |
| `scripts/visualize_robot_retarget_result.py` | MuJoCo viewer 回放 |

### 新机器人 onboarding

1. 在 [UMR Studio](https://hanyang9.github.io/UMR/umr_studio.html) 加载 MJCF、调 T-pose、**Copy T-pose Config**。
2. 复制 `robot_configs/humanoid_retarget_unitree_g1_example.json`，粘贴 `tpose_qpos`。
3. 运行对应 pipeline；**无需手写 human–robot 映射**。

### 已知边界（README）

- OmniContact 官方 G1 数据为 BVH；仓内 **不含** 内部 BVH→SMPL-X 转换器。
- GRAIL 运行时对 neutral SMPL-X 做 bundled G1 模板 overlay；衍生 baked 权重不随仓分发。
- HSI/HOI 凹物体需 [CoACD](https://github.com/SarahWeiii/CoACD) 凸分解后再进 MuJoCo 碰撞。

## 对 wiki 的映射

- 论文实体：[paper-umr-unified-motion-retargeting.md](../../wiki/entities/paper-umr-unified-motion-retargeting.md)
- 项目页：[umr-project.md](../sites/umr-project.md)
- 交叉： [GMR](../../wiki/methods/motion-retargeting-gmr.md)、[BeyondMimic](../../wiki/methods/beyondmimic.md)、[AdaPT](../../wiki/entities/paper-adapt.md)
