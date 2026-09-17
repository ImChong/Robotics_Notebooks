# DexCap（项目页）

> 来源归档（ingest · 2026-09-17）

- **标题：** DexCap: Scalable and Portable Mocap Data Collection System for Dexterous Manipulation
- **类型：** site / project-page
- **官方入口：** <https://dex-cap.github.io/>
- **论文：** <https://arxiv.org/abs/2403.07788>
- **代码：** <https://github.com/j96w/DexCap>
- **数据集：** <https://huggingface.co/datasets/chenwangj/DexCap-Data>
- **会议：** RSS 2024
- **入库日期：** 2026-09-17
- **一句话说明：** 可穿戴 SLAM+EMF 手套 mocap 与胸挂 RGB-D 点云观测，配合 DexIL（指尖 IK + 点云 Diffusion Policy）把野外人类灵巧示范迁移到 LEAP Hand 双臂平台。
- **开源状态（2026-09-17 核查）：** **已开源**（MIT）— 项目页与 GitHub 均列代码与 Hugging Face 数据集；仓库含采集（NUC/Windows）、处理、HDF5 构建与 robomimic 训练全链路。

## 页面公开信息

| 资源 | URL |
|------|-----|
| 项目页 | <https://dex-cap.github.io/> |
| 代码 | <https://github.com/j96w/DexCap> |
| 原始/处理后数据 | <https://huggingface.co/datasets/chenwangj/DexCap-Data> |
| HF Paper | <https://huggingface.co/papers/2403.07788> |
| 硬件更新教程（2024-08） | [Google Doc](https://docs.google.com/document/d/1ANxSA_PctkqFf3xqAkyktgBgDWEbrFK7b1OnJe54ltw/edit) |

## 核心摘录（策展）

1. **DexCap 硬件：** 胸挂 RGB-D LiDAR + 3 路 SLAM 追踪相机；背包容 mini-PC 与电源（约 40 分钟采集续航）；标定后 SLAM 相机移至手背支架追踪掌位，Rokoko EMF 手套测相对掌系的指尖 3D 位置。
2. **相对视觉 teleop 的优势：** 项目页强调遮挡场景（如握杯柄）下 VR 视觉手追踪易失败，而 EMF+SLAM 组合更稳。
3. **吞吐：** 宣称约为 teleoperation 的 **3×** 数据收集速度，接近自然人类动作节奏。
4. **DexIL：** 点云观测 + 指尖 IK 重定向到 LEAP Hand；可选 FK 生成机器人手点云 mesh 补视觉 gap；Diffusion Policy 预测 20 步、46 维双臂+双手动作。
5. **人机闭环修正：** rollout 时脚踏切换 **residual 腕部修正** 与 **全手 teleop IK** 两种模式；修正数据与原数据均匀采样 fine-tune。
6. **评测：** 六项灵巧任务 + 双手泡茶/剪刀等；30 分钟 mocap 即可无 teleop 自主 rollout；in-the-wild 采集可泛化到未见物体。

## 对 wiki 的映射

- [`wiki/entities/paper-notebook-dexcap-scalable-and-portable-mocap-data-collecti.md`](../../wiki/entities/paper-notebook-dexcap-scalable-and-portable-mocap-data-collecti.md)
- [`sources/repos/dexcap.md`](../repos/dexcap.md)
- [`sources/papers/humanoid_pnb_dexcap-scalable-and-portable-mocap-data-collecti.md`](../papers/humanoid_pnb_dexcap-scalable-and-portable-mocap-data-collecti.md)
