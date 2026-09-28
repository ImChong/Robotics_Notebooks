# KIWI（Kinematic Interface for the Wild）

> 来源归档

- **标题：** KIWI — Kinematic Interface for the Wild
- **类型：** site / project-page
- **链接：** <https://lingfeng.moe/KIWI/>
- **论文：** <https://arxiv.org/abs/2609.22809>
- **机构：** Autel US（论文 affiliation）
- **入库日期：** 2026-09-28
- **一句话说明：** 仅消费级 360° 相机（Insta360 X5/GO 3）的双臂野外示教套件：后镜头建共享 metric map、前镜头录操作；离线 IMU/音频同步；导出 6-DoF 双手/头轨迹、夹爪开度与 3DGS 场景。
- **开源状态（步骤 2.5，2026-09-28 核查）：** 项目页按钮为 **Code (coming soon)**；论文摘要写 hardware/software **将 fully open-source on website** → **待发布**（尚无 GitHub URL）。

---

## 页面公开资源

| 资源 | URL / 状态 |
|------|------------|
| 项目首页 | <https://lingfeng.moe/KIWI/> |
| arXiv | <https://arxiv.org/abs/2609.22809> |
| 代码 / 硬件 | **Coming soon**（页眉无 GitHub 链） |
| 联系 | lingfengsun1996@gmail.com |

## 首页核心数字（项目页摘录）

| 指标 | 数值 |
|------|------|
| 跨手定位失败帧占比（六条双臂录制） | **0.1%**（仅前镜头跨手对齐：**24.8%**，且丢整条录制） |
| 相对评测 fiducial 中位定位误差 | **4.5 mm** |
| 硬件 | 2× Insta360 X5（四镜头 + 单次 demo 重建全场景） |
| 多机同步 | 环境音频，毫秒级 |

## 三阶段管线（Capture → Reconstruct → Export）

1. **Record** — 双手腕双镜头相机 + 可选头戴 ego（GO 3）。
2. **Recover** — 视觉地图 + 同步音频 + IMU；地面/桌面平面估计。
3. **Export** — 同步视频、工具尖端位姿、夹爪开度 + **3D Gaussian Splat** 场景。

## 模块化硬件（页内）

- 快装 Arca-Swiss 改版：相机模块可在 **筷子夹爪 / 平行爪（Soon）/ 腕带（Soon）** 与 **Franka / YAM / OpenArm** 机器人法兰侧互换。
- 筷子夹爪：被动、指驱、轴承导向；保留机械反馈。

## 对 wiki 的映射

- 论文实体：**`wiki/entities/paper-kiwi-kinematic-interface-wild.md`**
- 论文摘录：**`sources/papers/kiwi_arxiv_2609_22809.md`**
- UMI 范式对照：[HiFi-UMI](../../wiki/entities/paper-hifi-umi.md)、[HandUMI](../../wiki/entities/handumi.md)
