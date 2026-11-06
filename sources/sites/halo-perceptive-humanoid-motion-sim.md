# HALO 项目页（apple-aiml-research.github.io/ml-halo）

- **URL：** <https://apple-aiml-research.github.io/ml-halo/>
- **论文：** [halo_perceptive_humanoid_motion_sim_arxiv_2611_06908.md](../papers/halo_perceptive_humanoid_motion_sim_arxiv_2611_06908.md)
- **代码：** <https://github.com/apple-aiml-research/ml-halo> — [sources/repos/halo-perceptive-humanoid-motion-sim.md](../repos/halo-perceptive-humanoid-motion-sim.md)
- **入库日期：** 2026-11-06

## 一句话说明

Apple HALO 官方项目页：机载栈时间线动画、画廊视频、数据管线图解、G1 的 3D 仿真查看器、系统图。

## 开源核查（2026-11-06）

- `curl` 返回 **HTTP 200**。
- 页内 Paper 导航按钮标 `Coming soon` / `Soon`、Code 按钮亦标 `Coming soon`（`aria-disabled`）；但页脚有 `github.com/apple-aiml-research/ml-halo` 链接，且该仓库已可 `git ls-remote` / `git clone`。**页面状态落后于仓库与 arXiv**。
- 页面文字摘要（作者表述）：编排器约每 2 s 选下一个子任务；生成器每 1/3 s 重规划 5 s 全身运动（150 帧 × 36 值），前 0.5 s（15 帧）作为干净 token 回灌；跟踪器 50 Hz 输出 29 关节目标；演示里的 Sit 任务成功率开关为 84%（RTC 关闭 / 推理时 / 训练时对比，对应论文 Table III）。
- 训练流程图：Kimodo 生成 → Infinigen 1,000 间 → 鱼眼渲染 → 生成器（250 H100 造数据 / 64 B200 训练）与跟踪器（4 H100）。

## 对 wiki 的映射

- [paper-halo-perceptive-humanoid-motion-sim](../../wiki/entities/paper-halo-perceptive-humanoid-motion-sim.md)
