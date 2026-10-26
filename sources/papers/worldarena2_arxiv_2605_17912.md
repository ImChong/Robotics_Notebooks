# WorldArena 2.0: Extending Embodied World Model Benchmarking on Modality, Functionality and Platform（arXiv:2605.17912）

> 来源归档（论文）

- **标题：** WorldArena 2.0: Extending Embodied World Model Benchmarking on Modality, Functionality and Platform
- **arXiv：** <https://arxiv.org/abs/2605.17912>
- **项目页：** <https://world-arena.ai/>
- **官方代码：** <https://github.com/WorldArena2/WorldArena-2.0>
- **榜单：** <https://huggingface.co/spaces/WorldArena/WorldArena2.0>
- **作者：** Yu Shang, Yinzhou Tang, Yiding Ma, Zhuohang Li, Lei Jin, Weikang Su, Xin Jin, Zhaolu Wang, Ziyou Wang, Xin Zhang, Haisheng Su, Weizhen He, Wei Wu, Haoyi Duan, Gordon Wetzstein, Xihui Liu, Dhruv Shah, Zhaoxiang Zhang, Zhibo Chen, Jun Zhu, Yonghong Tian, Tat-Seng Chua, Wenwu Zhu, Chen Gao, Yong Li
- **开源状态：** 官方 GitHub 仓库公开，标注 MIT；Hugging Face Space 展示榜单
- **入库日期：** 2026-10-26
- **一句话说明：** 在 WorldArena 1.0 基础上增加视触觉评测、在线 RL 环境用途和模拟/真机跨平台评估。

## 摘录：三条扩展轴

1. **模态：** 从视觉扩展到视触觉，评测多模态感知和预测。
2. **功能：** 从离线策略评估与规划，扩展到将世界模型作为在线交互 RL 环境，检验其支持策略优化的效用。
3. **平台：** 从仿真扩展到模拟与真机、多种具身平台，检验跨平台表现与 sim-to-real。

项目页概述评测覆盖 RoboTwin 2.0、LIBERO 与 AgileX split-type ALOHA；具体子任务、指标和复现配置应以代码仓库各分项文档为准。

**对 wiki 的映射：** 更新 [WorldArena 实体页](../../wiki/entities/worldarena.md)，保留 1.0 与 2.0 的版本差异。

## 小米模型与基准的关系

Hugging Face Space 的公开榜单收录 Xiaomi-Robotics-U0 等模型。这表示小米模型参加该基准评估；基准论文和代码由 WorldArena 研究团队维护，不应将 WorldArena 2.0 写成小米自研项目。
