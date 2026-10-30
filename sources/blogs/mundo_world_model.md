# Mundo: From seeing a place to simulating what happens there

> 来源归档（官方技术博客）

- **原文：** <https://huggingface.co/blog/Luo-Yihang/mundo-world-model>
- **发布日期：** 2026-10-23；**核查日期：** 2026-10-30
- **阅读范围：** 全文，包括构建、仿真、渲染、训练、应用与局限。
- **项目主页与开放状态：** [项目页归档](../sites/mundo-world-model.md)
- **仓库：** [官方仓库归档](../repos/mundo.md)

## 核心摘录（归纳）

- 3D 基础模型恢复几何与相机位姿，检测器给出初始框；编辑 VLM 与独立审阅 agent 通过多视图渲染比较和数值拟合迭代修正代理。
- 场景程序存储物体、位姿及物理属性，多数示例用 MuJoCo 计算接触与运动；代理帧和输入视图共同条件化视频扩散模型。
- 渲染器以约 4 万对真实或渲染视频及对应代理视频微调 Ref2VA；训练片段为 124 帧、640×480。
- 局限包括速度、小物体接触的漂浮与幻觉，以及缺少长时记忆。

## 对 wiki 的映射

- [Mundo 实体](../../wiki/entities/mundo-world-model.md)
