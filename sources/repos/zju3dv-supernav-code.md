# zju3dv/SuperNav

> 来源归档

- **标题：** SuperNav（官方实现）
- **类型：** repo
- **组织：** zju3dv
- **代码：** <https://github.com/zju3dv/SuperNav>
- **论文：** <https://arxiv.org/abs/2610.12126>
- **项目页：** <https://zju3dv.github.io/SuperNav/>
- **入库日期：** 2026-10-14
- **最近核对：** 2026-10-26
- **一句话说明：** 提供 agent runtime、面向导航 agent 的工具与 Skills、Habitat-GS / AI2-THOR 实验配置和评测工具；官方仓库还链接 Learned Executor 权重、HabitatGS-Benchmark 数据及已发布的 Unitree Go2 部署代码。

## 主要入口

| 内容 | 用途 |
|------|------|
| `supernav web --demo` | 本地浏览器演示 |
| `configs/experiments/` | 选择 agent、后端和实验设置 |
| Agent runtime / Tools / Skills | 承载循环决策、视觉点移动、进度管理和搜索/恢复指南 |
| Habitat-GS 与 AI2-THOR | 仿真运行和评估入口 |
| [Learned Executor 权重](https://huggingface.co/the0xka1/SuperNav-Learned-Executor) | 学习式执行后端权重 |
| [HabitatGS-Benchmark](https://huggingface.co/datasets/yyh929/HabitatGS-Benchmark) | 已发布的 InteriorGS 单物体/多物体导航测试任务 |
| [Unitree Go2 部署指南](https://github.com/zju3dv/SuperNav/blob/main/docs/release/en/go2.md) | Jetson 上的四路 GoPro RGB 与 LiDAR 地图支持的 PixelGoal 部署 |
| [InteriorGS 数据集](https://huggingface.co/datasets/spatialverse/InteriorGS) | 原始场景数据入口 |

## 复现边界

- 官方 README 要求 Python 3.12；仿真数据、模型服务和任务配置需按具体实验单独准备。
- 官方仓库于 2026-10-21 发布 Unitree Go2 部署指南：Jetson aarch64、四路 GoPro HERO8 USB 网络相机，并结合 LiDAR 地图执行 map-backed PixelGoal 导航。部署代码需按指南组装公开 DimOS 子模块及补丁；项目页的演示视频和可复现部署入口应分别理解。
- 高层任务完成判断由 MLLM 提出，benchmark 的成功评估独立计算；不要把 agent 的关闭会话等同于任务成功。

## 与本仓库知识的关系

| 页面 | 关系 |
|------|------|
| [SuperNav](../../wiki/entities/paper-supernav-agentic-navigation.md) | 论文实体与系统方法、结果归纳 |
| [视觉–语言导航](../../wiki/tasks/vision-language-navigation.md) | 任务与 agentic 导航路线比较 |
| [Uni-LaViRA](../../wiki/entities/paper-uni-lavira.md) | 多任务零样本具身导航对照 |
| [Qwen-RobotNav](../../wiki/entities/qwen-robot-nav.md) | 统一多任务导航模型路线对照 |

## 对 wiki 的映射

- 论文摘录：[`sources/papers/supernav_arxiv_2610_12126.md`](../papers/supernav_arxiv_2610_12126.md)
- 项目页：[`sources/sites/zju3dv-supernav.md`](../sites/zju3dv-supernav.md)
- 详情页：[`wiki/entities/paper-supernav-agentic-navigation.md`](../../wiki/entities/paper-supernav-agentic-navigation.md)
