# GraspKit 官方源码归档

- **类型：** repo / grasp synthesis / imitation learning
- **链接：** https://github.com/openrobokit/graspkit
- **文档：** https://github.com/openrobokit/graspkit/blob/main/docs/guide/pipeline.md
- **概览：** https://github.com/openrobokit/graspkit/blob/main/docs/guide/overview.md
- **源码入口：** https://github.com/openrobokit/graspkit/blob/main/scripts/launch.py
- **关联项目页：** https://openrobokit.com/graspkit/
- **许可证：** MIT（官方仓库 README/Licence 标识）
- **核查日期：** 2026-11-07
- **核对范围：** 官方 README、docs/guide/overview.md、docs/guide/pipeline.md、完整的 scripts/launch.py。
- **通读：** 已完成

## 工程事实

- 默认实验阶段为 `render_data`、`optimize_grasps`、`process_sim_data`、`train`。
- 优化生成抓取候选及轨迹，物理仿真筛掉失败候选；渲染点云后按 scene ID 拼接监督数据，再训练 CVAE 抓取策略。
- `scripts/launch.py` 用 `cli_experiment` 解析实验、`resolve_gpus` 分配 GPU，分阶段调度独立 worker；训练路径通过单 GPU 脚本或多 GPU `torchrun` 运行。
- `restart-from` 清理指定阶段及其后输出标记；`append` 保留重启阶段已有输出。资产、CUDA 以及 ManiSkill/spconv 等依赖要按官方版本准备。
- 可运行代码已公开，但没有在当前环境执行 GPU 训练、真机部署或性能复现。

## 对 wiki 的映射

- [GraspKit](../../wiki/entities/graspkit.md)
