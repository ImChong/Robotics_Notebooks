# RoboDojo-Mirror 公开镜面场景评测代码 / Public Mirror Scene Evaluation Adapter

- **项目主页：** https://ipplab.github.io/robodojo-mirror/
- **代码：** [iPPLab/robodojo-mirror](https://github.com/iPPLab/robodojo-mirror)
- **核对版本：** [fe311cef196afe155b066f487171ce2256a9c2cb](https://github.com/iPPLab/robodojo-mirror/tree/fe311cef196afe155b066f487171ce2256a9c2cb)
- **许可证：** [MIT](https://github.com/iPPLab/robodojo-mirror/blob/fe311cef196afe155b066f487171ce2256a9c2cb/LICENSE) 适用于本包；外部 RoboDojo/Isaac、上游策略检查点、历史 runner 和资产分别受各自条款约束。
- **核对范围：** 官方主页、固定提交的 README、复现说明、启动器、环境接入、RoboDojo 后端及 case/指标入口；这是源码静态核对，未安装 Isaac 或运行模拟评测。
- **通读：** 已完成

- [README](https://github.com/iPPLab/robodojo-mirror/blob/fe311cef196afe155b066f487171ce2256a9c2cb/README.md) 指定保留上游 evaluator、policy service 和原生计分。先用 `check-source` 检查接口；在自己的 evaluator 副本中，于 AppLauncher 前调用 `configure_launcher(args_cli)`，环境构造后且首次 reset 前 `attach(env, MirrorConfig(...))`。
- [integration.py](https://github.com/iPPLab/robodojo-mirror/blob/fe311cef196afe155b066f487171ce2256a9c2cb/src/robodojo_mirror/integration.py) 中 `attach` 安装 `MirrorController`；reset 先清理旧状态，再走原生 reset，最后后端 `prepare`；`get_obs` 按需返回 RGB（或 state-only），相机捕获期间不允许推进物理，正常物理子步后更新后端记录。
- [复现说明](https://github.com/iPPLab/robodojo-mirror/blob/fe311cef196afe155b066f487171ce2256a9c2cb/docs/reproduction.md) 区分历史 `paper` 位置版本与后来修正的双镜放置：新版两块镜板沿各自支撑平面法向独立偏移，结果不得混入原表。计划生成和日志聚合可用公开入口；原始运行记录、模型权重和部分诊断运行器是外部依赖。作者在准备好的本地环境做过配对 smoke test，不等于验证了全新部署环境。

- **Wiki：** [RoboDojo-Mirror](../../wiki/entities/paper-robodojo-mirror.md)
