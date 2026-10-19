# RecursiveGameCreator

> 来源归档；核查日期：2026-10-19。

- **标题：** Recursive Game Creator
- **类型：** repo
- **来源：** IMBALDY（GitHub）
- **链接：** <https://github.com/IMBALDY/RecursiveGameCreator>
- **论文：** <https://arxiv.org/abs/2610.08621> → [论文归档](../papers/recursive_game_creator.md)
- **项目页：** <https://imbaldy.github.io/recursive-game-creator/> → [项目页归档](../sites/recursive-game-creator.md)
- **Hugging Face：** <https://huggingface.co/papers/2610.08621>（论文索引）
- **许可证：** MIT
- **入库日期：** 2026-10-19
- **一句话说明：** 用多角色 coding-agent harness 规划、实现、执行游戏策略、审阅游玩体验并保留改进版本。
- **代码：** <https://github.com/IMBALDY/RecursiveGameCreator>（**已开源**）
- **Wiki：** [Recursive Game Creator](../../wiki/entities/paper-recursive-game-creator.md)

## 运行入口

README 要求 macOS 或 Linux、Git、Node.js 22/npm、已运行的 Docker daemon；Python 3.12 和项目工具由 setup 脚本管理。标准流程为 clone 仓库后运行 **sh scripts/setup.sh**，激活 **.venv**，再用 **game-agent start examples/quickstart-html.json** 启动 HTML quick start。仓库还提供 Godot quick start、状态查看、恢复和游戏导出命令。

项目可用 minimal 模式仅安装 Python 依赖并运行离线 mock 测试；真实生成需要配置模型认证。首次安装会下载依赖并构建镜像。

## 开放范围

仓库源码及文档公开，GitHub 元数据标注 MIT。论文、项目站及仓库 README 的身份一致；本次只核对文档和元数据，没有实际运行 setup 或端到端实验。
