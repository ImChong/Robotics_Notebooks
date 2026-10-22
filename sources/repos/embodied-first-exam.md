# Embodied First Exam（官方代码仓）

> 来源归档

- **标题：** Embodied First Exam
- **类型：** repo + benchmark suite + Harbor integration
- **代码：** <https://github.com/mll-lab-nu/embodied-first-exam>
- **项目页：** <https://www.embodied-first-exam.ai/>
- **评测框架：** [Harbor](https://github.com/laude-institute/harbor)
- **License：** MIT（仅代码仓；第三方模拟器、资产、参考解和基础镜像各自保留原许可证）
- **发布：** 2026-10-19（仓库 README News）
- **入库日期：** 2026-10-22
- **一句话说明：** 面向 coding agent 的具身任务考核：agent 在冻结的机器人任务场景中编写程序、运行并查看结果，再由独立沙箱验证；官方 README 列出 32 个基准、核心配置 449 个任务，完整配置为 2,128 个任务目录。

## 仓库入口

| 资源 | 路径 / 命令 | 说明 |
|------|-------------|------|
| 任务集合 | tasks/ | 多个机器人 benchmark 的任务与模式 |
| Harbor 插件 | embodied_first_exam/ | Harbor 数据集 / 运行集成 |
| 安装 | uv tool install --force -e . --with-executables-from harbor | 从源码安装，并安装仓库指定的 Harbor 命令 |
| 核心评测 | harbor run -d embodied-first-exam -a codex -m <model> | 默认模式运行核心任务 |
| 全量评测 | harbor run -d embodied-first-exam@all -a codex -m <model> | 使用完整任务集合 |

## 环境与边界

- 官方 Quick Start 面向 Linux x86_64，需要 Git、uv、Docker Engine（Compose v2 / BuildKit）；无 Docker 的场景可按官方运行文档改用 rootless Podman。
- GPU 需求因 benchmark 而异；README 同时列出可无 GPU 运行的任务与需要 NVIDIA GPU 的任务。
- 首次运行可能下载基准资产或构建镜像；磁盘空间、驱动、数据集许可及上游资产许可需逐项确认。
- 仓库 MIT 许可不自动覆盖第三方模拟器、资产、参考解决方案或基础镜像。

## 对 wiki 的映射

- 实体页：[Embodied First Exam](../../wiki/entities/embodied-first-exam.md)
- 官方站归档：[embodied-first-exam-ai.md](../sites/embodied-first-exam-ai.md)
- 对照：[RLE-Bench](../../wiki/entities/rle-bench.md) — 同样通过 Harbor 评 coding agent，但任务范围与任务设计不同
