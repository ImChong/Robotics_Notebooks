# APTree 官方仓库

> 来源归档（repo）

- **标题：** shshermin/APTree-automated-planning-tree
- **类型：** repo / paper implementation
- **代码：** <https://github.com/shshermin/APTree-automated-planning-tree>
- **论文：** <https://arxiv.org/abs/2610.18523>（[sources/papers/aptree_arxiv_2610_18523.md](../papers/aptree_arxiv_2610_18523.md)）
- **项目页：** 无
- **机构：** 斯图加特大学（ICD/CA、ISW）
- **入库日期：** 2026-10-17
- **一句话说明：** APTree DSL（MontiCore）、C# 行为树执行引擎、Python 规划服务与编辑器前端的合集，带 Docker 运行说明。

## 开源状态（核查日 2026-10-17）

- **已开源**：273 次提交，默认分支 `main`，1 star；未显示许可证；无 Releases。
- 目录：`APTreeDSL/`、`APTreeExecutionEngine/`（C#，含 `python_service/`）、`APTreeEditor/`（含 `BehaviorTreeFrontend`）、`InputGenerator/`、`docker/`；根下有 `Dockerfile`、`docker-compose.yml`、`DOCKER_README.md`、`BehaviorTreeMainProject.sln`、`DemonstratorFinal.bt`、`DemonstratorFaults.json`。
- `DOCKER_README.md`：需把 `enhsp.jar` 放到 `APTreeExecutionEngine/python_service/`；`docker build -t aptree:latest .`；`docker run -p 127.0.0.1:5000:5000 -p 5254:5254 aptree:latest`；Python 规划服务在 5000，ASP.NET 后端在 5254，均有 `/health`；DSL 校验接口 `/api/aptree/validate`；规划服务无认证，应只绑定回环地址，`PLANNER_DEBUG=1` 会开启可远程执行代码的调试器。
- 本次未在本地运行。

## 对 wiki 的映射

- [wiki/entities/paper-aptree.md](../../wiki/entities/paper-aptree.md)
