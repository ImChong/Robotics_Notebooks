# CodeActionBench

> 来源归档（repo）

- **标题：** CodeActionBench
- **类型：** repo
- **链接：** <https://github.com/lyhkk/CodeActionBench>
- **项目页：** <https://codeactionbench.org/>
- **论文：** <https://arxiv.org/abs/2609.33807>
- **机构：** 南洋理工大学（NTU）；独立研究者
- **许可证：** MIT（代码）；数据集计划 CC BY 4.0
- **入库日期：** 2026-10-15
- **一句话说明：** 以共享 robot API、reference harness 与隐藏 verifier 评测 agentic Code-as-Policy 的 25-task 仿真操作基准。

## 仓库入口（README，2026-10-15）

| 组件 | 说明 |
|------|------|
| `benchmark/tasks/` | 25 个任务定义 |
| `src/codeaction/`、`backend/` | 基准运行时与仿真后端 |
| `tools/setup.sh`、`tools/reproduce.sh` | 环境设置与检查/复现入口 |
| `examples/`、`docs/` | 可运行示例与部署、评测、扩展说明 |
| `QUICKSTART.md` | 环境安装、首个任务、完整评测与故障恢复 |

## 复现要求与开放边界

README 给出的主机要求为 Linux x86-64、NVIDIA GPU、Docker Compose、NVIDIA Container Toolkit 与 Python 3.10+；环境/演示约需 120 GB，完整评测和导出约需 150 GB。基础检查命令为：

```bash
bash tools/setup.sh venv
bash tools/reproduce.sh check --gpus 0
```

完整评测需要模型 API 或订阅额度。完整 trajectory、观察图像与视频数据集标记为 coming soon；代码已公开，论文全部评测轨迹包尚待发布。

## 与仓库内实体的关系

- [CodeActionBench 论文实体](../../wiki/entities/paper-codeactionbench.md)
- [CodeActionBench 项目页核查](../sites/codeactionbench-org.md)
- [操作任务](../../wiki/tasks/manipulation.md)
