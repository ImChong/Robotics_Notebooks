# RLE-Bench（官方评测仓）

> 来源归档

- **标题：** RLE-Bench: A Qualifying Exam for Coding Agents as Robot Learning Engineers
- **类型：** repo + benchmark CLI + Harbor integration
- **组织：** RLE-Bench contributors（Harvard × Georgia Tech 主导，见项目页）
- **代码：** <https://github.com/RLE-Bench/RLE-Bench>
- **项目页：** <https://rle-bench.github.io/>
- **博客：** <https://rle-bench.github.io/blog/>
- **评测框架：** <https://github.com/laude-institute/harbor>
- **License：** MIT
- **Python：** 3.12+（`uv` + Docker；GPU task 需 NVIDIA）
- **入库日期：** 2026-09-16
- **一句话说明：** 九任务族 Harbor 评测 CLI（`rlebench prepare/run/summarize/view`）、Docker 仿真栈、agent 适配器与 hidden verifier；**已开源可本地复现**。

## 仓库入口（策展）

| 资源 | 路径 / 命令 | 说明 |
|------|-------------|------|
| CLI 入口 | `rlebench/` · `rlebench/cli.py` | `list` / `doctor` / `prepare` / `run` / `summarize` / `view` |
| 任务族 | `tasks/task01` … `tasks/task09` | 每族 README：变体、预算、打分 |
| 仿真栈 | `sim/` |  pinned simulator 构建脚本 |
| 共享机器人 | `assets/robots/` | MJCF/URDF 与第三方 LICENSE |
| Agent 适配 | `rlebench/agents/` | Claude Code、Codex 等 |
| Harbor | 依赖 `laude-institute/harbor` | 沙箱评测与子进程 verifier |

## 典型复现路径

```bash
git clone https://github.com/RLE-Bench/RLE-Bench.git
cd RLE-Bench
make install && source .venv/bin/activate
rlebench doctor
rlebench prepare task08
rlebench run task08 -a claude-code -m "anthropic/MODEL_NAME"
rlebench summarize jobs
```

- Scope 示例：`task01`、`task01/L1`、`task01/L1/01-open-fridge`、`task06/rgb-only`
- GPU：`--device cuda:0`
- `--dry-run` 预览 Harbor 命令

## 开源状态（2026-09-16）

| 项 | 结论 |
|----|------|
| 评测代码 + 任务规范 | **已开源**（MIT） |
| Docker 镜像 / task assets | `make prepare` / `make taskXX-assets` 生成 |
| HF 数据 | `RLE-Bench/task05` 等 **部分** 公开 |
| 论文 PDF | **待发布** |

## 对 wiki 的映射

- 实体页：[RLE-Bench](../../wiki/entities/rle-bench.md)
- 站点：[rle-bench-github-io.md](../sites/rle-bench-github-io.md)
- 博客：[rle_bench_introducing_blog_2026-09-14.md](../blogs/rle_bench_introducing_blog_2026-09-14.md)
