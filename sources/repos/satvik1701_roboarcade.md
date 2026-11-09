# Satvik1701/RoboARCade — 官方可运行评测代码

- **仓库：** https://github.com/Satvik1701/RoboARCade
- **论文：** https://arxiv.org/abs/2611.08209
- **核查：** 2026-11-09；README、core/run_eval.py、core/run_seed.py、core/mcp_server.py 的运行入口及职责
- **开放状态：** 代码已公开；另需 Hugging Face assets、EGL、Bubblewrap、API key 或已授权运行时。

## 官方运行入口
1. `bash scripts/setup.sh` 安装依赖及模型/场景资产。
2. `source set_env.sh` 配置环境，`python -m core.run_eval --model MODEL_ID --seeds 1` 运行默认 tidy_counter。
3. `core.run_eval` 逐 seed 调用 `core.run_seed` 启动仿真服务和 agent sandbox。
4. `core.mcp_server` 加载所选接口包 `interfaces/<name>/tools.py` 并在不同 UI/SimKernel 上执行动作请求。
5. `core.grade_episode` 对完整/子目标完成情况评分，运行结果、轨迹与视频写入实验目录。

## 边界
- 以上来自 README 和实际源码入口，未下载 assets、执行仿真或验证论文所有数字。
- [主节点](../../wiki/entities/paper-roboarcade.md)
- [论文归档](../papers/roboarcade_arxiv_2611_08209.md)
