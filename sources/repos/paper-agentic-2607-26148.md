# jianzhou0420/MIP 官方代码核查

- **代码仓库：** <https://github.com/jianzhou0420/MIP>
- **开源协议与分支：** MIT License，默认分支 `main`（快照显示 23 次提交）
- **对应论文：** [Embodied Agents Take Control (MIP, arXiv:2607.26148v4)](../papers/paper-agentic-2607-26148.md)
- **配套依赖仓：** <https://github.com/jianzhou0420/EmbodiedScore-envs>（MIT License，Gymnasium 1.3 仿真封装与冻结子集）、<https://github.com/SWE-agent/mini-swe-agent>（MIT License，轻量开源 ReAct 脚手架）
- **核对范围：** 静态核对 GitHub 仓库主页快照（`project_page.txt`）、`README.md` 预览、目录树与文件样本清单（`files_sample` 含 `runner.py`、`mip/core/`、`mip/harnesses/`、`mip/tools/`、`mip/envs/`、`mip/prompts/`、`mip/configs/`、`scripts/mip_paper_cells.sh`、`splits/`、`ui/`）。静态代码核对，未本地运行。
- **通读：** 未完成
- **开源范围判断：** **已开源**论文全部仿真评测流水线、三套智能体脚手架适配（Claude Agent SDK / Codex CLI / mini-swe-agent）、五类 MCP 工具臂（Minimal / Waypoint / Hybrid / SLAM / Hybrid+SLAM）、实验配置 YAML、`mip100`/`rand100` 评测子集与可视化轨迹浏览器。Unitree Go2 真机控制端实现未见于当前公开目录清单。

## 已核对目录与入口

1. **统一启动入口与复现脚本：**
   - `runner.py`：主运行入口，按 `std_mip_<benchmark>_<interface>.yaml` 加载配置，并通过命令行参数选择 `harness=`（`cc` / `codex` / `mini`）、`model=`（`mip/configs/models/models.yaml`）与 `effort=`；支持 `+run.fake=true run.episodes=0` 做无 token 连通性自检。
   - `scripts/mip_paper_cells.sh`：汇总论文各实验表对应的 `runner.py` 命令矩阵。
   - `ui.server`（`python -m ui.server`）：浏览器端查看 `outputs/<harness>/<run name>/` 归档的事件日志、每帧观测图像与成本统计。
2. **核心运行时 `mip/core/`：**
   - 调度与会话：`mip/core/run.py`、`mip/core/episode.py`、`mip/core/bus.py`、`mip/core/retry.py`。
   - MCP 与环境通信：`mip/core/bridge.py`、`mip/core/toolset.py`、`mip/core/envclient.py`、`mip/core/envpool.py`、`mip/core/envserver.py`。
   - 模型 API 与回调统计：`mip/core/api/`（`anthropic_wire.py`、`openai_wire.py`、`vllm.py`、`ollama.py`、`fake_wire.py` 及 `providers/`）与 `mip/core/callbacks/`（`cost.py`、`jsonl.py`、`pose.py`、`raw.py`、`snapshot.py`、`stats.py`、`vram.py`）。
3. **脚手架适配层 `mip/harnesses/`：**
   - `claude_sdk.py`（对应 `configs/harness/cc.yaml`）、`codex_cli.py`（对应 `configs/harness/codex.yaml`）、`mini_swe.py` 与 `mip/harnesses/mini/`（`main.py`、`nav_agent.py`、`bridge_toolset.py`、`env.py`、`model.py`，对应 `configs/harness/mini.yaml`）、`fake.py`、`process.py`、`protocol.py`。
4. **工具集 `mip/tools/` 与环境适配 `mip/envs/`：**
   - 基础导航工具 `mip/tools/nav/`：`observe.py`、`step.py`、`answer.py`、`view.py`。
   - 航点工具 `mip/tools/waypoint/`：`observe.py`、`observe_waypoints.py`、`goto.py`、`panorama.py`、`predictor/engine.py`。
   - SLAM 工具与后端 `mip/tools/slam/`：`get_pose.py`、`get_map.py`、`get_trajectory.py`、`look_around.py`、`mark.py`、`preview_path.py`、`docker/Dockerfile` 及 `world/`（`_slam.py`、`_slam_server.py`、`_mapping.py`、`_frontier.py`、`_plan.py` 等）。
   - 环境后端 `mip/envs/embodiedscore/`：`env.py`、`hook.py`（对接 `EmbodiedScore-envs` + `habitat-sim 0.3.3`）。
5. **提示词、实验配置与评测子集：**
   - `mip/prompts/`：`nav_4.yaml`、`objnav_6.yaml`、`eqa_6.yaml`、`nav_wp.yaml`、`nav_hybrid.yaml`、`nav_slam.yaml`、`nav_hybrid_slam.yaml` 及 `fragments/standard.yaml`。
   - `mip/configs/experiments/`：`std_mip_{r2r,rxr,hm3d,hm3d_v2,mp3d,hmeqa}_minimal.yaml` 及 `std_mip_r2r_{waypoint,hybrid,slam,hybrid_slam}.yaml`。
   - `splits/`：收录 R2R-CE `rand100` 与其余 5 个基准的 `mip100` 子集及哈希校验清单 `splits/manifests/`（与 `EmbodiedScore-envs` 内置副本逐字节一致）。

## 边界说明

- 本归档仅基于仓库公开快照的目录树、`README.md` 与 `INSTALL.md` 引用进行静态结构核对，**未在本地配置双 Conda 环境、Habitat-Sim 0.3.3 数据集、SLAM Docker 或模型 API 进行实际运行**。
- 仓库 README 明确标注：WayPoint / Hybrid 单元需额外启动航点预测器服务，SLAM 单元需构建并启动 `mip/tools/slam/docker/Dockerfile`。

## 映射

- [Embodied Agents Take Control (MIP) 论文实体与源码运行时序图](../../wiki/entities/paper-agentic-2607-26148.md)
