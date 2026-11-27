# lianegalanti/Pigey 官方仓库归档

- **仓库地址：** https://github.com/lianegalanti/Pigey
- **对应论文：** [Addressing the Orchestration Gap in Generalist Robots via Physical Agency](https://arxiv.org/abs/2607.21725)（[论文源归档](../papers/paper-agentic-2607-21725.md)）
- **项目页：** https://lianegalanti.github.io/Pigey/
- **默认分支：** `main`（82 stars / 3 forks / 35 commits，核对快照）
- **开源协议：** 仓库根目录未附带独立 `LICENSE` 文件（arXiv 论文正文声明 `CC BY 4.0`）
- **开源状态：** 已开源（核心编排代码与系统提示词公开，上游底层策略与 TAMP 依赖均指向公开仓库）
- **核对边界：** 静态代码核对，未本地运行（真机端依赖 Franka Research 3、Robotiq 2F-85、ZED 2i/ZED-Mini 硬件及本地 GPU 工作站服务；仿真端依赖 `openpi` `pi05_libero` 服务与外部 VLM / Gemini Robotics-ER API）

## 仓库目录与已核实入口文件

根据 `lianegalanti/Pigey` 根目录结构与 `README.md` 说明，仓库定位为“在公开上游技能栈之上运行的轻量化智能体编排胶水层”，按运行环境分为两套对齐实现：

| 目录 / 文件 | 运行环境 | 核实职责与入口说明 |
|---|---|---|
| `real/agent.ts` | 真机（Franka FR3 + $\pi_{0.5}$-DROID） | TypeScript 编排器主入口（通过 `bun agent.ts "<instruction>"` 启动），串联 VLM API 与三类机器人服务端点 |
| `real/agent-system.md` | 真机 | 编排器系统提示词，规定工具 Schema、决策路由顺序、`pi0.5` 短指令规范、消歧/遮挡/记忆恢复策略与 `Perceive -> Done` 纪律 |
| `sim/agent_sim.py` | 仿真（LIBERO / LIBERO-PRO） | Python 编排器与工具集实现，经 LiteLLM 调用多厂商 VLM，并调用 `gemini/gemini-robotics-er-1.6-preview` 生成 `#N` 编号检测框 |
| `sim/spatial_tools.py` | 仿真 | 解析式抓取（`Grasp`）、放置（`Place`）与空间辅助工具函数 |
| `sim/README.md` | 仿真 | 仿真环境配置、`pi05_libero` 策略服务启动与 Blackwell/B300 GPU 适配说明 |
| `README.md` / `index.html` / `static/` | 项目展示与说明 | 安装命令、上游依赖列表、架构图（`static/figs/`）与真机任务演示视频（`static/videos/`） |

## 运行栈与上游依赖映射

1. **真机运行时端点（`real/agent.ts`）**：
   - `TAMP_URL`（默认 `http://<workstation>:7777`）：封装 [TiPToP](https://github.com/tiptop-robot/tiptop) 服务，内部组合 Gemini Robotics-ER、SAM 2、FoundationStereo、M2T2 抓取预测与 cuRobo 运动规划。
   - `DROID_HOST`（默认 `tcp://<workstation>:9876`）：封装 [DROID](https://github.com/droid-dataset/droid) 与 polymetis 控制接口，负责相机观测、机器人状态读取与 VLA 控制步执行。
   - `PI0_URL`（默认 `ws://<workstation>:8000`）：由 [openpi](https://github.com/Physical-Intelligence/openpi) 提供的 $\pi_{0.5}$-DROID WebSocket 策略推理服务（checkpoint：`gs://openpi-assets/checkpoints/pi05_droid`）。
   - 可选环境变量：`AGENT_MAX_STEPS`、`AGENT_FAIL_LIMIT`、`AGENT_LLM_TIMEOUT_MS`、`AGENT_SYSTEM`、`AGENT_SKIP_WRIST_REC`。
2. **仿真运行时入口（`sim/agent_sim.py`）**：
   - CLI 参数：`--mode harness --suite <suite_name> --task <id> --episode <id> --policy-host localhost --policy-port 8000 --model <litellm_model> --max-llm-steps 35 --max-steps-per-rollout 500 --out-dir results/`。
   - 经 LiteLLM 支持跨推理模型切换，已验证模型包括 Gemini 3.5 Flash、Claude Opus 4.7、Claude Sonnet 4.6、Claude Haiku 4.5、Claude Fable 5、Gemini 3.1 Pro、GPT-5-mini 等。
