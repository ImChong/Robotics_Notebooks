# SIMPLE（USC PSI Lab）

> 来源归档（ingest）

- **标题：** SIMPLE — Simulation-Based Policy Learning and Evaluation for Humanoid Loco-manipulation
- **类型：** repo
- **链接：** <https://github.com/physical-superintelligence-lab/SIMPLE>
- **机构：** USC Physical Superintelligence (PSI) Lab
- **论文：** arXiv:2606.08278 — [simple_arxiv_2606_08278.md](../papers/simple_arxiv_2606_08278.md)
- **项目页：** <https://psi-lab.ai/SIMPLE> — [psi-lab-simple.md](../sites/psi-lab-simple.md)
- **入库日期：** 2026-07-16
- **本次核查：** 2026-11-21（官方 README、pyproject.toml、评测 CLI 及 SONIC 控制类）
- **一句话说明：** 基于 AMO/SONIC 栈的人形 loco-manipulation **全栈仿真环境**：MuJoCo+Isaac Sim 双引擎、Gym 评测接口、运动规划与 VR 遥操作数据采集，并集成 Ψ₀、π₀.₅、GR00T 等主流 VLA 评测脚本。

## 仓库要点

- **定位：** full-stack simulation environment for humanoid loco-manipulation；非单篇算法 repo，而是 **benchmark + 数据生成 + 策略集成** 基础设施。
- **策略集成：** 原生支持 **Psi0、Pi05、GR00T** 等主流 VLA；README 含各任务 L0/L1/L2 成功率对照表。
- **评测数据：** OOD 评测环境托管于 Hugging Face [`USC-PSI-Lab/psi-data`](https://huggingface.co/datasets/USC-PSI-Lab/psi-data/tree/main/simple-eval)。
- **任务示例：** `G1WholebodyXMovePickTeleop-v0`、`G1Wholebody BendPickMP-v0`、`G1Wholebody HandoverTeleop-v0` 等 G1 全身任务族。
- **底层运控：** 构建于 **AMO/SONIC** 全身 tracking；高层 VLA 输出 kinematic trajectory + base command。

## 开源状态

- **已开源**（截至 2026-07-16）：训练/评测/数据采集代码公开；评测资产与部分数据在 HF `psi-data`。
- 与论文 Abstract「We will open-source our entire codebase」一致；以仓库实际目录为准复现。

## 源码运行入口（以官方 README / 实际源码为依据）

- **官方安装：** 递归初始化 submodules，`uv sync --all-groups --index-strategy unsafe-best-match`，需 CuRobo 时运行 `bash scripts/install_curobo.sh`；要求 NVIDIA RTX GPU，完整配置以仓库文档为准。
- **CLI 映射：** `pyproject.toml` 的 `eval-decoupled-wbc = simple.cli.eval_decoupled_wbc:typer_main` 和 `eval = simple.cli.eval:typer_main` 是不同入口。前者实际源码在 `src/simple/cli/eval_decoupled_wbc.py`：`gym.make(env_id,...)` 构建环境，动态实例化策略 Agent，循环 `get_action(...)` → `env.step(action)`，记录 success 与视频。
- **推理服务：** 官方 README 在 Ψ₀ 仓库先启动 `scripts/deploy/serve_psi0_simple.sh`（或 `serve-psi0-sonic-http` Docker 服务），SIMPLE 客户端用 `--host` / `--port` 连接；服务与仿真分不同进程/仓库。
- **README 实际示例：** `TASK_NAME=G1WholebodyXMovePickTeleop-v0 uv run eval-decoupled-wbc simple/G1WholebodyXMovePickTeleop-v0 psi0_decoupled_wbc train --data-format lerobot --data-dir data/SIMPLE-eval/G1WholebodyXMovePickTeleop-v0/dr-level-0 --host 127.0.0.1 --port 21000 --headless`（需预置评测数据，先启动对应端口策略服务）。
- **WBC 代码：** `src/simple/agents/sonic_decoupled_wbc_agent.py` 的 `SonicDecoupledWbcAgent` 实例化 G1 机器人模型与 `get_wbc_policy`，核心控制步长 `0.02s`（50Hz）；MuJoCo 物理子步在论文附录为 500Hz，Isaac Sim 用于按配置渲染。
- **新发布记录：** 2026-11-11 的 `main` 是 Isaac Sim 4.5 + MuJoCo 3.3；`isaacsim-6.2-mujoco-3.12` 分支支持 Isaac Sim 6.2 / MuJoCo 3.12。已公开 decoupled WBC **12 Teleop + 5 MP** 数据集、Ψ₀ 权重、评测 episode，不能和论文 16 项 benchmark 混为一谈。
- **复现边界：** 本次只核查公开代码、README 与接口，未执行 GPU 策略评测，因此不声称实测复现通过。

## 对 wiki 的映射

- [SIMPLE 论文实体](../../wiki/entities/paper-loco-manip-161-075-simple.md)
- [Isaac Gym / Isaac Lab](../../wiki/entities/isaac-gym-isaac-lab.md)
- [Loco-Manipulation](../../wiki/tasks/loco-manipulation.md)
