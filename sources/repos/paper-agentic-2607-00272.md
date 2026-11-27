# NVlabs/ASPIRE — 官方源码仓库归档

- **官方仓库：** <https://github.com/NVlabs/ASPIRE>
- **许可证：** Apache-2.0（主项目代码与文档；第三方依赖保留各自许可，见 `NOTICE`）
- **默认分支：** `main`（仓库快照 Star 数：1,078）
- **关联论文与项目页：** [arXiv:2607.00272](https://arxiv.org/abs/2607.00272) / [NVIDIA Research 项目页](https://research.nvidia.com/labs/gear/aspire/)；论文摘录见 [paper-agentic-2607-00272.md](../papers/paper-agentic-2607-00272.md)
- **核查状态：** `静态代码核对，未本地运行`（基于仓库目录抽样 `files_sample`、README 预览及论文 Appendix G.1 代码结构说明）

## 仓库定位

`NVlabs/ASPIRE` 是论文《ASPIRE: Agentic /Skills Discovery for Robotics》（v1.0）的官方开源代码库，将“闭环逐原语多模态执行引擎 + 独立验证准入的技能库 + 程序级进化搜索”打包为兼容 Claude Code 与 Codex CLI 的 Agent Skills 工作流及统一 CaP（Code-as-Policy）测试框架，并附带 YAM 双臂真机 System 2 运行手册与任务示例。

## 已核实的目录与入口文件

依据仓库根目录抽样、README 及论文 Appendix G.1，核心模块与入口分布如下：

- **核心 CaP 运行时与适配器（`src/cap_harness/`，Appendix G.1 与 README 说明）：**
  - `api.py`、`contracts.py`：公开 CaP 编程接口与契约定义
  - `run.py`、`batch.py`：单任务与批量执行入口
  - `providers/`：感知、抓取与运动规划后端（`sam3`、`graspnet`、`curobo`、`pyroki`）
  - `libero/`、`robosuite/`、`behavior/`、`yam_real/`：四大基准及真机环境适配器
  - `validation/`：held-out 评估器
- **Agent Skills 编排与调试流水线（`.claude/skills/`、`.codex/skills/`、`.agents/skills/`）：**
  - `.claude/skills/iterative-debugging/SKILL.md`：迭代调试主入口，含 Coordinator 提示 `main-agent-prompt.md`、Actor 提示 `subagent-prompt.md`、独立验证器提示 `verifier-prompt.md` 与 `clean-task-slate.md`
  - `.claude/skills/iterative-debugging/scripts/`：编排与工具脚本，含 `campaign.py`（战役配置）、`dispatch.py`（按任务填充模板与调度）、`init_run.py`、`run_validation.py`（held-out 验证）、`lint_skills.py`（技能提案预检）、`scene_snapshot.py`、`trace_stats.py`、`audit_seeding.py`、`gen_progress.py`、`watchdog.py`
  - `.claude/skills/iterative-debugging/behavior/SKILL.md`：BEHAVIOR-1K 专用增量块执行协议及 `score_campaign.py`、专用技能目录
  - `.claude/skills/iterative-debugging/skills/`：已积累的可复用技能库（`README.md`、`grasp.md`、`localize.md`、`manipulation.md`、`task-exploration.md`、`transport.md`）
  - `.claude/skills/evosearch/SKILL.md`：进化搜索入口及 `scripts/evosearch_eval.py`、`scripts/analyze_evosearch_traces.py`
  - `.codex/skills/`：Codex CLI 兼容入口（`iterative-debugging/SKILL.md`、`evosearch/SKILL.md`）
  - `.agents/skills/`：真机 System 2 技能入口（`cap-tool-usage/SKILL.md`、`manipulation-tasks/SKILL.md`、`manipulation-tasks/rubiks-cube.md`）
- **配置、引导脚本、文档与示例：**
  - `configs/`：`environments.json`、`dependency-lock.json`、`profiles/`（`l40.json`、`rtx4090.json`、`rtx5090.json`、`rtx5880.json`）、`validation/*.yaml`
  - `scripts/`：`bootstrap_libero.sh`、`bootstrap_behavior.sh`、`bootstrap_curobo.sh`、`bootstrap_graspnet.sh`、`bootstrap_provider.sh`、`bootstrap_providers.sh`
  - `examples/`：`libero/`、`robosuite/`、`behavior/` 基准脚本与 `yam_real/` 真机任务上下文（`rubiks_cube/`、`unscrew_bottle/`、`pill_in_bottle/`、`cup_stacking/`）
  - `docs/`：`host-setup.md`、`libero-pro.md`、`robosuite.md`、`behavior.md`、`api-reference.md`、`system2.md`、`yam-real.md`、`run-artifacts.md`

## 复现边界与安全警示

- **环境与凭据依赖：** 仿真复现需配备 CUDA GPU、`uv` 包管理器、外部前沿 LLM API 密钥（如 Claude Code / Codex），以及受限权重的 Hugging Face Token（用于下载 SAM3 权重）。
- **安全边界（README 明确警告）：** ASPIRE 直接执行 LLM 生成的 Python 代码且具备完整 `import` 权限，trial 进程与 `watchdog.py` **并非安全沙箱**。官方要求在隔离主机运行、限制网络与敏感目录挂载，严禁赋予仿真智能体访问物理硬件的权限；真机运行需严格遵守 `AGENTS.md` 与 `docs/system2.md` 操作规程并取得操作员单独授权。
- **静态核对说明：** 本归档仅完成仓库结构、文档与脚本入口的静态核对，未在本地 GPU 或真实机器人上实际运行测试。

## 对 wiki 的映射

- [ASPIRE: Agentic /Skills Discovery for Robotics](../../wiki/entities/paper-agentic-2607-00272.md)
