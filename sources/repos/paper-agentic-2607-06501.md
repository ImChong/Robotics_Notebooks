# threefruits/HUME 官方实现仓库归档

- **仓库地址：** https://github.com/threefruits/HUME
- **关联论文：** [arXiv:2607.06501](https://arxiv.org/abs/2607.06501)（RSS 2026）
- **开源协议：** MIT License
- **默认分支：** `main`
- **开源范围判定：** 部分开源（仿真部分完整开源，含 Block Processing World 与 AI2-THOR/ProcTHOR 移动操作两个仿真域及五种实验配置；明确不包含真机 Fetch 系统与 Autolife S2 家电面板操作组件）
- **核对边界：** 基于快照的静态代码结构与 `README.md` 核对（未本地运行与复现）

## 仓库目录与入口文件核对

- **依赖与测试：**
  - `pyproject.toml`、`uv.lock`：Python ≥ 3.8 环境锁定，通过 `uv sync --all-extras` 安装（含 AI2-THOR / ProcTHOR）。
  - `scripts/install_fast_downward.sh`：编译 Fast Downward 并导出 `HUME_FAST_DOWNWARD`。
  - `test_generated_env_ai2.py`、`test_generated_env_blockworld.py`、`test_generated_env_solver.py`：代表性用例与求解器回归测试（`uv run pytest -q`）。
- **核心规划模块（`hume/`）：**
  - `hume/planning/base.py`、`hume/planning/model.py`、`hume/planning/pddl.py`、`hume/planning/solver.py`：规划基础抽象、模型扩展管理、PDDL 问题构建与 Fast Downward 求解器接口。
  - `hume/planning/block_processing/{hypotheses.py, llm.py, prompts.py, symbolic.py, verification.py}`：Block Processing World 域的假设生成、LLM 接口、Prompt、符号规划转换与验证逻辑。
  - `hume/planning/mobile_manipulation/{hypotheses.py, llm.py, prompts.py, symbolic.py, verification.py}`：Mobile Manipulation 域的假设生成、LLM 接口、Prompt、符号规划转换与验证逻辑。
  - `hume/pddl/block_processing/domain.pddl` 与 `hume/pddl/mobile_manipulation/{domain.pddl, domain_ai2.pddl, mobile_manipulation.pddl}`：PDDL 域定义文件。
  - `hume/worlds/block_processing/`：Block Processing World 环境配置与任务定义（如 `data/environments/3b3r_0/` 至 `3b5r_7/`）。
- **实验入口与分析脚本（`experiments/`）：**
  - `experiments/block_processing/run.py`、`plot_results.py`、`finalize_results.py`：Block Processing World 评测入口与结果汇总绘图。
  - `experiments/mobile_manipulation/run.py`、`plot_results.py`、`finalize_results.py`、`summarize_logs.py`、`replace_task_results.py`：AI2-THOR/ProcTHOR 移动操作评测入口与日志分析。
  - 支持的 `--planner_type` 五种配置：`llm_nh_nv`、`llm_h_nv`、`llm_h_v`、`pddl_h_nv`、`pddl_h_v`（需配置 `OPENAI_API_KEY`）。
- **对 wiki 的映射：** [Hypothesis-driven Model Expansion under Uncertainty for Open-World Robot Planning (HUME)](../../wiki/entities/paper-agentic-2607-06501.md)
