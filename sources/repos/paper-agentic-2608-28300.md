# XinGP/MaCoPlanner 官方源码归档

- **仓库地址：** <https://github.com/XinGP/MaCoPlanner>
- **许可证：** MIT License
- **默认分支：** `main`
- **关联论文：** [MaCoPlanner: LLM-Assisted Manual-Compiled Task Planning with Proactive Safety Verification for Robotic Industrial Panel Operation](paper-agentic-2608-28300.md)（arXiv:2608.28300）
- **核对边界：** 静态代码与仓库元数据核对（基于已抓取仓库目录清单、README 与公开审计说明），**未在本地实际运行或连接真机驱动**

## 仓库核心结构与入口文件

仓库采用标准 `pyproject.toml` Python 包结构（要求 Python $\ge 3.10$），核心实现位于 `src/macoplanner/`：

- **统一编排与 CLI 入口：**
  - `src/macoplanner/__main__.py` 与 `src/macoplanner/cli.py`：提供 `list-tasks`、`verify`、`plan`、`retrieve`、`settings`、`execute`、`batch` 与 `benchmark list/validate` 子命令。
  - `src/macoplanner/pipeline.py`：公开无泄漏（leak-free）Python API `plan_task`、`retrieve_rules`、`verify_plan`，返回状态为 `verified` / `rejected` / `unverified`（仅 `verified` 会填充 `verified_plan`）。
  - `src/macoplanner/settings.py` 与 `src/macoplanner/runtime_helpers.py`：固定默认规范配置（如 E5 检索器、`--max-refinements 3` 预算与防静默篡改校验，需显式 `--allow-custom-config` 方可修改）。
- **设备建模、知识资源与形式化规则：**
  - `src/macoplanner/devices/{engine.py, generator.py, vfd.py}`：3 类设备（Engine、Generator、VFD）的动作原语模式、规则检查与安全状态机（FSM）。
  - `src/macoplanner/resources/{engine, generator, vfd}/`：每类设备均包含 `tasks.json`、`knowledge.json`、`rules.json` 与代表性面板图 `panel.jpg` / `panel.png`。
- **检索与证据门控：**
  - `src/macoplanner/retrieval/e5.py`：默认使用 `intfloat/e5-base-v2` 密集向量检索（支持 `--local-files-only` 离线限制），同时保留 TF-IDF 与 Haystack BM25 作为显式备选。
- **物理定位与受限执行适配器：**
  - `src/macoplanner/execution/{actions.py, dobot.py, vision.py, models.py}`：面向 `move`、`press`、`turn`、`observe` 原语的严格执行适配器，默认 **dry-run**；真机运行需额外安装 Dobot 驱动、提供现场标定的控制映射（模板见 `examples/control_map.example.json`），并显式传入 `--live --acknowledge-hardware-risk`。
- **可复现基准与离线测试：**
  - `src/macoplanner/benchmarks/api.py` 与 `src/macoplanner/benchmarks/data/`：内置 400 条检索基准（覆盖 `engine` / `generator` / `laser` / `vfd_simulator`）与 1,440 条结构化规划评测记录（`planning/batch_0309/`）。
  - `tests/test_*.py`：覆盖基准校验、CLI、执行适配器、管线、资源、检索与运行时配置辅助的离线单元测试。

## 开源范围与复现边界

- **已包含：** 统一规划与校验 API/CLI、3 类设备可执行 profile、400 检索基准 + 1,440 条规划记录、离线验证器（`macoplanner benchmark validate`）与单元测试。
- **有意排除（`docs/open-source-audit.md`）：** 原始设备 PDF 手册、历史开发版本、ROS 构建树、Dobot 厂商驱动、硬件标定文件、原始实验日志与本地模型权重缓存（`intfloat/e5-base-v2` 需在线从 Hugging Face 拉取或本地缓存）。
- **安全边界：** 仓库声明 MaCoPlanner 仅校验编码后的任务级规则，非经认证的工业安全系统，不负责碰撞规避、力控限幅、定位误差或工厂级联锁保护。

对 wiki 的映射：[MaCoPlanner](../../wiki/entities/paper-agentic-2608-28300.md)
