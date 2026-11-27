# Admire-ljb/AeroWeaver 官方仓库核查归档

- **仓库地址：** https://github.com/Admire-ljb/AeroWeaver
- **项目主页：** https://admire-ljb.github.io/AeroWeaver/
- **关联论文：** [AeroWeaver (arXiv:2609.18520)](https://arxiv.org/abs/2609.18520) · [论文原文归档](../papers/paper-agentic-2609-18520.md)
- **默认分支：** `main`
- **许可证：** MIT License
- **开源状态：** 已开源（包含后端服务、技能库与注册表、记忆与在线强化模块、Mock/AirSim/PX4-Gazebo 适配器、前端控制台、实验与测试目录）
- **核对边界：** 静态代码核对（基于 GitHub 仓库页面、README 与目录文件清单样本），未在本地拉取运行

## 已核实的目录与入口结构

- **运行入口与前后端构建（来自 `README.md`）：**
  - 环境依赖：Python 3.10+、Node.js 22.12+（或 20.19+）、npm 10+。
  - 启动顺序：`pip install -r requirements/mock.txt` → `npm --prefix frontend ci` → `npm --prefix frontend run build` → `python backend/server.py`，默认在 `http://127.0.0.1:5001` 启动 Mock 与 Manual 模式控制台，切换 Autonomous 模式并配置 LLM 后可执行自然语言蜂群任务。
- **核心模块分布（来自仓库文件样本）：**
  - **服务与配置：** `backend/server.py`、`backend/config.py`、`backend/config/safety_config.yaml`、`backend/config/sim_config.yaml`
  - **指挥官与分布式决策：** `backend/brain/commander.py`、`backend/swarm/commander.py`、`backend/brain/planner_agent.py`、`backend/brain/agent_loop.py`、`backend/brain/uav_agent_context.py`、`backend/brain/mission_progress.py`
  - **执行运行时：** `backend/runtime/agent_runtime.py`、`backend/runtime/uav_agent_runtime.py`、`backend/runtime/exector.py`
  - **空中技能层与文档：** `backend/skills/base_skill.py`、`backend/skills/registry.py`、`backend/skills/skill_loader.py`、`backend/skills/skill_doc_generator.py`、`backend/skills/mpe_skills.py`、`backend/skills/motor_skills.py`、`backend/skills/swarm_skills.py`、`backend/skills/docs/mpe/skill.md`、`backend/skills/docs/mock_tasks/skill.md`、`backend/skills/soft_docs/`
  - **经验记忆与在线适应：** `backend/memory/swarm_experience.py`、`backend/memory/mpe_experience.py`、`backend/memory/roles.py`、`backend/memory/shared_memory.py`、`backend/memory/episodic_memory.py`、`backend/memory/skill_memory.py`
  - **仿真与真机适配器：** `backend/adapters/adapter_manager.py`、`backend/adapters/mock_adapter.py`、`backend/adapters/airsim_adapter.py`、`backend/adapters/px4_adapter.py`、`backend/adapters/gazebo_direct_adapter.py`、`backend/adapters/mavsdk_adapter.py`、`backend/sim/mpe_runtime.py`
- **仓库安全声明：** `README.md` 明确提示其为研究软件，物理飞行前须先在仿真中验证，并配备独立地理围栏、急停开关与操作员监督。
- 对 wiki 的映射：[AeroWeaver 论文实体页](../../wiki/entities/paper-agentic-2609-18520.md)
