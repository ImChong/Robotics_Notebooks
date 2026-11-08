# Simular Agent-S：Agent S3 官方源码

> 来源归档（repo / 源码与部署）

- **类型：** repo
- **仓库：** <https://github.com/simular-ai/Agent-S>
- **S3 模块：** <https://github.com/simular-ai/Agent-S/tree/main/gui_agents/s3>
- **README：** <https://github.com/simular-ai/Agent-S/blob/main/gui_agents/s3/README.md>
- **论文：** <https://arxiv.org/abs/2510.02250>
- **许可证：** Apache-2.0（仓库根目录 `LICENSE`）
- **入库日期：** 2026-11-08
- **核对范围：** S3 README 全文、根 LICENSE；`gui_agents/s3/agents/agent_s.py`、`worker.py`、`grounding.py`、`code_agent.py`，`cli_app.py`、`utils/local_env.py`，`bbon/behavior_narrator.py` 与 `comparative_judge.py` 的主类、执行入口及关键调用路径。
- **通读：** 已完成
- **沉淀到 wiki：** [Agent S3](../../wiki/entities/paper-agent-s3.md)

## 开源及运行范围

| 组件 | 官方实现 / 边界 |
|---|---|
| GUI Agent | `AgentS3.predict` → `Worker.generate_next_action`；`OSWorldACI` 提供交互动作 grounding |
| 本机 CLI | `gui_agents/s3/cli_app.py`；`run_agent` 获取屏幕截图并 `exec(code[0])` 执行；该 CLI **不运行 BJudge**，默认最多迭代 15 步 |
| Code Agent（可选） | `--enable_local_env` 后 `call_code_agent` → `CodeAgent.execute` → `LocalEnv` 执行 Python/Bash |
| BJudge 基础模块 | `bbon/behavior_narrator.py` 与 `comparative_judge.py`；其存在**不意味着** CLI 自动实现论文 10 条独立轨迹的编排 |
| 安装 | `pip install gui-agents` 或源码 `pip install -e .` |
| 环境 | 官方标明 macOS/Windows/Linux、单显示器设计；需模型 API 与独立 UI grounding 服务 |

### README 提供的单轨 CLI 示例

```bash
agent_s --provider openai --model gpt-5-2025-08-07 \
  --ground_provider huggingface --ground_url http://localhost:8080 \
  --ground_model ui-tars-1.5-7b \
  --grounding_width 1920 --grounding_height 1080
```

UI-TARS grounding 服务、推理 API 与桌面权限须另行配置；上面的 localhost 为**示例服务地址**，不是内置可立即调用的云服务。

## 安全性与复现注意

- 在用户会话权限下执行 GUI 动作；`--enable_local_env` 可执行任意 Python/Bash，建议沙箱、最小权限、操作确认及备份。
- GitHub 源码为 Apache-2.0，不表示 GPT-5／Claude／grounding 端点推理免费。
- 当前源码结构核对只支持 **S3 单轨 CLI 与 BJudge 组件接口** 的存在；论文 72.6% 须另行搭建多轨 benchmark / 隔离环境，不能用 CLI 即开即得。

## 交叉来源

- [原始论文](../papers/agent_s3_arxiv_2510_02250.md)
- [官方文章](../sites/simular_agent_s3.md)
- [统一实体页](../../wiki/entities/paper-agent-s3.md)
