# Argus（Pantheon）

- **类型：** repo
- **标题：** Argus: Dense annotations and data-quality checks for robot-learning episodes
- **链接 / 代码：** https://github.com/Pantheon-Industries-Inc/argus
- **项目页：** https://pantheon.inc/research/argus
- **数据面板：** https://pantheon.inc/data-board
- **在线工具：** https://data.pantheon.inc/review
- **入库日期：** 2026-10-02
- **开源状态：** 已开源；代码 Apache-2.0，公开标注 CC BY 4.0；原始录像保留各数据集许可。
- **交叉归档：** [官方文章](../sites/pantheon-argus.md)
- **沉淀到 wiki：** [Argus 工具详情](../../wiki/entities/pantheon-argus.md)

## README 核查

读取 LeRobot v2/v3、MCAP、普通视频和压缩包；支持遥操作机械臂、UMI 夹爪、人类第一视角三类 setup。输入包括多相机、状态/动作和任务指令；缺少指令时由视频推断任务。

| 模块 | 入口与职责 |
|------|------------|
| prepare | `python -m prepare`：读取原始格式，生成 episode sidecar，不重新编码 |
| checks | `python -m checks`：检查相机配对、位姿跳变、夹爪信号和采集质量 |
| label | `python -m label`：精确解码、选帧、分辨率路由、提示词和 VLM 调用 |
| review | `python -m review`：自有数据完整审计 |
| board | `python -m board`：构建/服务可视化面板，导出 JSON/JSONL |
| compare / gate | 模型对比与帧级回归用例 |

## 复现入口

需要 uv、ffmpeg >= 5.1，以及 OpenAI 或 OpenRouter API key。部分 Hugging Face 数据需先接受使用条款；代码开放不等于标注推理免费。

```bash
git clone https://github.com/Pantheon-Industries-Inc/argus.git
cd argus
uv sync --frozen
uv run pytest
uv run python -m review --data path/to/data --rig teleop_arms --out data/review/mine --free
uv run python -m review --data path/to/data --rig teleop_arms --out data/review/mine --cap 20
uv run python -m board serve --board data/review/mine --clips data/review/mine/clips
```

`--free` 构建请求但不调用模型；`--cap` 限制新 episode 调用预算。标注包含 timeline、completion、goal_alignment、operator_mistakes、recovery、data_issues、state_changes 和 scene_graph。模型输出不保证逐位复现，应保存 commit、请求、分辨率和实际 provider。
