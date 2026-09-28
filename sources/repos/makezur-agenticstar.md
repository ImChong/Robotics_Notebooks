# makezur/agenticSTAR

> 来源归档（ingest）

- **名称：** agenticSTAR（AgentSTAR harness）
- **类型：** repo
- **URL：** <https://github.com/makezur/agenticSTAR>
- **许可证：** MIT（`harness/views/sweeps/lib/_vendor/` 含 SciPy BSD 片段）
- **项目页：** <https://agenticstar.github.io/>
- **论文：** [arXiv:2609.24487](https://arxiv.org/abs/2609.24487)
- **入库日期：** 2026-09-28
- **一句话说明：** **Agentic 单目铰接/刚性物体** shape + track harness：Claude Code / Codex 读 `AGENT_TASK.md` 写 `scene.py`（Blender 原语 + 关节），配合 **pose sweep** 与 **temporal report** 迭代；输出 GLB + `pose.json`。

## 开源状态

- **已开源**（2026-09-28）：安装脚本、示例 capture、supervised run 工具链完整可跑。

## 关键入口（README 对齐）

| 路径 | 作用 |
|------|------|
| `install.sh` | artscript env、Blender 4.2.5、pi3x env |
| `AGENT_TASK.md` | agent 任务说明 |
| `conventions/` | `scene.py` 合约（shape vs pose 字段分离） |
| `harness/views/sweeps/` | pose 搜索 sweep/apply |
| `harness/analysis/temporal/` | 序列 temporal diagnostic |
| `tools/run_kf.sh` | 单次 keyframe 监督 run（`--agent claude|codex`） |
| `tools/make_capture.py` | Pi3X 相机 + keyframes |
| `examples/garden_shears/` |  committed 示例 input/capture |

## 运行要求

- Linux x86_64、NVIDIA GPU、micromamba、tmux、bubblewrap、ffmpeg 等。
- `ANTHROPIC_API_KEY` / `OPENAI_API_KEY`；agent 在 bwrap 沙箱内仅访问模型 API 主机。
- 论文设置：`--enable mechanism`；GPT-5.6-Sol + `critic` 分支（新模型 README 建议默认 off）。

## 关联

- 论文：[agenticstar_arxiv_2609_24487.md](../papers/agenticstar_arxiv_2609_24487.md)
- 项目页：[agenticstar-github-io.md](../sites/agenticstar-github-io.md)
- Wiki：[paper-agenticstar.md](../../wiki/entities/paper-agenticstar.md)
