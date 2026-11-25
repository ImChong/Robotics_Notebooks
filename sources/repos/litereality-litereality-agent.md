# LiteReality-Agent（LiteReality/LiteReality-Agent）

- **代码：** <https://github.com/LiteReality/LiteReality-Agent>（公开，Apache-2.0）
- **项目主页：** <https://litereality.github.io/agent/>
- **交互式查看器：** <https://litereality.github.io/agent/vr/index.html>
- **论文：** [arXiv:2610.01863](https://arxiv.org/abs/2610.01863)
- **扫描 App：** [LiteReality Scanner](https://apps.apple.com/gb/app/litereality/id6774158260)（App Store 免费）
- **示例扫描：** [LiteReality/example-scans](https://github.com/LiteReality/example-scans)
- **实体页：** [litereality-agent](../../wiki/entities/litereality-agent.md)
- **论文归档：** [litereality_agent_2610_01863](../papers/litereality_agent_2610_01863.md)

- **固定版本：** `28ffe1fd4a1b03c93f1877a33a8bc9b806f72fe1`（2026-11-25 核对）

## 一句话说明

公开端到端室内重建与仿真工具链：uv run litereality run scans/<scan> 执行 scene init 与 agentic authoring；uv run litereality stage simulate run/<scan> 输出 MuJoCo 场景 MJCF。依赖 Blender 5.x、图像生成 API、Agent CLI，以及 Modal 托管模型或 Linux ≥24 GB GPU。

## 开源状态

- **已开源：** Agent CLI、代码与部署脚本；仓库采用 Apache-2.0。
- **已发布：** LiteReality Scanner App Store 应用，官方 README 标注免费。
- **可试跑数据：** example-scans 仓库提供公开扫描样例。
- **仿真范围：** 可导出 MuJoCo 场景 MJCF；物理属性仍需按具体用途验证。

## 固定版本运行核对

- `src/litereality_agent/pipeline/stages.py` 明确顺序为 ingest → reconstruct → seed → author → publish → simulate；最后一步不属于必需阶段。`--quality-pass` 本身不是下游读入项。
- `doc/Sim-Ready-intergration/Mujoco.md` 与 simulate 实现确认房间输出在 `realism_authoring/mujoco/`，包括 `scene.xml` 与 `export_report.json`。侧车匹配失败将明确标记 fallback；这与默认将所有资产视为已辨识不同。
- 本次静态走读 CLI 与公开源码，未调用模型服务、扫描、运行动力学仿真或验证实物参数。

## 交叉链接

- [LiteReality-Agent 实体页](../../wiki/entities/litereality-agent.md)
- [论文来源](../papers/litereality_agent_2610_01863.md)
- [项目页来源](../sites/litereality-agent-github-io.md)
- [Articraft 实体页](../../wiki/entities/articraft.md)
