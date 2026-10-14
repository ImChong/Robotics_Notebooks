# LiteReality-Agent（LiteReality/LiteReality-Agent）

- **代码：** <https://github.com/LiteReality/LiteReality-Agent>（公开，Apache-2.0）
- **项目主页：** <https://litereality.github.io/agent/>
- **交互式查看器：** <https://litereality.github.io/agent/vr/index.html>
- **论文：** [arXiv:2610.01863](https://arxiv.org/abs/2610.01863)
- **扫描 App：** [LiteReality Scanner](https://apps.apple.com/gb/app/litereality/id6774158260)（App Store 免费）
- **示例扫描：** [LiteReality/example-scans](https://github.com/LiteReality/example-scans)
- **实体页：** [litereality-agent](../../wiki/entities/litereality-agent.md)
- **论文归档：** [litereality_agent_2610_01863](../papers/litereality_agent_2610_01863.md)

## 一句话说明

公开端到端室内重建与仿真工具链：uv run litereality run scans/<scan> 执行 scene init 与 agentic authoring；uv run litereality stage simulate run/<scan> 输出 MuJoCo 场景，并提供 URDF / MJCF 导出。依赖 Blender 5.x、图像生成 API、Agent CLI，以及 Modal 托管模型或 Linux ≥24 GB GPU。

## 开源状态

- **已开源：** Agent CLI、代码与部署脚本；仓库采用 Apache-2.0。
- **已发布：** LiteReality Scanner App Store 应用，官方 README 标注免费。
- **可试跑数据：** example-scans 仓库提供公开扫描样例。
- **仿真范围：** 可导出 MuJoCo / URDF / MJCF；物理属性仍需按具体用途验证。

## 交叉链接

- [LiteReality-Agent 实体页](../../wiki/entities/litereality-agent.md)
- [论文来源](../papers/litereality_agent_2610_01863.md)
- [项目页来源](../sites/litereality-agent-github-io.md)
- [Articraft 实体页](../../wiki/entities/articraft.md)
