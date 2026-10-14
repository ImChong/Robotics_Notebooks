# LiteReality-Agent 项目页与演示

- **项目主页：** <https://litereality.github.io/agent/>
- **论文：** [arXiv:2610.01863](https://arxiv.org/abs/2610.01863) · [HTML 全文](https://arxiv.org/html/2610.01863v1)
- **交互式 3D 查看器：** <https://litereality.github.io/agent/vr/index.html>
- **代码：** [LiteReality/LiteReality-Agent](https://github.com/LiteReality/LiteReality-Agent)（公开，Apache-2.0）
- **配套扫描 App：** [LiteReality Scanner on App Store](https://apps.apple.com/gb/app/litereality/id6774158260)（免费）
- **相关项目博客：** <https://litereality.github.io/agent/litereality-agent-post/>
- **实体页：** [litereality-agent](../../wiki/entities/litereality-agent.md)
- **论文归档：** [sources/papers/litereality_agent_2610_01863.md](../papers/litereality_agent_2610_01863.md)

## 一句话说明

LiteReality-Agent 将 iOS RGB-D 房间扫描转成可编辑、可交互并可导入 MuJoCo 的室内场景；coding agent 在 Room.py 中按扫描证据迭代，使用独立的几何、碰撞与支撑关系质量门，并提供浏览器查看器及仿真导出。
