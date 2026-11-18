# ffmpeg-skill 官方主页

> 原始资料归档 · 项目官网

- **标题：** ffmpeg-skill: give your coding agent a video editor
- **类型：** site
- **链接：** https://kajisho5.github.io/ffmpeg-skill/
- **代码：** https://github.com/kajisho5/ffmpeg-skill （**已开源，MIT**；对应[仓库归档](../repos/kajisho5-ffmpeg-skill.md)）
- **入库日期：** 2026-11-18
- **知识页：** [ffmpeg-skill](../../wiki/entities/ffmpeg-skill.md)
- **核对范围：** 通读官方主页的 Demos、How it works、Tools、For agents、Install、FAQ 与文末源码/文档入口；安装和限制以 GitHub README、SKILL 和源码再次核对。
- **通读：** 已完成

## 官网核心信息

1. **定位：** 将本地 FFmpeg 的 42 项媒体处理能力包装为 coding agent 的执行工具，支持 Claude Code / Cursor / Codex / MCP。
2. **流程：** 实测视频参数（`probe`）→ 尽量无损操作 → 平台约束检查（`check`）→ 核实输出并检查缩略图（`look`）。
3. **结构化调用：** 通过工具自身的 argparse 生成 MCP schema；MCP 默认列 12 项，按名称可调用全部 42 项。
4. **边界：** 自动化检测可以验证时长、码率、音频电平与容器等参数，不能凭自动校验判断视频观感是否达到预期。
5. **安装需求：** FFmpeg 6.1+、Python 3.10+；`npx ffmpeg-skill` 为默认入口，`--cursor`、`--codex`、`--all` 为其他安装目标。
6. **与 Remotion：** 该项目面向**已有素材的裁剪与交付**，Remotion 用 React 描述并渲染视频，两者可以组合，不能互相替代。

## 核查说明

官网演示的 59.98 秒、1080×1920、30fps 为**展示的示例输出**，不是本次独立执行的结果。官网有完整的源代码与文档链接，并非仅发布概念/演示视频。

## 一手入口

- [官方主页](https://kajisho5.github.io/ffmpeg-skill/)
- [GitHub README](https://github.com/kajisho5/ffmpeg-skill/blob/main/README.md)
- [技术契约说明](https://github.com/kajisho5/ffmpeg-skill/blob/main/docs/contract.md)
