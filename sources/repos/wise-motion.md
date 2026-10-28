# Wise Motion

> 来源归档

- **类型：** repo
- **来源：** WiseWong6
- **链接：** https://github.com/WiseWong6/wise-motion
- **入库日期：** 2026-10-27
- **说明：** 根据动作需求匹配现有动效参考，并导出源码及接入说明的 Agent Skill。
- **开源状态：** 技能和自有源码采用 Apache-2.0；第三方代码、字体和素材各有许可。部分原始媒体与 Blender 场景需另行准备，详见仓库 NOTICE 与各工程说明。
- **Wiki：** [Wise Motion](../../wiki/entities/wise-motion.md)

## 资料要点

- README 目录统计为 315 个单个动作、99 个插画单图和 48 个组合片段，共 462 项；此为入库时快照。
- `SKILL.md` 要求仅在明确调用时使用；`match.mjs`、`show.mjs`、`export.mjs` 分别负责检索、查看和导出。
- 本地 `catalog/index.html` 可直接打开；普通条目导出组件接入示例，独立工程条目导出工程文件。导出不会自动安装依赖，遇同名目标文件会停止。
- 字幕、分镜、配色与整片组合由调用方设计；渲染单个动效需准备依赖并使用 Remotion。

## 参考链接

- [README](https://github.com/WiseWong6/wise-motion/blob/main/README.md)
- [SKILL.md](https://github.com/WiseWong6/wise-motion/blob/main/SKILL.md)
- [版权与来源](https://github.com/WiseWong6/wise-motion/blob/main/NOTICE.md)
