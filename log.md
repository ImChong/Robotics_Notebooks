# Change Log

## 2026-04-07

### 重构启动
- 将项目方向从“思维导图资源堆叠”转向“机器人研究与工程知识库”
- 新增 `AGENTS.md`
- 新增 `schema/` 规则目录
- 新增 `sources/`、`wiki/`、`exports/` 骨架
- 新增首批 MVP wiki 页面
- 暂不修改当前思维导图网页渲染逻辑

### 设计决策
- 底层知识组织采用 wiki / 图结构，而不是强制树结构
- 展示层（思维导图）后续通过 `exports/` 从 `wiki/` 导出
- `paper-reading` 与 `Robotics_Notebooks` 分工明确：前者深读论文，后者整理知识图谱
