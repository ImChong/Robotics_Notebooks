# img2threejs（img2threejs/img2threejs）

> 来源归档

- **标题：** img2threejs — Image → procedural Three.js reconstruction
- **类型：** repo（Agent Skill + 分阶段重建工具链）
- **作者 / 维护方：** img2threejs（README 当前署名与组织页）
- **链接：** <https://github.com/img2threejs/img2threejs>
- **官方演示画廊：** <https://img2threejs.io/>（前身画廊：<https://hoainho.github.io/img2threejs-showcase/>）
- **入库日期：** 首次 2026-07-23；本次核对 2026-10-22
- **一句话说明：** 将单张参考图逐阶段重建为可审阅、可编辑的程序化 Three.js TypeScript 模型；采用 reconstruction-by-code，不以摄影测量或网格提取为核心。
- **当前版本：** 2.0.0（主仓 `SKILL.md` frontmatter）
- **GitHub 指标快照：** 18,663 stars、1,570 forks（GitHub API，2026-10-22；数字会变化）
- **开源状态：** **已开源** — 主仓许可证为 Apache-2.0；仓库包含 `SKILL.md`、`forge/`、`grimoire/`、文档与脚本。配套插件通过单独仓库分发，个别插件可能需要授权访问。
- **项目身份：** 当前官方主仓为 `img2threejs/img2threejs`；早期归档的 `hoainho/img2threejs` 是该项目的旧仓库地址，不另建实体。
- **沉淀到 wiki：** 是 → [`wiki/entities/img2threejs.md`](../../wiki/entities/img2threejs.md)

## 为何值得保留

- **图像到可编辑代码资产：** 与直接生成网格文件的服务不同，主流程生成 `ObjectSculptSpec` 与 TypeScript Three.js 工厂，适合检查、修改和纳入代码版本管理。
- **质量门控可复用：** 规约和 Python 脚本分别处理视觉评审与确定性检查，阶段门可在模型生成过浅、几何错误或证据不足时阻止继续。
- **机器人应用边界明确：** 可用于仿真场景的外观道具、展示和原型；项目未宣称输出具备机器人动力学、惯量、接触参数或 URDF/MJCF 关节语义。

## README / SKILL 摘要（核对于 2026-10-22）

- **定位：** Agent Skill，支持 Hermes、Claude Code、Codex、OpenCode；宿主提供视觉判断与浏览器预览能力。
- **安装：** `npx --yes img2threejs@latest install`；基础技能不是 `img2` 插件，勿通过 `img2 add` 安装。
- **环境：** Node.js 18+、Git、Python 3.10+；`forge/` 核心脚本仅使用 Python 标准库。
- **项目状态：** 主仓 `SKILL.md` 标为 2.0.0；v2.0 引入独立的插件 harness 与域插件架构。
- **核心流程：** 图片适用性检查 → 类别 / 复杂度评估与 quality contract → `ObjectSculptSpec` → 严格质量校验 → 逐 pass 生成 → 浏览器截图与参考图对比 → 确定性指标和 agent vision 评审 → 继续或修正。
- **构建阶段：** blockout、structural、form、material、surface、lighting、interaction、optimization；每阶段通过门控后再进入下一阶段。
- **主要产物：** 完整组件与材质规格 JSON；返回 `THREE.Group` 的 TypeScript 工厂；运行时层级可携带 pivots、sockets、colliders 等信息。角色工作流还可输出 rig / skeleton 数据。
- **插件边界：** `plugin-character`、`plugin-cs2` 等为独立分发的域插件；README 列出的 `plugin-img2glb` 是经托管 TRELLIS 服务输出 GLB 的可选插件路径，与基础的代码重建管线不同。
- **单图限制：** 参考图不可见的背面和内部结构只能推断；应报告近似区域与置信度，不能把视觉相似度等同于测量真值。
- **协议：** Apache License 2.0。

## 与机器人研究和工程的映射

- 可为 Three.js / WebGL 演示、仿真场景外观及合成内容原型生成可读模型代码。
- 导入 Isaac、MuJoCo 或其他机器人仿真时，仍需自行核对尺度、坐标系、碰撞几何、质量 / 惯量、接触参数和关节定义；该工具的动画层级不等于机器人运动学模型。
- 和 [Articraft](../../wiki/entities/articraft.md) 等仿真就绪资产路线对比时，应把它看作图像到程序化图形代码的工具，而非物理校准资产生成器。

## 对 wiki 的映射

- 主实体页：[`wiki/entities/img2threejs.md`](../../wiki/entities/img2threejs.md)
- 官方在线演示归档：[`sources/sites/img2threejs-io.md`](../sites/img2threejs-io.md)
- 旧版画廊归档：[`sources/sites/img2threejs-showcase.md`](../sites/img2threejs-showcase.md)
- 相关主题：[`wiki/entities/3dgenstudio.md`](../../wiki/entities/3dgenstudio.md)、[`wiki/entities/articraft.md`](../../wiki/entities/articraft.md)、[`wiki/concepts/sim2real.md`](../../wiki/concepts/sim2real.md)

## 参考链接

- 主仓与快速开始：<https://github.com/img2threejs/img2threejs>
- Skill 入口：<https://github.com/img2threejs/img2threejs/blob/main/SKILL.md>
- 架构说明：<https://github.com/img2threejs/img2threejs/blob/main/docs/ARCHITECTURE.md>
- 官方演示画廊：<https://img2threejs.io/>
- Apache-2.0 许可证：<https://github.com/img2threejs/img2threejs/blob/main/LICENSE>
