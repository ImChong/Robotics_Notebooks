# capfab/seebfm-media — SeeBFM 官方网站与仿真回放源码

> 来源归档（ingest；此仓库是演示应用，不是 SeeBFM 训练控制实现）

- **仓库：** <https://github.com/capfab/seebfm-media>
- **类型：** GitHub Pages / Three.js 交互回放和项目视频资源
- **开源程度：** 官方媒体与 Playground 的 JavaScript 浏览器实现已公开；未看到 `train.py`、训练配置/策略推理入口、策略权重或真实机器人部署栈。
- **核查日期：** 2026-11-11
- **核对范围：** README.md；`playground/index.html`、`playground/viewer.js`（加载、回放、UI、typed 匹配、初始化）、`playground/match.js`；`slides/spec.js`（全部 15 段）及 `media/overview.vtt`。已确认浏览器回放调用关系；未把这些文件视作控制策略实现。
- **通读：** 已完成

## 实际回放入口（按公开源码）

1. `playground/index.html` 用 ES module 导入 `viewer.js` 和 Three.js。
2. `viewer.js` 读取 `data/index.json` 的 models/prompts/cells/attempts 映射，读取 `data/robot.json` 和 `data/robot.bin` 初始化网格。
3. 用户选模型/提示/地形/尝试后，`start()` 加载 `data/a/${key}.bin.gz` 与 `data/cells/${id}.bin.gz`；`decodeAttempts()` 解码时序关节、根位姿，`terrain.js` 载入地图网格。
4. `frame()` 插值动作，`fk.js` 的 `forward()` 进行前向运动学，`setPose()` 更新 Three.js Mesh，`renderer.setAnimationLoop` 连续渲染。
5. `match.js` 用 `Xenova/all-MiniLM-L6-v2` 浏览器嵌入查找最近的预录句子（阈值 0.48）；这不是实时调用语言模型生成任意新动作，`index.html` 明示 “closest sentence we ran”。

## 注意事项

- Playground 显示录制的 attempt（含失败的 crossing/stayed/fell），不进行实时物理求解或策略前向推理。
- 网站展示的高度扫描点可由网页的 `surface.height` 计算并可视化，不等于原训练模型在浏览器上运行。
- 该代码来源能支持**网站回放时序图**，不足以支持 **SeeBFM 策略训练/推理时序图**。
- [项目页归档](../sites/seebfm-project-page.md) · [论文状态归档](../papers/seebfm_preprint_2026.md) · [知识节点](../../wiki/entities/paper-seebfm.md)
