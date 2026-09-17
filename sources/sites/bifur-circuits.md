# Bifur-circuits（MIT HCIE 项目页）

> 来源归档（ingest · 2026-09-17）

- **标题：** Bifur-circuits: Interactive and Modular Metamaterial Building Blocks Via Bifurcated Geometries
- **类型：** site / project-page
- **官方入口：** <https://hcie.csail.mit.edu/research/Bifur-circuit/bifur-circuits.html>
- **会议：** UIST 2026（ACM Symposium on User Interface Software and Technology）
- **机构：** 麻省理工（MIT）— HCIE / CSAIL；合作含东京大学、密歇根大学
- **入库日期：** 2026-09-17
- **一句话说明：** 机械+电气双模态的 auxetic 超材料积木，经 bifurcation 指数级扩展可稳态构型，内嵌 I2C 拓扑感知，面向可变家具、 tangible 控制器与可重构夹爪原型。
- **开源状态（2026-09-17 核查）：** **待发布** — 项目页列「Open Source / PDF / DOI / Slides」按钮，但 **href 均为空**（同区标注 Coming Soon）；仅 [YouTube 视频](https://youtu.be/eCUYbfXCvME) 可访问。**截至入库日未列 GitHub 或 Zenodo。**

## 页面公开资源

| 资源 | URL / 状态 |
|------|------------|
| 项目页 | <https://hcie.csail.mit.edu/research/Bifur-circuit/bifur-circuits.html> |
| 视频 | <https://youtu.be/eCUYbfXCvME> |
| PDF / DOI / Slides / Code | 按钮占位，链接未发布 |
| MIT News | <https://news.mit.edu/2026/mit-engineers-create-system-for-building-shape-changing-smart-devices-0827> |
| MIT MechE | <https://meche.mit.edu/news-media/mit-engineers-create-system-building-shape-changing-smart-devices> |
| MIT CSAIL | <https://www.csail.mit.edu/news/mit-engineers-create-system-building-shape-changing-smart-devices> |

## 核心摘录（策展）

1. **双层可重构：** 装配级（增删模块）+ 构型级（mechanical bifurcation 过临界阈值分裂为新稳态）。
2. **电气模块化：** 导电 TPU 内布线；任意旋转/压缩/扭转后邻块间仍保持有效 unique circuit；RP2040 PCB 热压于 connector 顶面。
3. **感知：** root 节点 I2C 递归发现拓扑 → JSON 连接图 → Java/C++ 形状识别 UI；20 单元链检测约 5 s（线性缩放）。
4. **耐久：** 10k 次压缩/反向循环后电导无退化（texture analyzer + 万用表抽检）。
5. **工具链：** Fusion 360 参数化单元/connector 导出；Java+Processing 3D 预览与 C++ 交互模板导出。
6. **应用示范：** 24 单元人尺度家具（茶桌/阅读椅/收纳/折叠）；四态 tangible 游戏控制器。

## 对 wiki 的映射

- [`wiki/entities/paper-bifur-circuits.md`](../../wiki/entities/paper-bifur-circuits.md)
- [`sources/papers/bifur_circuits_uist_2026.md`](../papers/bifur_circuits_uist_2026.md)
