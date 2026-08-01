# Holotile — Disney Research（项目页）

> 来源归档（ingest · 机构项目页）

- **标题：** Holotile
- **类型：** site（硬件 / 沉浸式交互研究项目页）
- **机构：** Disney Research（Los Angeles）· Walt Disney Imagineering R&D
- **官方入口：** <https://la.disneyresearch.com/holotile/>
- **入库日期：** 2026-08-01
- **一句话说明：** 由 Lanny Smoot 主导的全向活动地板：六边形阵列可做被动全向行走与主动编程运动，面向 VR「无限步行」与物体「遥移」类演示。

## 为什么值得保留

- 官方 Holotile 专页是 Disney Research LA 在 **Immersive Technologies** 方向上少有的硬件项目锚点，与同站 [Research 总览](disney-research-la.md) 及角色机器人线（Olaf / BDX）形成「乐园/沉浸体验 × 机器人」对照。
- 项目页本身几乎无实现细节；工程机理需结合 **专利 US10416754B2** 与公开演示/媒体报道归纳，便于后续做全向跑步机 / 多人 VR 地面对比时有可溯源入口。

## 开源状态（2026-08-01 核查）

| 项 | 结论 |
|----|------|
| 项目页资源 | 仅短文案介绍；无 PDF / 数据集 / API 文档链接 |
| 代码 / CAD / 固件 | **未开源**（页面与站点导航无 GitHub / HF / Zenodo） |
| 专利 | [US10416754B2](https://patents.google.com/patent/US10416754B2/en)（发明人含 Lanny S. Smoot 等；2019-09-17 授权） |
| 演示视频 | 第三方报道嵌入 YouTube（如 [designboom 报道所用片段](https://www.youtube.com/watch?v=68YMEmaF0rs)）；非仓库可复现资产 |
| 复现 | 以专利描述 + 公开演示为准；wiki **不写安装/BOM 步骤** |

## 项目页要点（原文归纳）

- 地板由大量 **hexes（六边形单元）** 组成，可支持：
  - **passive omnidirectional locomotion**（被动全向行走）
  - **active, programmed movement**（主动、可编程运动）
- 叙事灵感来自 *Star Trek* **Holodeck**：在有限房间内「任意方向无限行走」。
- 用例叙事覆盖 **VR metaverse** 与物体 **「telekinesis」**（地面主动搬运物体）。
- 创建者标注为 Disney Research Imagineering legend **Lanny Smoot**，并由更广 R&D 团队支持；持续探索公司内外用例。

## 专利 / 机理锚点（非项目页正文，供 wiki 编译）

- **专利标题：** *Floor system providing omnidirectional movement of a person walking in a virtual reality environment*
- **核心结构（专利摘要归纳）：** 模块化主动地砖上的大量 **倾斜接触盘（contact disks）**；通过 **盘朝向（orient）** 设定抬起接触段方向，再 **绕轴旋转** 使支撑物沿任意水平方向移动；控制器可对多用户/多物体做独立路径管理，避免撞墙或互撞。
- **与传统全向跑步机的对比动机（专利背景）：** 单用户皮带式全向跑步机、碗形滑面、球形笼等难扩展到多人、且噪声/安全包络差；Holotile 路线强调 **模块地砖 + 盘阵列**。

## 对 wiki 的映射

1. **[Disney Holotile（实体页）](../../wiki/entities/disney-holotile.md)** — 主升格页
2. **[Disney Research LA（机构实体）](../../wiki/entities/disney-research-la.md)** — 研究总览枢纽
3. 交叉：[Character Animation vs Robotics](../../wiki/concepts/character-animation-vs-robotics.md)、[Locomotion](../../wiki/tasks/locomotion.md)（对照「机器人自身行走」vs「地面代偿行走」）

## 推荐继续阅读

- [Disney Research Holotile 专页](https://la.disneyresearch.com/holotile/)
- [US10416754B2（Google Patents）](https://patents.google.com/patent/US10416754B2/en)
- [Fast Company：Imagineer 讲解 HoloTile](https://www.fastcompany.com/91019277/a-disney-imagineer-explains-how-they-made-the-holotile-floor-a-magical-walkway-that-moves-in-any-direction)
