# The Robotics Map — 机器人产业生态地图（官网）

- **标题：** The Robotics Map: robotics companies across the stack
- **类型：** site / curated industry map
- **官网：** https://roboticsmap.org/
- **维护者：** Sam Padilla、Varun Nair（官网署名）
- **官网数据更新时间：** 2026 年 10 月（网页标注）
- **入库日期：** 2026-11-17
- **一句话说明：** 以 11 个技术层级组织机器人产业公司的公开交互地图，显示层间交叉和企业清单。
- **核对范围：** 已读官网首页的地图读法、层级数量、交叉连接、维护署名及全部 11 个层级的官方分类页（列表与筛选说明）；抽查 Models、Simulation、Data tools、Runtime & ops、Deployments 等企业描述。没有逐一打开 292 家公司的外部网站，也未验证各家商业声明。
- **通读：** 已完成

## 官方站点原文核对（2026-11-17 访问）

官网显示 **292 家公司**、**48 家横跨至少两个层级**（截至官网注明的 2026-10 更新），按从数据来源到现场部署的方向阅读。数字是**该站收录规模**，不是市场总量或第三方审计结果。

| 组别 | 官网层级（分类页） | 官网页面展示数量 |
|---|---|---:|
| Sources | [Capture hardware](https://roboticsmap.org/l/capture-hardware) | 26 |
| Sources | [Data producers](https://roboticsmap.org/l/data-producers) | 16 |
| Inputs | [Components](https://roboticsmap.org/l/components) | 21 |
| Inputs | [Data services](https://roboticsmap.org/l/data-aggregation) | 53 |
| Inputs | [Data tools](https://roboticsmap.org/l/data-tools) | 19 |
| Core | [Simulation](https://roboticsmap.org/l/simulation) | 23 |
| Core | [Models](https://roboticsmap.org/l/models) | 28 |
| Core | [Robots](https://roboticsmap.org/l/robots) | 67 |
| In the field | [Runtime & ops](https://roboticsmap.org/l/runtime) | 22 |
| In the field | [Testing & safety](https://roboticsmap.org/l/evals) | 9 |
| In the field | [Deployments](https://roboticsmap.org/l/deployments) | 60 |

**地图的官方读法：**

1. 每个站点代表产业链层级，旁边数字代表该层级的收录企业数量。
2. 两层之间画线，代表同一企业真实经营的产品覆盖两个层级；线的数字代表重合企业数，线越粗重合越多。
3. 点击层级可查看公司名录与标签/地区筛选；层内环绕的公司仅属于该层，连线上的公司跨层经营。
4. 官方称：同时在四个及以上层级出现的企业不画连线，而是出现在其各个层级内部；当时仅 NVIDIA 属于该情形。
5. 官网显示的最高跨层连接包括 Deployments ⇄ Robots（19）、Data services ⇄ Data tools（4）、Deployments ⇄ Models（4）、Models ⇄ Robots（3）、Runtime & ops ⇄ Simulation（3）。

**官方展示例子（非独立商业核实）：** Simulation 分类有 MuJoCo、Gazebo、Webots、NVIDIA；Models 分类有 Physical Intelligence、Google DeepMind、Hugging Face LeRobot、Skild AI；Deployments 分类有 Waymo、Symbotic、Zipline 等。这些仅说明网站如何分类，不意味着这些公司的产品性能得到本站独立核验。

## 源码与数据开放核查

- 官网首页和所读的分类页**没有列出 The Robotics Map 自身的公开 GitHub 仓库、原始公司数据下载地址或可运行项目入口**；因此不建 `sources/repos/`，也不把地图等同开源项目。
- 官网含公司建议入口，说明新增企业会经人工审核；网页未公开可复现实验基准或准确度审计流程。
- 如后续官网发布数据/API/源码，需要重新核实其许可与数据字段，而非仅凭网页可访问推断开源。

## 对 Robotics_Notebooks 的映射

- 独立实体：[The Robotics Map](../../wiki/entities/robotics-map.md)
- 同类项目：[awesome-physical-ai（aichr）](../../wiki/entities/awesome-physical-ai-aichr.md)、[具身产业库](../../wiki/entities/embodied-industry-db.md)
- 主题入口：[VLA](../../wiki/methods/vla.md)、[Manipulation](../../wiki/tasks/manipulation.md)

## 来源

- [The Robotics Map 官方首页](https://roboticsmap.org/)
- [Models 分类页](https://roboticsmap.org/l/models)
- [Simulation 分类页](https://roboticsmap.org/l/simulation)
- [Deployments 分类页](https://roboticsmap.org/l/deployments)

> 版权与范围：本归档仅记录元数据、结构、少量示例和核对结论，未抓取或重发布官网全部企业列表。
