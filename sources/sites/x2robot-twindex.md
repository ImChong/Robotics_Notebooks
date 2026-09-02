# TwinDEX Project Page（自变量机器人）

> 来源归档

- **标题：** TwinDEX — a twinned system for dexterous manipulation from robot-free data collection and policy deployment
- **类型：** site / project page（硬件共设计 + 无本体采数产品/研究页）
- **URL：** <https://x2robot.com/pages/twindex>（英文：<https://x2robot.com/en/pages/twindex>）
- **新闻稿：** <https://www.prnewswire.com/news-releases/twindex-introduces-a-scalable-path-from-robot-free-data-collection-to-real-world-dexterous-manipulation-302867559.html>（2026-09-02）
- **机构：** 自变量机器人（X Square Robot）
- **核心贡献：** Ian Huang\*†、Jing Shu\*；通讯 Hao Wang‡、Qian Wang（\* 同等贡献；† 项目负责人）
- **日期：** 项目页与新闻稿 2026-09-02
- **入库日期：** 2026-09-02
- **一句话说明：** 可穿戴三指外骨骼与同构机器人末端成对共设计；宣称用纯 robot-free 数据训策略，数据效率与真机遥操作重叠，采数吞吐约 **5.3×**。页面为 Next.js 前端渲染；开源判断以页内链接与 GitHub 组织检索为准。

## 开源核查（2026-09-02）

| 入口 | 状态 |
|------|------|
| Homepage | 已挂链 — <https://x2robot.com/en/pages/twindex> |
| Code / GitHub | **未列**。页内 HTML 无 `github.com` / `huggingface` / `arxiv.org`；GitHub 检索 `twindex` 机器人仓 **0** 条；HF 模型检索 **0** |
| Paper | 页底 **BibTeX: Coming soon.**；正文称精度/同步消融见 *technical report*，入库日 **无** 公开 arXiv / PDF |
| Checkpoints / 数据集 | **未列** HF / Zenodo / 下载入口 |
| 媒体 | 演示视频在阿里云 OSS `x2robot-open.oss-cn-shenzhen.aliyuncs.com/twindex/`（概览 / cap-twisting / 化学实验等） |
| 同机构相关仓（**不是 TwinDEX**） | [X-Square-Robot/XRZero-G0](https://github.com/X-Square-Robot/XRZero-G0) 为 2026-04 **VR+专用夹爪** 无本体采数（arXiv:2604.13001，HF `XRZero-G0-3K`）；[sdk_hand](https://github.com/X-Square-Robot/sdk_hand) 为 **ZBL 五指手** ROS 2 SDK。二者均 **未** 从 TwinDEX 页链出，勿写成 TwinDEX 复现入口 |

**开放程度：确认未开源。** 无训练/推理代码、无 CAD/固件、无数据集；论文/技术报告待发布。

## 页面内容要点

- **形态：** 三指（拇指 / 食指 / 中指），**9 DoF、7 主动 + 2 被动**。对 4 / 6 / 7 / 8 主动配置做 dexterity benchmark 后选定 7 主动：相对 6 主动有跃迁，相对 8 主动边际主要在穿戴舒适。
- **三原则：** Dexterity（形态取舍）· Consistency / Correspondence（采数端与部署端对齐）· Scalability（可穿戴 + 无机器人）。
- **Correspondence 五维：** 运动学同构（DoF / 关节轴 / 连杆比例；外骨骼转轴与人手共轴）· 接触力学（同材料/几何/表面 + 同位置触觉）· 外观（接触壳体一致，非接触结构蒙布）· 采数精度（关节 / 腕定位 / 相对 vs 绝对 / jitter / drift，细节在技术报告）· 模态间时间同步（视觉、触觉、关节、腕定位）。
- **标定可视化：** 用腕位姿 + 手指关节把双手 URDF 重投影到头相机，重叠度反映运动学校准与跨模态同步。
- **主张数字：** 采数有效吞吐相对真机遥操作 **>5.3×**；robot-free 与 on-robot 的 **data-efficiency 曲线重叠**（宣称统计显著范围内）；策略 **仅** 用 robot-free、无干预/真机遥操作数据。外骨骼结构上兼容真机遥操作，留作未来补干预接口，**不是** 当前结果的前提。
- **任务：** cap twisting、扫帚簸箕、抽翻笔记本、工具箱卡扣、注射器；长程 **标准化化学实验** 单段未剪辑（Overview 写 **25** 子动作，Conclusion 与新闻稿写 **24**——归档时保留该不一致）。
- **局限（页内自述）：** 目前限于单桌面、物体类别有限；长时穿戴人体工学未系统研究；更高灵巧（多指精密装配）可能要加 DoF。
- **新闻稿补强：** 同步多视角 RGB、6-DoF 腕位姿、手指关节、指尖触觉；「数百条」robot-free episode、无 on-robot 训练数据。

## 对 wiki 的映射

- 沉淀 **[`wiki/entities/twindex.md`](../../wiki/entities/twindex.md)**
- 交叉：[Teleoperation](../../wiki/tasks/teleoperation.md)、[灵巧操作数据采集指南](../../wiki/queries/dexterous-data-collection-guide.md)、[数据手套 vs 视觉遥操作](../../wiki/comparisons/data-gloves-vs-vision-teleop.md)、[mimic U1](../../wiki/entities/mimic-wearable-u1.md)、[DEUX / Glove X](../../wiki/entities/xyz-deux.md)、[HandUMI](../../wiki/entities/handumi.md)
