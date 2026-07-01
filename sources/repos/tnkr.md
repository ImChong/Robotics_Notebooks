# Tnkr

> 来源归档

- **标题：** Tnkr
- **类型：** repo / platform（协作与文档平台，非单一 GitHub 仓）
- **链接：** https://tnkr.ai/
- **入库日期：** 2026-05-28
- **一句话说明：** 面向开源机器人项目的「全栈仓库」：硬件 CAD、布线说明、软件版本、部署与数据/模型在同一项目空间协作，并内置 AI 工程助手 Leonardo。
- **沉淀到 wiki：** [tnkr](../../wiki/entities/tnkr.md)

---

## 为什么值得保留

传统机器人项目常把 **机械（Onshape/SolidWorks）**、**代码（GitHub）**、**数据与部署（Drive/Notion/Slack）** 拆在多个工具里，缺少可版本化、可 fork 的「单一事实来源」。Tnkr 公开叙事定位为 **robotics 的 GitHub**：把四大学科（hardware / electronics / deployment / software）收进同一项目仓库，并强调社区 **rebuild、remix、回传运行数据** 以改进模型。

## 官方与一手资料

| 资料 | 链接 | 说明 |
|------|------|------|
| 官网 | https://tnkr.ai/ | 产品定位、Leonardo、协作工作流（站点有 bot 防护，自动化抓取可能失败） |
| 发布视频 | https://www.youtube.com/watch?v=nLVeWpSb38U | *We built the GitHub for robotics \| Tnkr launch*（@TnkrdotAI） |
| 营销站镜像 | https://tnkrai.framer.website/ | Framer 落地页，文案与官网一致时可作对照 |
| **范例项目** | https://tnkr.ai/open-duck-mini/open-duck-mini-v2 | [Open Duck Mini v2](../sites/tnkr-open-duck-mini-v2.md) 完整 Hardware+Software 文档树 |

## 核心能力（归纳）

### 1. 统一项目仓库

- 接入已有 **GitHub**、**Onshape**（及叙事中的 SolidWorks 等 CAD 生态）。
- 从 CAD 生成 **分步装配说明**；支持在模型/说明上 **标注电路与线束**，避免「只存在老师傅脑子里」的接线知识丢失。
- 在平台内跟踪 **软件提交**，直接浏览关联代码。
- 生成/连接 **仿真资产** 与 **在线部署**，用于数据采集与分析（公开材料未给出具体仿真后端清单，入库时标为「待核实」）。

### 2. 文档与社区协作

- 发布 **开源硬件 / 软件 / 数据贡献指南**，他人可 **重建、remix**，并 **回传操作数据** 改进策略或模型。
- **POV 装配视频 → 分步教程**：将第一视角装配过程转为可检索的步骤文档。
- 项目发现：人形、四足等多类开源机器人项目浏览与贡献入口（以官网分类为准）。

### 3. Leonardo（AI 工程助手）

- 分析 **POV 视频、CAD、代码**，生成 **一键文档**、改进建议与 **实时排障**。
- 叙事能力包括：把模糊操作视频结构化为「拧到第几个孔」级步骤、指出潜在结构问题（第三方报道用语，需以产品实际能力为准）。

### 4. 模型部署（公开叙事）

- 宣称支持将 **视觉或控制策略** 等模型 **一键部署到实机** 做测试（细节、支持硬件列表以官方文档为准）。

## 与常见工具的分工（维护者笔记）

| 维度 | Tnkr（公开叙事） | 常见对照 |
|------|------------------|----------|
| 项目组织 | 硬件+电气+软件+部署+数据/模型一体 | GitHub 偏代码；CAD 在 Onshape；数据在 Drive |
| 学习/训练数据 | LeRobot 等偏 **数据集与策略库** | Tnkr 偏 **整机项目复现与贡献闭环** |
| 机器人描述编辑 | 导入 CAD/装配流 | [URDF-Studio](../../wiki/entities/urdf-studio.md) 偏 URDF/MJCF 专业编辑与 BOM |
| 动捕/数据格式 | 运行数据回传改进模型 | [LeRobot](../../wiki/entities/lerobot.md)、[FreeMoCap](../../wiki/entities/freemocap.md) 等 |

## 对 wiki 的映射

- 新建实体：[tnkr](../../wiki/entities/tnkr.md)
- 建议交叉更新：[lerobot](../../wiki/entities/lerobot.md)、[urdf-studio](../../wiki/entities/urdf-studio.md)、[humanoid-robot](../../wiki/entities/humanoid-robot.md)
- 发布视频摘录：[tnkr_launch_youtube_nlv.md](../blogs/tnkr_launch_youtube_nlv.md)
