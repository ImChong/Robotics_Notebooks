# Spatial and Semantic Reasoning for LLM-Driven Robot Navigation via MCP

> 论文来源归档（ingest）

## 元数据

- **题目：** Spatial and Semantic Reasoning for LLM-Driven Robot Navigation via MCP
- **中文题目：** 基于 MCP 的大语言模型驱动机器人导航空间与语义推理
- **作者：** Jungsoo Lee, Jaegyun Park, Wansoo Kim
- **机构：** 汉阳大学、汉阳大学 ERICA 校区
- **类型：** paper
- **预印本：** https://arxiv.org/abs/2609.27340v1
- **全文 HTML：** https://arxiv.org/html/2609.27340v1
- **PDF：** https://arxiv.org/pdf/2609.27340v1
- **DOI：** https://doi.org/10.48550/arXiv.2609.27340
- **提交日期：** 2026-09-23（v1）
- **核对日期：** 2026-11-24
- **核对范围：** arXiv HTML v1 摘要、§1–6、式 (1)–(8)、图 1–4 的说明、表 1–3、探索提示词附录与参考文献。
- **通读：** 已完成
- **官方实验室报告：** https://harco.hanyang.ac.kr/2026/10/29/news_iccas2026.html （ICCAS 2026 报告）
- **代码 / 数据：** 截至核对日，论文与实验室报告未列本研究可运行实现或数据下载入口。论文引用的 https://github.com/robotmcp/ros-mcp-server 是第三方通信底座，不是作者新增表示层的源码。

## 论文摘录（归纳）

- **问题：** ROS 数值占据栅格不便直接供多模态模型推断空间结构；定制接口也限制复用。
- **方法（§3）：** 将地图渲染为含米制网格和机器人姿态的图像，同时保存带观察位姿的 YAML 语义路点。借 MCP 工具访问这些表示和既有 ROS 接口。
- **实验（§4，表 3）：** Isaac Sim Nova Carter、Cartographer、Nav2；Claude Sonnet 4.6 / GPT-5.5 的建图覆盖率分别为 97.83% / 99.78%，最小房间任务成功 26/30 / 28/30，灭火器关联路点任务均为 30/30。
- **边界（§5）：** 仅验证室内仿真；房间大小和走廊辨识仍可能出错；语义目标为观察路点而非物体精确坐标，完整标注条件下的结果不能外推到稀疏或动态环境。
- **对 wiki 的映射：**
  - [论文实体](../../wiki/entities/paper-mcp-spatial-semantic-navigation.md)

## 复现说明

附录给出逐路点探索、导航确认、相机观察及语义写入的提示词流程；论文未给新增工具的公开运行入口。不得以第三方 ROS-MCP Server 的源码替代论文实现。
