# LIFe — Learning Native Interaction via Local Interaction Fields（待核验线索）

> 来源归档（ingest · 待补一手资料；2026-11-09）。本条仅保存用户提供的视频截图中**可直接辨认**的信息，不作为论文全文或已核验项目文档。

- **标题：** LIFe: Learning Native Interaction via Local Interaction Fields
- **类型：** video（视频截图线索，非视频原文）
- **线索来源：** 用户在 2026-11-09 上传的短视频平台截图（原会话附件 `IMG_7308.png`；附件未纳入 Git 仓库）。
- **截图所见账号：** `QihanZhao`（平台展示账号；尚未核实是否作者或项目成员）
- **截图可读文字：** “LIFe”；“Learning Native Interaction via Local Interaction Fields”；“Move by Teleop. Interact Natively.”
- **原始论文：** 待核验；不得将同名或名称相近的 LIFE 工作当作本项目。
- **项目主页：** 待核验。
- **官方代码：** 待核验（未判定已开源、部分开源或未开源）。
- **核对范围：** 仅核对用户上传截图中可见的标题、英文标语及篮球人机交互画面；未取得视频正文、论文 PDF、项目主页或官方源码，故没有核对技术方法、实验结果、作者及开源承诺。
- **通读：** 未完成
- **入库日期：** 2026-11-09
- **状态：** 一手资料缺失；禁止标记为完成/生成基于猜测的技术流程图。

## 截图可直接确认的内容

1. 海报标题为 **LIFe**，副标题为 **Learning Native Interaction via Local Interaction Fields**。
2. 海报标语为 **“Move by Teleop. Interact Natively.”**
3. 画面展示一人与人形机器人之间的篮球交互场景；画面本身**不能证明**机器人是否自主完成交互、采用何种传感与控制方式。
4. 短视频平台账号显示 **QihanZhao**；这只能作为搜索线索，不能推断论文作者/机构。

## 尚待一手证据的关键问题

| 维度 | 当前状态 | 完成要求 |
| --- | --- | --- |
| 作者、机构、日期 | 未核验 | 以论文或项目主页署名为准 |
| Local Interaction Fields 定义与学习目标 | 未核验 | 通读论文的方法与数学定义 |
| Teleoperation 与 native interaction 的接口 | 未核验 | 根据原文绘制 Mermaid 数据/控制流 |
| 训练、传感器和部署方式 | 未核验 | 根据原文实验与配置逐项核对 |
| 评测、消融及局限 | 未核验 | 记录原文表格/实验设置与条件 |
| 源码开放状态 | 未核验 | 打开官方项目页检查真实 GitHub/数据链接 |
| 源码运行时序图 | 暂不适用 | 若有可运行源码，阅读真实入口后补 Mermaid `sequenceDiagram` |

## 与相近名称工作的区别

**不要混淆** 本截图的 LIFe 与 *Learning to Live: A Step Toward Lifelong Embodied Intelligence*（LIFE，arXiv:2611.00180）：两者**标题不同**，目前没有证据证明属于同一项目。后者不是本条 ingest 的一手材料。

## 对 wiki 的映射

- 暂**不创建** `wiki/entities/` 实体节点，避免在没有原始论文和可核验流程时产生低健康度、重复或误导性节点。
- 待取得原始材料并通读后，依照 `schema/ingest-workflow.md`：分别归档 `sources/papers/`、`sources/sites/`、`sources/repos/`（存在时），创建**一个**稳定 `project_id` 的 wiki 实体节点；添加有效内链、原文 Mermaid 主流程图及有代码时的源码时序图。
- 完成 `make ci-preflight`、`npm run lint:mermaid`、`python3 scripts/report_health.py --changed`，以实测 **3/3** 作为可标完成的条件。未达成之前保持 PR 为草稿。

## 核验记录

- **2026-11-09：** 已用精确英文标题、缩写及英文标语查找公开论文/项目页/代码；搜索未返回可确认为该项目的一手原文。
- **判定：** 该搜索结果只能证明**本次未能找到资料**，不能证明项目未发表或代码未公开。不得据此编造 arXiv 编号、项目网址、作者、开源状态或实验数字。
