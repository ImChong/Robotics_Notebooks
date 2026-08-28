# Previewing the Model Hardware Standard（Anthropic 公告）

> 来源归档

- **标题：** Previewing the Model Hardware Standard
- **类型：** site / news（官方研究预览公告）
- **来源：** Anthropic Beneficial Deployments
- **链接：** https://www.anthropic.com/news/model-hardware-standard-research-preview
- **项目页：** https://modelhardwarestandard.com/
- **发布日期：** 2026-08-27
- **入库日期：** 2026-08-28
- **一句话说明：** Anthropic 与 HHMI Janelia 合作推出 **Model Hardware Standard（MHS）** 研究预览：用标准化驱动（read/write 原语 + 自然语言设备标签）让任意 agent harness 经 MCP / CLI / API 安全操作实验室与产线硬件。
- **开源状态：** **宣称将开源 / 研究预览未公开规范仓** — 公告明确「ahead of making the standard open source」；项目页邀请申请 access。截至 2026-08-28 **无** 公开 GitHub 规范仓或 SDK。见 [项目页归档](./modelhardwarestandard-com.md)。
- **沉淀到 wiki：** [Model Hardware Standard](../../wiki/concepts/model-hardware-standard.md)、对照 [MCP](../../wiki/concepts/model-context-protocol.md)

---

## 抓取说明

- 以 **2026-08-28** 对 Anthropic News 页公开 HTML 正文抽取为准。
- 合作方案例与厂商名单以公告当日为准；后续预览进展需回项目页核对。
- 申请入口以项目页 “Apply for access” 为准，勿把新闻页内嵌按钮 URL 当稳定 API。

---

## 问题与方案

| 主题 | 摘要 |
|------|------|
| **痛点** | 实验室/产线设备各有私有编程接口，互不通信；接 AI 更要为每个设备写翻译器。集成常以周/月计。 |
| **主张** | MHS 把集成压到小时/分钟级；agent 可并行编排显微镜、液体处理、机械臂等，并在部分故障上自行恢复。 |
| **范围** | 任何 **有可编程接口** 的设备；**模型无关**；任意 agent harness 可用 MCP 等标准协议接入。 |
| **起源** | Alek Kemeny（Anthropic Beneficial Deployments）与 Arco Bast（HHMI Janelia 博士后）：Bast 先做共享内存字典让异厂商激光/调焦/相机以内存速度互通，再与 Kemeny 把 AI 接进去。 |

### 驱动如何工作（公告语义）

1. **标准化驱动** 把 OS 与设备之间翻译成少量原语：如 `read`（读温度）/ `write`（设温度）。
2. 设备以标准格式 **可发现**，网络上 agent 与设备不必再写定制翻译器。
3. **自然语言 tags** 写入代码里读不到的物理特性（例如机械臂质量，用于安全操作）；用户可手写或由 agent 访谈采集。驱动据此自动生成参考文件：可测什么、可调什么、**强制安全限**。
4. 控制通路三条，可并用：**MCP**、**CLI**、**代码文件（API）**。长任务或比在线推理更快的操作，把驱动命令链进确定性脚本，设备自行跑完。

Claude 在测试中表现出科学家式探索：调激光 → 看相机里光斑如何动 → 把学到的序列打成脚本，整段对齐变成一条命令。

---

## 早期合作案例（公告）

| 伙伴 | 做什么 |
|------|--------|
| **Genentech** | BCA 蛋白定量：液体处理 + 机械臂 + 读板仪 PoC |
| **华盛顿大学 Baker / Pinglay labs** | 远程仪表盘；agent 监督 qPCR 看扩增曲线适时停止；臂与液体处理无碰撞递板 |
| **CMU** | 串稀释剂量–反应，约 **3×** 快于人工；四类仪器跨三台互不兼容电脑 |
| **HHMI Janelia** | 把原先 7 套厂商程序、无共享接口的显微镜 rig 统一编排（Ahrens lab / Virginie Ruetten） |
| **QuEra Computing** | 中性原子量子计算机激光锁频；agent 写的控制器 **99.3%** 无人工恢复 lock |
| **Tetsuwan Scientific** | 经 ResearchOS 跑 qPCR，服务加州 San Pedro Creek 污染公民科学 |

### 厂商 / 平台侧（「正在加 MHS 支持」）

AWS Strands Robots（预览期私有预发布包）、Automata LINQ、Danaher、Doosan Robotics、MBF Bioscience ScanImage、QIAGEN QIAsymphony Connect、Tecan Fluent、Universal Robots。下一阶段点名 **Hugging Face LeRobot** 与 **Raspberry Pi**（Camera MHS Driver 测试成功）。

---

## 局限（公告自述）

- Claude 从文本/图像学物理世界，**空间与物理推理仍需专家盯场**。Genentech：泡沫导致的物理失败会被模型当成软件 bug，必须人教「这是物理，要用物理手段修」。
- **没有编程接口的硬件** 尚不支持；在与厂商做内置驱动。
- 研究预览用于补安全评测与 **physical safety roadmap**；开源时会发布预览期发现作为安全部署指南。

---

## 对 wiki 的映射

| 主题 | 目标 wiki |
|------|-----------|
| MHS 协议概念 | `wiki/concepts/model-hardware-standard.md` |
| 与 MCP 的软件/硬件分工 | `wiki/concepts/model-context-protocol.md` |
| LeRobot 将加 MHS | `wiki/entities/lerobot.md` |
| 实验室/产线臂编排 vs 学习式 VLA | `wiki/tasks/manipulation.md`、`wiki/concepts/llm-robotics-control-interfaces.md` |
| 物理访问安全 | `wiki/concepts/safety-filter.md` |

## 参考链接

- <https://www.anthropic.com/news/model-hardware-standard-research-preview>
- <https://modelhardwarestandard.com/>
- MCP 对照：<https://www.anthropic.com/news/model-context-protocol>
