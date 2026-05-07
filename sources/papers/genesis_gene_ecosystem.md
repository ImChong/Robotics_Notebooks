# genesis_gene_ecosystem

> 来源归档（ingest）

- **标题：** Genesis 生态辨析：GENE-26.5（Genesis AI）与 Genesis 物理仿真（Genesis-Embodied-AI）
- **类型：** paper / blog / product-brief（混合）
- **来源：** arXiv、公司官网、行业媒体、开源仓库
- **入库日期：** 2026-05-07
- **最后更新：** 2026-05-07
- **一句话说明：** 归档「GENE-26.5 机器人基础模型」公开信息与「Genesis 生成式通用物理引擎」论文及工程链接，并在仓库内与同名品牌做明确区分。

## 名称辨析（必须先读）

机器人社区里存在两条常被混称的 **Genesis** 线索：

| 线索 | 主体 | 典型产物 | 本仓库 wiki 映射 |
|------|------|-----------|-----------------|
| **A. Genesis-Embodied-AI** | 学术与开源社区主导的仿真项目 | 开源物理引擎与仿真平台、技术报告与 arXiv 论文 | [genesis-sim](../../wiki/entities/genesis-sim.md) |
| **B. Genesis AI（公司）** | 全栈机器人公司（美国 San Carlos 等） | 闭源 **GENE-26.5**「机器人大脑」与硬件/数据管线 | [gene-26-5-genesis-ai](../../wiki/entities/gene-26-5-genesis-ai.md) |

二者英文品牌相近，但技术栈与开放程度不同；写论文或做选型时应核对作者机构与仓库域名。

## 核心资料摘录

### 1) GENE-26.5（Genesis AI 公司，2026 年前后公开发布）

- **公司官网：** <https://genesis-ai.company/>
- **行业报道（The Robot Report，2026-05）：** <https://www.therobotreport.com/genesis-ai-introduces-gene-foundation-model-more-dexterous-manipulation/>
- **财经媒体转载（Yahoo Finance）：** <https://finance.yahoo.com/sectors/technology/articles/genesis-ai-unveils-gene-26-130000684.html>
- **核心公开主张（综合上述来源，非经同行评审）：**
  - GENE-26.5 定位为面向物理操作的 **机器人基础模型**，强调长时域、双手协调与高灵巧任务（烹饪多步流程、实验操作、线束整理、魔方、钢琴等演示场景）。
  - 与模型并列宣传的包括：**人形尺度灵巧手**、以及宣称更低成本、更高采集效率的 **数据手套 / 数据引擎**，用于缓解机器人学习中的数据瓶颈。
- **局限：** 截至本入库日，未检索到与 GENE-26.5 同名的 **arXiv / 顶会论文**；技术细节以公司与媒体报道为主，适合作为「产业动向」引用，不宜与已发表论文混排为同一证据等级。
- **对 wiki 的映射：**
  - [gene-26-5-genesis-ai](../../wiki/entities/gene-26-5-genesis-ai.md)

### 2) Genesis: A Generative and Universal Physics Engine for Robotics and Beyond（Genesis Team, arXiv 2024）

- **arXiv：** <https://arxiv.org/abs/2412.12919>
- **项目主页（能力展示 + BibTeX）：** <https://genesis-embodied-ai.github.io/>
- **代码仓库：** <https://github.com/Genesis-Embodied-AI/Genesis>
- **文档：** <https://genesis-world.readthedocs.io/en/latest/index.html>
- **IBM Research 对 ICRA 2025 收录版本的条目（含 DOI 链）：** <https://research.ibm.com/publications/genesis-a-generative-and-universal-physics-engine-for-robotics>
- **速度对比技术报告（仓库）：** <https://github.com/zhouxian/genesis-speed-benchmark>
- **核心贡献（提炼）：** 在统一框架内集成多类物理求解与耦合；强调 GPU 并行与工程 API；上层规划逐步开放「生成式智能体」以自动化多模态数据（与 RoboGen 等研究线衔接，见项目页引用）。
- **对 wiki 的映射：**
  - [genesis-sim](../../wiki/entities/genesis-sim.md)
  - [simulation_tools](./simulation_tools.md)（已有条目，本文件作为总档交叉引用）
  - [isaac-gym-isaac-lab](../../wiki/entities/isaac-gym-isaac-lab.md)
  - [sim2real](../../wiki/concepts/sim2real.md)

### 3) 相关研究入口（非 Genesis 本体论文，常被同一项目页引用）

- **RoboGen（生成式机器人技能管线展示页）：** <https://robogen-ai.github.io/>

## 当前提炼状态

- [x] 名称辨析与链接核对
- [x] wiki 页面映射确认
- [ ] 若后续出现 GENE-26.5 正式技术报告或论文，应在本文件追加独立条目并更新实体页「参考来源」
