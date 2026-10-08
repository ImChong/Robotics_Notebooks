# 公司路线资料补齐 v1

- 基线：`main` / `aaa4fec74`（2026-10-05 核查）。
- 范围：公司路线现有 16 家公司的代表作品、已归档工程入口、详情溯源与版本边界。
- 原则：不宣称收录全部公司资料；发布日期不以入库日代替；API、代码、权重、数据分别核查。

## 覆盖与处理

| 公司 | 本轮处理 | 仍有的公开信息边界 |
| --- | --- | --- |
| Physical Intelligence | FAST / MEM / RLT 从策展入口补为机制、评测读法、结论与实现边界 | MEM / RLT 官网访问 403，依据原始论文；完整官方实现未确认 |
| 1X | 独立 Redwood AI 策略页，与同名 World Model 分开 | 官网策略发布页未列模型资产 |
| Figure | 补实 Helix 初版，新增 Helix 02 架构与日期；连回公司页 | 模型代码、权重和训练数据未见公开入口 |
| Skild AI | 保留已有 Brain、S1、自博弈后训练详情与路线 | 公开叙述不等于全套模型可复现 |
| Google DeepMind | 增补 Robotics 1.5 路线、1.0→1.5→2 历史及 API 边界 | VLA 模型与 ER API / 编排示例分开 |
| NVIDIA | 增补 N1.7 工程流程、Cosmos Curator / Transfer / Cookbook | 工程文章日期不视为 N1.7 首发日；模块许可分别核查 |
| Light Origins | 增补公司总览、Lightbot 0、LightNav-ER 与 InsightBench | 不推定完整训练数据、整机设计或独立 ER 权重开放 |
| Galbot | WAM 独立详情、WBC 0.5 对应 Humanoid-GPT；补实 GraspVLA | WBC 训练/全量数据待发布；WAM 完整资产未确认 |
| Galaxea | G0 / G0Plus 历史 revision 边界；修复 FastWAM 缺失来源并对齐源码时序 | 历史版本不使用 G0.5 命令推定兼容；模型许可有区别 |
| AgiBot | 补实 Colosseo / GO-1、Genie Envisioner；补全四项阶段路线与 GE-Act 2 | BFM / AGILE / Studio 公开信息有限；后续版本不继承 V1 开放状态 |
| LimX | TRON1 部署页补机制、配置、真机接口；保留 COSA / FluxVLA | 平台可操作不等于每个模型训练资产开放 |
| Unitree | 补遥操作→Isaac Lab 仿真→LeRobot 训练/部署入口 | 持续维护仓不赋造发布日期；模型/数据逐项目核对 |
| Delta Intelligence | 补入 16 家来源总索引，保留已有 Δ₀ 深读 | 归档技术页未列完整模型资产 |
| Robbyant | 补实 Depth / Vision / Video / VA；对齐代码、权重与数据范围 | VA 2.0 技术报告、shared-backbone VA 与全量训练池不同 |
| Reward AI | 补入来源总索引，保留 OM-1 与 DexCap 前序边界 | DexCap 开源不能替代 OM-1 模型发布 |
| Symbiosis Robotics | 补入来源总索引，保留 DPC 独立详情 | 技术页未列模型资产；演示不作统一成功率排名 |

## 验收

- [x] 16 家来源索引与公司路线对齐。
- [x] 公司模型节点不再回链总对照页；共享详情明确版本沿革。
- [x] 本轮薄弱页补充机制、工程读法、局限、参考来源；新页有 wiki 入链。
- [x] 未确认日期保留空值，并提供 `date_note` 和可读说明。
- [x] 回归测试覆盖真实详情 ID、日期合法性、空日期理由、来源存在与策展占位退化。
- [x] GitHub Actions 模式 preflight 与全量测试通过；导出质量 13/13。
- [x] 浏览器逐家核对 16 家、87 节点和详情链接；Envisioner 源码时序图正常渲染，无 console 错误。
- [x] 独立复核并补充 RLT 关键阶段 / 完整任务与人工切换口径。
- [x] 仅提交源文件，创建 [PR #2529](https://github.com/ImChong/Robotics_Notebooks/pull/2529) 交由用户 review，未合并。

## 验证记录

### RAI Institute 公司路线增补（2026-10-08）

- 公司入口扩为 18 家，RAI 按 2022 年成立插入，同步首页、路线 JSON、公司对照与官方来源索引。
- 七个时间轴节点：六个既有项目详情（ZEST、AthenaZero、Sumo、Robot Juggling、SMPC-to-RL、Exploy）与一个新机构总览；总览有 wiki 入链及 Mermaid 研究关系图。
- 日期使用官方公开事件：ZEST arXiv v1 的 01-30、AthenaZero 博客 04-07、Sumo arXiv v1 的 04-09、抛接演示 05-27、SMPC-to-RL arXiv v1 的 08-12、Exploy 博客 10-07；分别说明期刊/后续论文日期。
- Sumo 当前项目页指向 rai-opensource/sumo，补 README 仿真/无界面规划入口及本地来源字段；没有运行其重型仿真或真机栈。RAI 与 Boston Dynamics 以及 RobotecAI/rai 框架分别阅读。
- 定向路线/身份检查 19 项与前端回归 75 项通过；PR 派生文件 guard 通过，导出图谱零孤儿。完整 lint、搜索、测试及 Actions 结果记录在 PR。
- 浏览器截图未完成：当前执行环境拒绝 Chromium 创建进程 Unix socket；不以自动化结构测试代替已完成的视觉验证。

### RoboParty 公司路线增补（2026-10-06）

- 在 main 工作区完成源文件修改后切出 PR 分支，以 main 为目标提交 review。
- 公司名单扩为 17 家；RoboParty 按 2025 年成立插入，八个节点复用现有 wiki：Roboto Origin、Party OS、INTACT、Know-How 阅读地图、hhtools、MimicLite、UFO、TeCH。
- 同步首页入口、对照矩阵与官方来源索引。Hero 数字保留部署统计，由现有 JS 从入口链接计数；调整旧测试的静态数字假设。
- 知识边界：Agent Reach / Jina 与浏览器已核查飞书目录及部分正文，新增章节未确认；微信本次验证码阻断；数据生成待发布、VLA / Agent 规划；MimicLite 当前策略不沿用首批性能数字；INTACT 上游与 RoboParty fork 开放状态分开，并补运行时序图。
- 验收：`GITHUB_ACTIONS=true make ci-preflight` 零阻塞问题、搜索通过、导出 13/13；常规 `make ci-preflight` 仅 20 条既有 freshness 失败，相应文件与 main 无 diff，未修改其复核日期。
- `make ci-test` 在 `/tmp` Python 3.12 隔离环境全通过：486 tests、769 subtests、前端 75 tests；ruff、mypy、依赖审计通过，覆盖率 63.70%。
- Chrome DevTools MCP 验证首页动态 17 家、展开入口、RoboParty 八节点及有效详情 ID、无 console 错误；INTACT 两张 Mermaid 正常渲染。截图本地保存在 `.cursor-artifacts/screenshots/roboparty-company-roadmap.png` 与 `intact-runtime-detail.png`，不提交二进制。
- 独立只读复核通过；修正 INTACT fork 摘要的历史残留，确认飞书归档未夸大为全文深读。

- `make ci-test`：485 tests、769 subtests；前端 73 tests；ruff、mypy、依赖审计通过。
- 常规 `make ci-preflight`：零断链、零缺来源、零孤儿；仅因 6 条既有 freshness 问题未通过。这些页为 `null-space-control`、`hqp`、`tsid`、`crocoddyl`、`capture-point-dcm`、`lip-zmp`，相应 wiki/source 与基线无 diff。本轮不将它们标记为已复核。
- `GITHUB_ACTIONS=true make ci-preflight`：按仓库线上规则跳过历史 freshness，lint / search / export 全通过。
- Chrome DevTools MCP：逐家公司挂载真实页面，87 个节点数量与链接均对齐 JSON；新增详情与 Mermaid 运行图可读。仅生成本地截图，不提交派生文件或二进制。
- 额外 Codex CLI review 被自动审批拒绝（潜在向外部模型服务传输未提交内容），已用当前会话内独立只读复核替代。

### 宇树公司路线时间核对（2026-10-06）

- 在同步后的 main 工作区核对现有八节点，保留节点范围并按可核实事件时间排序；本轮未修改其他公司对象。
- G1 按官网 2024-05-13 产品发布；RL Gym、XR Teleoperate、LeRobot、Sim IsaacLab 分别以 2023-10-11、2024-08-06、2024-10-18、2025-06-24 含实现的官方提交标记代码历史起点，不声称正式首发或首次公开。
- WMA-0 区分 2025-09-15 训练/推理与权重、09-22 部署代码；VLA-0 记录 2026-01-29 代码/权重发布；WLA-1.0 区分 2026-09-11 ER 权重、09-20 动作专家训练代码、09-28 Base 权重与微调代码。
- WLA 保留 2026-09-18 来源快照并更新当前开放状态；补运行时序图及 Dex1 服务协议限制，不推定全部 64 任务和完整训练池可复现。
- 日期证据写入官方仓库及 G1 官网来源，编译到原有详情；不新增重复 wiki、不修改 catalog/log 或部署统计。
- 定向公司路线测试 7 项通过；独立日期/内容、测试与可维护性复核通过。Chrome DevTools MCP 验证八节点顺序、有效详情链接和 WLA 两张 Mermaid，无 console 错误。
- 验证截图：`.cursor-artifacts/screenshots/unitree-company-timeline.png` 与 `unitree-wla-release-runtime.png`，仅本地保存供 PR 展示。
- 常规 `make ci-preflight` 仅 20 条既有 freshness 失败，相应 wiki/source 与 main 无 diff；未修改无关页面复核日期。
- `GITHUB_ACTIONS=true make ci-preflight` 通过：零阻塞 lint、搜索回归通过、导出质量 13/13；派生产物均 gitignore。
- `make ci-test` 全通过：486 tests、769 subtests、前端 75 tests；ruff、mypy 与 Python 依赖审计通过，覆盖率 63.70%。

### Skild AI 公司路线核对（2026-10-07）

- 官方博客、论文 HTML 与项目页交叉核对；成立年份采用公司公告的 2023 年，月份未确认。
- 原 3 节点补为 7 个按公开事件排序的节点：Brain 技术介绍、视觉运动控制、LocoFormer、人视频微调、工业合作部署、S1、自博弈。共享 Brain 能力复用公司详情，不新增重复项目实体。
- LocoFormer 论文署名机构为 Skild AI；区分 2025-09-24 公司博客、09-28 arXiv v1，Light Origins 为后续引用方。
- S1 使用官方列表 2026-08-18；自博弈使用正文 2026-09-23。内部研发回顾不作公开首发，S1-class 足球模型不推定与操作演示同一 checkpoint。
- 开放状态核查官方项目/博客入口；未见模型资产入口，不由 GitHub 组织仓库数推定待发布。同步 WBC 阅读视角、公司对照与日期来源归档。
- 定向测试 18 项、前端回归 75 项通过；远端提交树与本地源文件树 SHA 一致。完整 preflight 和 Actions 结果见 [PR #2571](https://github.com/ImChong/Robotics_Notebooks/pull/2571) 验证记录；未合并。
