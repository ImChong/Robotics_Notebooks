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

### NVIDIA Isaac Gym 时间点补充（2026-10-08）

- 新增 **Isaac Gym Preview / 2020-10**，位于 Isaac Lab 之前，复用 `entity-isaac-gym`。
- 日期依据为官方 Preview 论坛 2020-10-31 开放公告；后续 12 月介绍博客与 2021 年论文日期分开记录。
- 路线卡片展示日期口径与依据链接，详情页同步来源与版本边界；验证结果见补充 PR。

### 22 个未注明日期节点核查（2026-10-08）

- 对当前 19 家公司路线的 22 个空日期节点逐一核查：18 个补齐可确认的事件月份，4 个保留空值，并将有日期节点按月重新排序。
- 来源采用官方 News / Release、arXiv v1 与明确含实现的提交；TRON1 不沿用早期 PointFoot 仓库日期，G0 / G0Plus 分开版本，UFO / TeCH 区分命名事件，MimicLite 与 hhtools 明确当前版本。
- RAI 采用机构成立事件；Lightbot 0 采用官方跑酷演示，均在标题与日期口径中说明。日期、代码历史、产品首发与商业交付不混用。
- 保留空日期：AstraBrain-WAM、Light Origins 公司与全栈、LingBot 全栈、Know-How 文档；公司与文档持续维护入口不借用邻近模型或第三方文章日期。
- 新增[日期证据汇编](../../sources/sites/company-roadmap-date-audit-2026-10-08.md)，复用 18 个既有详情节点；路线卡片展示日期口径并提供官方日期依据链接。
- 定向公司路线测试 7 项、趋势公司行为测试 3 项通过；19 家真实数据渲染、18 个证据链接与 4 个空日期说明一致，HTML 转义检查通过。全量测试与 preflight 结果在 PR 验证记录中列明。

### 星动纪元 PR #2583 冲突同步（2026-10-08）

- 合入 main 已发布的 RAI Institute 路线，保留两边机构、来源、矩阵与开放范围说明；当前路线共 19 家，首页与索引计数一致。
- 星动纪元七节点保持不变；RAI 与其他 main 公司对象不改写，成立年份排序与首页顺序保持一致。
- 公司路线与项目身份定向测试 19 项、前端回归 75 项通过；完整 preflight 与远端 CI 结果见 PR。

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

### 星动纪元公司路线增补（2026-10-08）

- 公司扩为 18 家，按 2023 年成立插入；成立月份以 WRC 展商自述的 2023-08 为准，同步首页入口、对照矩阵与官方来源索引。
- 七节点：Humanoid-Gym、VPP、ERA-42、L7、teleop_client、xbot_sdk_api、robotera_vla。复用原有研究/工程实体，ERA-42 与 L7 落在公司详情，不重复建项目节点。
- 论文日期：2024-04-08 / 2024-12-19 v1；产品日期：ERA-42 2024-12-23、L7 2025-07-22。工程接口首发未确认，空日期附说明，不填入库日期。
- 深化 VPP 与三项工程详情；区分 VPP 联合研究、ERA-42 产品、M7 π₀.₅ 示例，保留项目页和 arXiv CALVIN 指标差异；SDK 与遥操作依赖厂商服务/授权环境。
- 验收：Python 全量 500 项、前端 75 项通过；`GITHUB_ACTIONS=true make ci-preflight` 按线上规则跳过 freshness，lint 零阻塞、搜索回归和导出 13/13 通过。初次浅克隆导致活动数据 guard 失败，补齐完整 git 历史后重跑通过。
- Chromium 下载为损坏压缩包，未生成浏览器截图；未重跑模型训练或真机实验。仅提交源文件与日志碎片，PR Actions 结果见 PR。

### 遗漏节点全面核查（2026-10-11）

- 范围：当前 29 家公司的路线时间线，对照仓库 wiki / sources 中的官方来源与官网博客、仓库、arXiv 列表；5 路只读核查共提出 99 项候选，收入 56 项（总节点 189 → 245）。
- 收录标准：公司自己的机器人技术产出（模型、论文、基准、仿真、整机开源、官方工程仓），页面通过 `test_company_roadmaps.py` 约束，日期可单独确认；论文用 arXiv v1 日期（入选项已用 arXiv API 复核），代码仓取默认分支根提交并在 `date_note` 标明不证明首发，无日期留空。
- 新增：NVIDIA 20、Galbot 8、Google DeepMind 8、AgiBot 5、Robotera 3、Galaxea 3、Unitree 2、RAI 2、Limx 2、XPENG 1、Xiaomi 1、1X 1、Figure 1、Physical Intelligence 1、X Square 1、Lightwheel 1、Simate 1。
- 未收入：智驾 / 通用视觉论文（XPENG 6 项）、联合署名或第三方牵头（GroundingPI、τ₀-VLA、EffVLA、FastStair、HEFT、FoldNet++、PASSAGE、Open X-Embodiment）、合作新闻稿与融资类（Lightwheel 合作稿、PeritasAI）、低置信或日期靠推断（LeIsaac、AimRT、CTS、RT-1 / DreamerV3 / dm_control、各 SDK 仓）、`sources/repos/unitree.md` 已限定范围的 SDK / 仿真桥仓（unitree_sdk2 / mujoco / ros2）。
- 无 wiki 页面的官方发布不新建页面，留作后续入库：Figure 03、1X NEO 发布、LimX Oli、XPENG Fe0 / Si0 / Capek 0.5、Delta D1 等。
- 待核对：AgiBot `GO-2` 节点日期 2026-01 取自 arXiv:2601.11404，而 wiki 页写 2026-06 发布，日期口径未统一，本次未改。
