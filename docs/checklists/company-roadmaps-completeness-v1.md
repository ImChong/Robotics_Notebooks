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
- [ ] 仅提交源文件，创建 PR 交由用户 review。

## 验证记录

- `make ci-test`：485 tests、769 subtests；前端 73 tests；ruff、mypy、依赖审计通过。
- 常规 `make ci-preflight`：零断链、零缺来源、零孤儿；仅因 6 条既有 freshness 问题未通过。这些页为 `null-space-control`、`hqp`、`tsid`、`crocoddyl`、`capture-point-dcm`、`lip-zmp`，相应 wiki/source 与基线无 diff。本轮不将它们标记为已复核。
- `GITHUB_ACTIONS=true make ci-preflight`：按仓库线上规则跳过历史 freshness，lint / search / export 全通过。
- Chrome DevTools MCP：逐家公司挂载真实页面，87 个节点数量与链接均对齐 JSON；新增详情与 Mermaid 运行图可读。仅生成本地截图，不提交派生文件或二进制。
- 额外 Codex CLI review 被自动审批拒绝（潜在向外部模型服务传输未提交内容），已用当前会话内独立只读复核替代。
