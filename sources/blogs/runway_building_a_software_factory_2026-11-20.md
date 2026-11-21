# Building a Software Factory — Runway 软件工厂工程实践

> 来源归档；仅编译观点和数据，不转载文章全文。

- **类型：** blog（Runway 官方 Engineering 博客）
- **作者：** Alejandro Alvarez（Technical Staff）
- **发布日期：** 2026-11-20
- **原文：** https://runway.com/news/engineering/building-a-software-factory
- **入库日期：** 2026-11-21
- **核对范围：** 完整阅读 Every Step Needed an Engineer、Let Agents Get to Green / Review / Approve / Run QA / Watch Main / Run the Factory、Where Each Station Stands Today、The Engineer's Role 和结尾；对照正文统计、图注的口径及阶段边界。
- **通读：** 已完成
- **开源状态：** 工程经验文章，未提供该内部软件工厂各组件的公开可运行仓库；不推断其内部实现可复现。
- **沉淀到 wiki：** [Agent-first 软件工厂与 Harness Engineering](../../wiki/concepts/agentic-software-factory-harness.md)

## 核心问题与机制

写代码的 agent 增多后，PR 在 CI、评审、审批、QA、合并后故障和调度上排队。Runway 依次把「写变更并跑绿 CI → 带上下文的 AI Review → 按风险审批 → 浏览器驱动 QA → 告警及回滚 → 协调器派单/排序合并」交给 agent，不直接从自动写代码跃迁至无约束自动部署。

- **Get to Green：** 内部云服务追踪 PR 至 CI 绿灯，自动修测试、处理评审意见、rebase；11 月上半月抽样 100 个 agent 驱动合并 PR，78 个没有人类提交。
- **Review：** reviewer 读取计划、diff 和周边代码，向 coding agent 退回问题；核心 API 仓库 11 月上半月 93% AI review 评论线程被处理或导致改码（9 月为 84%）。
- **Approve：** 独立 agent 评级低/中/高风险，不确定时向上取整；低风险可由 AI 独立审批，中风险取决于仓库上限；高风险及数据库迁移、密钥、认证、CI/部署配置等敏感路径保留人工/owner 审批。9–10 月 977 次 AI 批准中 15 次被人反对（约 1.5%）。
- **QA：** agent 在预览环境用浏览器扮演用户，定位真实故障、生成修复任务并重跑场景。首次四天发现 86 个确认故障；图注明确该四天内修复 74 个、其中 67 个已重跑验证，避免把累计“后续都修好”误当四天结果。
- **Watch Main：** 告警触发重试、诊断、已知修复和升级；协调器在主干不绿时停止自身合并，发布后由健康检查触发自动回滚。前两周约 140 个告警，除 1 个之外均未需升级叫人。
- **Coordinator：** issue tracker / 告警 → 分派任务 → 按闸门排序合并；先 shadow mode，再人工检查动作，正式上线前两周执行 14,000+ agent 任务、产生约 1,900 PR。

## 数字读法

- 2026 年 10 月，云 agent 驱动的合并 PR 占 38%，6 月为 0.1%；11 月截至文中日期为约 30%，作者注明统计只包含可归因于 agent 的 PR。
- 核心 API 仓库首次批准等待时间从 9 月约 2.3 小时降至 11 月中旬约 5 分钟；六个可比仓库的每周合并平均值由 9 月约 929 个变为 10 月中旬后约 2235 个（约 2.4 倍）。
- 10–11 月 AI-only approval 的回滚率约 0.10%，人工批准约 0.5%；作者强调两组 PR 风险构成不同，不宜直接因果比较。

## 失效保护与局限

Runway 列出的实际护栏包含只读影子运行、**合并时重查确切 commit 的门禁**、幂等外部动作、按规则升级人工和一次调用回滚。工程师仍负责明确产品目标、建筑决策、风险规则、评估指标及解除 agent 卡点。上述数字是公司内部经验，不是跨团队对照实验；不等于可以对机器人真机执行取消审批或弱化硬件安全联锁。

## 官方来源

- [Runway, Building a Software Factory（2026-11-20）](https://runway.com/news/engineering/building-a-software-factory)
