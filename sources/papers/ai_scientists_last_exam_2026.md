# AI Scientists' Last Exam: Nature as the Last Verifier Towards Closed-Loop Discovery

> 一手论文来源归档；本条以作者团队官方 PDF 链接为主，公开代码和文档用于有限交叉核查。

- **类型：** paper（尚未公开正式会议/期刊版本）
- **标题：** AI Scientists' Last Exam: Nature as the Last Verifier Towards Closed-Loop Discovery
- **作者：** Gan Zhang、Hubert He、Xu Huang 等；README 标注前三位为共同一作，完整名单以 CITATION.cff 为准
- **版本：** 官方仓库 main 分支所指向的 manuscript PDF；README 将其标注为 unpublished manuscript; work in progress（2026）
- **论文 PDF：** <https://github.com/MasterAI-EAM/ai-scientists-last-exam/blob/main/paper/AI_Scientists_Last_Exam.pdf>
- **项目页：** <https://aisle.science/>
- **源码仓：** <https://github.com/MasterAI-EAM/ai-scientists-last-exam>
- **入库日期：** 2026-11-09
- **核对范围：** 已读取官方 README、任务说明及 verifier 入口、wetlab/README.md 和 wetlab/VALIDATION.md；PDF 大小约 31.5 MB，当前读取环境无法取得可逐页解析的原文件，论文全文（方法、实验、局限、附录）尚未核对。
- **通读：** 未完成

## 已从其他一手资料确认的事实（不能替代论文全文）

1. AISLE 面向科学智能体实验决策：提出实验、读反馈，在实验预算和任务约束下达成可验证目标；项目基于 Harbor 对 agent–model 配对运行评测。
2. 公开任务环境含模拟、计算及已有实验数据查询；独立 wetlab 子框架提供实验票据提交、等待、恢复、封存和离线验证，但并不意味着所有 benchmark 任务已经接入真实实验室。
3. 色彩匹配示例任务每次配方共 200 μL、最多 12 次提交；观测来自冻结 LightGBM 预测模型，非现场湿实验测量。达到 CIEDE2000 差异 ΔE00 ≤ 1 为该样例的成功条件。
4. wetlab/VALIDATION.md 区分了 Docker/模拟网关测试、真实模型调用测试、以及实际物理实验；没有提供能据此宣称完整自主湿实验能力已验证的证据。

## 待核论文内容

- 论文对完整 benchmark 的正式任务分类、样本规模、统计方法和评分聚合公式。
- 论文表格/图中的模型结果及置信区间、消融、实验成本和作者局限。
- 论文附录、表注与正文用语核对。以上暂不以网站实时排行榜替代。

## 对 wiki 的映射

- [AISLE / AI Scientists' Last Exam 论文与项目唯一实体页](../../wiki/entities/paper-ai-scientists-last-exam.md)
- [AISLE 官方站点归档](../sites/aisle-science.md)
- [AISLE 官方仓库归档](../repos/ai-scientists-last-exam.md)
