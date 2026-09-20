# Wiki 健康报告

## [2026-09-20] lint | health-check | 自动化 wiki 健康检查

共发现 **0** 个问题（另含 **76** 条信息型预警）：

### ⚠️ 孤儿页（无入链）（0 个）
- 无

### ⚠️ 缺少关联页面区块（0 个）
- 无

### ⚠️ 缺少参考来源区块（0 个）
- 无

### 💡 缺少英文缩写速查区块（新建/大幅改写页须补齐；全库 backlog 信息型）（0 个）
- 无

### ❌ 英文缩写速查位置错误（应在「一句话定义」之后、「为什么重要」之前）（0 个）
- 无

### ❌ 断链（内链目标不存在）（0 个）
- 无

### ❌ 禁止的 [[...]] wikilink 写法（请用标准 Markdown）（0 个）
- 无

### ❌ 未闭合的 <https://...> 自动链接（表格/列表行缺少结尾 >）（0 个）
- 无

### ❌ 引用了不存在的 sources/ 文件（0 个）
- 无

### ❌ 同一 frontmatter arxiv ID 出现在多个页面（一篇论文只允许一个 canonical 节点）（0 个）
- 无

### ❌ Sources 孤儿（sources/papers 死链）（0 个）
- 无

### ⚠️ 陈旧页面（sources 比 wiki 新，建议 review）（0 个）
- 无

### ⚠️ 可能过期（updated: 距今 > 180 天）（0 个）
- 无

### ⚠️ 潜在矛盾（跨页面相反定性描述）（0 个）
- 无

### ⚠️ 空壳页面（< 200 字）（0 个）
- 无

### 💡 频繁提及但缺少 wiki 页面的概念（0 个）
- 无

### 💡 多页以加粗/反引号高频引用但缺独立 concepts/methods/formalizations 页（信息型，不阻塞 CI）（15 个）
- Resource（被 32 个页面以加粗/反引号引用，但无独立 concepts/methods/formalizations 页，建议评估新建）
- community（被 21 个页面以加粗/反引号引用，但无独立 concepts/methods/formalizations 页，建议评估新建）
- Hardware（被 21 个页面以加粗/反引号引用，但无独立 concepts/methods/formalizations 页，建议评估新建）
- Course（被 18 个页面以加粗/反引号引用，但无独立 concepts/methods/formalizations 页，建议评估新建）
- standard（被 16 个页面以加粗/反引号引用，但无独立 concepts/methods/formalizations 页，建议评估新建）
- Benchmark（被 14 个页面以加粗/反引号引用，但无独立 concepts/methods/formalizations 页，建议评估新建）
- Courses（被 14 个页面以加粗/反引号引用，但无独立 concepts/methods/formalizations 页，建议评估新建）
- person（被 13 个页面以加粗/反引号引用，但无独立 concepts/methods/formalizations 页，建议评估新建）
- list（被 12 个页面以加粗/反引号引用，但无独立 concepts/methods/formalizations 页，建议评估新建）
- Datasets（被 11 个页面以加粗/反引号引用，但无独立 concepts/methods/formalizations 页，建议评估新建）
- Manipulation（被 11 个页面以加粗/反引号引用，但无独立 concepts/methods/formalizations 页，建议评估新建）
- Benchmarks（被 10 个页面以加粗/反引号引用，但无独立 concepts/methods/formalizations 页，建议评估新建）
- lab（被 10 个页面以加粗/反引号引用，但无独立 concepts/methods/formalizations 页，建议评估新建）
- method（被 10 个页面以加粗/反引号引用，但无独立 concepts/methods/formalizations 页，建议评估新建）
- conference（被 9 个页面以加粗/反引号引用，但无独立 concepts/methods/formalizations 页，建议评估新建）

### ⚠️ Frontmatter 缺少 type 字段（0 个）
- 无

### ⚠️ 知识库活跃度警告（git / log.md）（0 个）
- 无

### ⚠️ 缺少摘要字段（summary/description）（0 个）
- 无

### ⚠️ Query 页面格式不完整（缺 Query 产物/参考来源/关联页面）（0 个）
- 无

### ⚠️ Formalization 页面缺少公式块（0 个）
- 无

### ⚠️ Formalization 公式变量缺少正文物理含义解释（0 个）
- 无

### ⚠️ README checklist 链接版本不一致（0 个）
- 无

### ⚠️ 图谱孤儿节点预警（graph-stats.json）（0 个）
- 无

### ⚠️ Methods 页面缺少 Formalization/Concept 链接（0 个）
- 无

### ⚠️ Methods 页面缺少主要路线区块（0 个）
- 无

### ⚠️ Entities 页面缺少 Methods/Tasks 关联出边（0 个）
- 无

### ❌ 工具实体缺少可派生的所属机构（0 个）
- 无

### 💡 高频引用 methods/ 缺 queries/ 或 comparisons/ 落地（信息型，不阻塞 CI）（0 个）
- 无

### 💡 paper-* 实体 frontmatter 缺 arxiv/venue/code 来源键（信息型，不阻塞 CI）（6 个）
- wiki/entities/paper-erez-simulation-tools-comparison-icra-2015.md
- wiki/entities/paper-gautier-khalil-inertial-parameter-identification-1988.md
- wiki/entities/paper-gevers-identification-information-matrix-2009.md
- wiki/entities/paper-golemo-neural-augmented-robot-simulation.md
- wiki/entities/paper-kadian-sim2real-predictivity.md
- wiki/entities/paper-khosla-robot-dynamics-parameter-identification-1985.md

### 💡 paper-* 实体正文缺「方法/评测/对比」三段式（信息型，不阻塞 CI）（24 个）
- wiki/entities/paper-acosta-validating-simulators-real-world-impacts.md（缺 评测 / 对比）
- wiki/entities/paper-awesome-humanoid-robot-learning.md（缺 评测 / 对比）
- wiki/entities/paper-epopt-robust-policies-model-ensembles.md（缺 评测 / 对比）
- wiki/entities/paper-erez-simulation-tools-comparison-icra-2015.md（缺 评测 / 对比）
- wiki/entities/paper-eysenbach-off-dynamics-rl.md（缺 评测 / 对比）
- wiki/entities/paper-gautier-khalil-inertial-parameter-identification-1988.md（缺 评测 / 对比）
- wiki/entities/paper-gevers-identification-information-matrix-2009.md（缺 评测 / 对比）
- wiki/entities/paper-golemo-neural-augmented-robot-simulation.md（缺 评测 / 对比）
- wiki/entities/paper-kadian-sim2real-predictivity.md（缺 评测 / 对比）
- wiki/entities/paper-khosla-robot-dynamics-parameter-identification-1985.md（缺 评测 / 对比）
- wiki/entities/paper-kovalev-differentiable-simulation-locomotion-sysid.md（缺 评测 / 对比）
- wiki/entities/paper-le-lidec-contact-models-comparative-analysis.md（缺 评测 / 对比）
- wiki/entities/paper-muratore-bayesian-optimization-domain-randomization.md（缺 评测 / 对比）
- wiki/entities/paper-peng-dynamics-randomization-sim2real.md（缺 评测 / 对比）
- wiki/entities/paper-polysim-multi-simulator-humanoid-sim2real.md（缺 评测 / 对比）
- wiki/entities/paper-rapt-sim2real-ood-detection.md（缺 评测 / 对比）
- wiki/entities/paper-rarl-robust-adversarial-rl.md（缺 评测 / 对比）
- wiki/entities/paper-schwarke-differentiable-simulation-locomotion-corl.md（缺 评测 / 对比）
- wiki/entities/paper-skyfall-gs.md（缺 对比）
- wiki/entities/paper-smith-legged-robots-keep-learning.md（缺 评测 / 对比）
- wiki/entities/paper-splatsim-gaussian-splatting-sim2real.md（缺 评测 / 对比）
- wiki/entities/paper-tan-quadruped-agile-locomotion-sim2real.md（缺 评测 / 对比）
- wiki/entities/paper-up-osi-universal-policy-online-sysid.md（缺 评测 / 对比）
- wiki/entities/paper-vidu-s2.md（缺 对比）

### 💡 paper-* 实体缺「结论」章节（信息型；后续 ingest 必做）（0 个）
- 无

### 💡 dataset 实体正文缺「规模/模态/许可证/重定向就绪度」速查维度（信息型，不阻塞 CI）（9 个）
- wiki/entities/painode-083-argoverse2.md（缺 重定向就绪度）
- wiki/entities/painode-085-bridgedatav2.md（缺 重定向就绪度）
- wiki/entities/painode-089-epickitchens100.md（缺 重定向就绪度）
- wiki/entities/painode-090-nuscenes.md（缺 重定向就绪度）
- wiki/entities/painode-092-rh20t.md（缺 重定向就绪度）
- wiki/entities/painode-093-rlds.md（缺 重定向就绪度）
- wiki/entities/painode-094-robomind.md（缺 重定向就绪度）
- wiki/entities/painode-096-somethingsomethingv2.md（缺 重定向就绪度）
- wiki/entities/painode-097-waymoopendataset.md（缺 重定向就绪度）

### 💡 陈旧声明（含绝对化措辞但同主题有更晚更新页，建议复核；信息型，不阻塞 CI）（0 个）
- 无

### 💡 动力学/仿真/物理概念页缺回链「仿真物理保真度」知识链枢纽（信息型，不阻塞 CI）（1 个）
- wiki/concepts/robot-simulation-three-layers.md

### 💡 接触/力控/操作概念页缺回链「接触力旋量闭环」知识链枢纽（信息型，不阻塞 CI）（0 个）
- 无

### 💡 VLM/VLN/VLA/VLX/World-Model 家族概念/对比页缺回链「具身大模型分类学选型闭环」知识链枢纽（信息型，不阻塞 CI）（2 个）
- wiki/concepts/humanoid-cognitive-reliability-gap.md
- wiki/concepts/retrieval-augmented-generation.md

### 💡 benchmark/evaluation 实体/对比/概念页缺回链「具身大模型评测基准选型闭环」知识链枢纽（信息型，不阻塞 CI）（14 个）
- wiki/entities/painode-009-alfred.md
- wiki/entities/painode-010-arnold.md
- wiki/entities/painode-011-carlaleaderboard.md
- wiki/entities/painode-012-colosseum.md
- wiki/entities/painode-013-furniturebench.md
- wiki/entities/painode-018-minedojo.md
- wiki/entities/painode-019-openeqa.md
- wiki/entities/painode-021-robothor.md
- wiki/entities/painode-023-teach.md
- wiki/entities/painode-024-vlabench.md
- wiki/entities/paper-bench2dex.md
- wiki/entities/paper-handedit.md
- wiki/entities/paper-kadian-sim2real-predictivity.md
- wiki/entities/paper-robovad.md

### 💡 actuator/eda/foc 实体/对比/概念页缺回链「执行器驱动链选型闭环」知识链枢纽（信息型，不阻塞 CI）（1 个）
- wiki/entities/paper-mechanical-intelligence-info-theory.md

### 💡 detection/segmentation/perception/semantic-mapping 实体/对比/概念/方法页缺回链「机器人视觉感知栈选型闭环」知识链枢纽（信息型，不阻塞 CI）（4 个）
- wiki/entities/paper-leap-quadruped-active-perception.md
- wiki/entities/paper-pose-semantic-legged-exploration.md
- wiki/entities/paper-rapt-sim2real-ood-detection.md
- wiki/entities/paper-robovad.md

📊 Sources 覆盖率：4607/4607 (100%) wiki/entity 页有 ingest 来源
