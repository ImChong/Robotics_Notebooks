# Rhoda AI：官网研究页、新闻稿与团队页核查

- **类型：** 官方站点索引（研究页 + 新闻稿 + 团队页）
- **名称：** Rhoda AI（官网自称 Rhoda；暂无官方中文名）
- **官方研究入口：** <https://www.rhoda.ai/research>
- **新闻稿：** <https://www.rhoda.ai/news/press-release>（2026-03-10）
- **团队页：** <https://www.rhoda.ai/team>
- **核查日期：** 2026-10-16
- **方法：** `curl` 读取上述页面的服务端渲染 HTML，去标签后核对标题、日期、数字与外链；文中数字均对照原文，不取自二手转述。
- **一句话说明：** 以「互联网规模视频预训练 + 闭环视频预测控制」做机器人基础模型的商业公司；官网公开 3 篇研究博文，均无 arXiv、代码、权重或数据集入口。

## 官网研究页（2026-10-16 共 3 篇）

列表页只显示月日，不显示年份；年份依据下列口径判定。

| 官网日期 | 标题 | 官网入口 | 年份依据 | 本库节点 |
| --- | --- | --- | --- | --- |
| Oct 13 | Giving Robots a Sense of Touch | `/research/giving-robots-a-sense-of-touch` | 文末引用块 `rhoda2026tactile` 与页脚 © 2026；新闻页未列此篇 | [触觉 + DVA](../../wiki/entities/rhoda-dva-tactile-sensing.md) |
| Sep 10 | Does Scaling Web-Video Pre-training Help Real Robots Do Real Work? | `/research/scaling-web-video-pretraining` | 新闻页标注 2026-09-10 | [网络视频预训练 scaling](../../wiki/entities/rhoda-scaling-web-video-pretraining.md) |
| Mar 10 | Causal Video Models Are Data-Efficient Robot Policy Learners | `/research/direct-video-action` | 新闻页标注 2026-03-10，与新闻稿同日 | [Direct Video-Action（DVA）](../../wiki/entities/rhoda-direct-video-action.md) |

## 三篇要点（均为公司自报，无第三方复现）

- **DVA（2026-03-10）：** 把机器人策略重述为视频生成。因果视频模型预测短时未来画面，独立的逆动力学模型把预测画面转成末端动作，闭环运行；强调原生因果预训练、Context Amortization、KV-cache 复用与 Leapfrog Inference。自述「总计约 10 小时机器人数据」即可可靠完成长程任务，实际任务为 10–20 小时。案例：Decanting（拆箱倒轴承并分拣包装，11 小时数据，单次未剪辑运行 1.5 小时）；Container Breakdown（拆解约 50 磅工业周转箱，17 小时数据，连续 160 分钟无干预）；长上下文记忆（仅保留最近 8 帧时会重复已完成步骤）；一次示范模仿。**无成功率百分比，无量化基线对比，无参数量**。
- **网络视频预训练 scaling（2026-09-10）：** 固定逆动力学模型、任务（14 步、6 个子任务、每箱 10 kg 的拆包分拣）与后训练流程，只改视频模型规模与预训练算力。「at-speed completion rate」要求 100 秒内无人工干预完成。规模 XS → S → M → L：3.7% → 65.0% → 75.3% → 84.7%（L 仍未饱和）；M 模型算力 0.08× / 0.18× / 0.37× / 1×：57.8% / 67.3% / 74.5% / 75.3%；自定义的 DINO FD（预训练检查点在留出网络视频上的 Fréchet 距离）与任务表现在 7 个检查点上同序。全文累计超过 200 小时真机评测，每策略约 100 到数百次试验。作者自列局限：单任务、单本体、每条件仅一次后训练、规模实验未做算力匹配、结论仅为相关性。
- **触觉 + DVA（2026-10-13）：** 新触觉传感器，薄型（页面图示高度 8 mm），可测法向力与剪切力，软硬件协同标定并抑制漂移 / 迟滞；羽毛轻触响应仅几毫牛，量程到 40 N 以上（哑铃），接触振动采样约 500 Hz；同条件下所测现成光学与磁性传感器无可辨别响应。多模态 DVA 以历史触觉视频预测未来触觉视频，再由逆动力学模型转动作。盲操作任务（遮挡腕部与头部相机）：训练 1 个光滑球 + 1 个纹理球，测试 4 个光滑球 + 3 个纹理球（其中 2 个在直径、硬度、纹理上不同于训练球），作者称仍全部取回纹理球。**无成功率、无试验次数、无传感器价格与空间分辨率**。贡献者 Andrew Xie（实习期间）、Leo Dong、David B. Lindell。

## 公司信息（新闻稿与官网）

- **公开亮相：** 2026-03-10 新闻稿，发稿地 Palo Alto, CA；原文「after 18 months in stealth」。**官网与新闻稿均未写成立日期**，把 2026-03 往前推约 18 个月只能得到 2024 年下半年的**推测**；网络搜索摘要里有第三方称 2024 年成立（未读原文，不作一手依据）。
- **产品叙事：** FutureVision 定位为公司「智能层」基础模型，预期未来授权给不同软硬件平台；底层架构称 Direct Video Action（DVA），先在数亿条互联网视频上预训练，再用少量机器人数据后训练。
- **融资（公司公告）：** 4.5 亿美元 Series A；投资方含 Capricorn Investment Group、Khosla Ventures、Leitmotif、Matter Venture Partners、Mayfield、Premji Invest、Prelude Ventures、Temasek、Xora 与 John Doerr 等个人。新闻页列出的 Bloomberg 标题称估值 17 亿美元，属媒体口径，不是公司公告，正文本次未读。
- **领导层：** 新闻稿——联合创始人兼 CEO Jagdeep Singh；首席科学官 Eric Ryan Chan（曾任 WorldLabs 生成模型架构师）；斯坦福教授 Gordon Wetzstein 列为领导层。团队页——Jagdeep Singh（CEO、联创）、Eric Chan（Chief Scientist、联创）、Gordon Wetzstein（Scientific Advisor、联创）、Andrew Wooten（CPO、联创）、Changan Chen（CRO、联创）、Steve Tirado、Alex Bergman。
- **工业叙事（自报）：** 新闻稿称某高产量制造评估中，一个零部件处理流程每轮不到两分钟、无人干预，超过客户 KPI；首页提及汽车、制造、物流、电商，未点名新闻稿中的客户。首页另述机器人平台使用定制执行器，额定负载 25 kg、峰值 40 kg。

## 开放范围核查（2026-10-16）

- 3 篇研究页、新闻稿、团队页的 HTML 中**没有** Rhoda 自有的 GitHub / Hugging Face / ModelScope / Zenodo 链接，也**没有**本工作的 arXiv 编号；页面里的 arXiv 与外链都是引用他人工作（DVA 页含 DreamZero 项目页等）。
- 模型参数量、训练数据规模、传感器价格与规格、客户名称均**未公开**。
- 结论：**确认未开源**（截至入库日官网未列代码、权重或数据）；所有性能数字为公司博客自报，不可独立复现。

## 对 wiki 的映射

- 公司入口：[Rhoda AI](../../wiki/entities/rhoda-ai.md)
- 三篇研究节点：[DVA](../../wiki/entities/rhoda-direct-video-action.md)、[网络视频预训练 scaling](../../wiki/entities/rhoda-scaling-web-video-pretraining.md)、[触觉 + DVA](../../wiki/entities/rhoda-dva-tactile-sensing.md)
- 公司路线对照：[机器人基础模型与通用人形：公司技术路线对照（2026）](../../wiki/comparisons/robot-foundation-model-company-paths-2026.md)
- 展示数据：`docs/company-roadmaps.json`；首页入口：`docs/index.html`
