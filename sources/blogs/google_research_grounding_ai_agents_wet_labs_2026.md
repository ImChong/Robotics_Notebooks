# Grounding AI agents in biological reality: Building blocks for autonomous wet labs（Google Research 博客）

> 来源归档（blog / Google Research 官方发布文，综述三篇配套论文）

- **标题：** Grounding AI agents in biological reality: Building blocks for autonomous wet labs
- **类型：** blog（官方一手发布文，同时是该项目的「总览」；三篇论文各有独立节点，本归档不替代它们）
- **作者 / 机构：** Pooja Rao（Research Scientist）、Petar Sirkovic（Software Engineer）、Annalisa Pawlosky（Research Scientist），Google（Google Research）；致谢页写明研究由 Annalisa Pawlosky 牵头、与 Pooja Rao、Petar Sirkovic 共同完成
- **原始链接：** <https://research.google/blog/grounding-ai-agents-in-biological-reality-building-blocks-for-autonomous-wet-labs/>
- **发布日期：** 页面署名 2026-10-16（页面抓取日 2026-11-07，页内「其他文章」最晚为 2026-11-05）
- **标签（页面 Labels）：** General Science、Health & Bioscience、Robotics
- **入库日期：** 2026-11-07
- **抓取方式：** `curl` 取页面 HTML（约 190 KB），用 `python3 -I` 去掉脚本 / 样式 / 导航 / 页脚后通读正文，并保留全部超链接；页面里的图像在 HTML 中只有 alt 文字与行内图注，图像本身（图表像素）未逐张目视
- **一句话说明：** Google Research 把三份内部预印本 PDF（VERID 语义数字孪生、适配体沙盒、表位切除蛋白结合剂设计）串成「把 AI agent 接到湿实验台」的路线叙述，称它们是「人类主导的自主发现」的构件；博客自己没有代码或数据发布。

## 核对范围与通读

- **核对范围：** 博客正文全文（含引言、三个小节、Safety 小节、结语、Acknowledgements、三张图的行内图注与 alt、页首「Quick links」与全部正文超链接）；页面导航、站点页脚与「其他文章」推荐只扫了标题。对照口径时另读了三篇 PDF 的**摘要与引言**，并就博客里出现的具体数字与论断，在各 PDF 的结果段 / 图注 / 方法段做了针对性检索；三篇论文**没有通读**（各自有独立节点负责全文）。
- **通读：** 已完成

## 链接核查（2026-11-07，HTTP 状态码）

| 博客中的链接 | 状态 | 开放情况 |
|---|---|---|
| VERID paper `https://www.gstatic.com/science-ai/verid_semantic_lab.pdf` | 200，16 页 | 可公开下载；PDF 标题为 *Semantic Representation for Agentic Self-Play in Automated Laboratories*（博客只叫 VERID paper / report）；版权行 CC BY 4.0；文件头日期 2026-10-22 |
| Agentic aptamer sandbox paper `.../agentic_aptamer_sandbox.pdf` | 200，27 页 | 可公开下载；标题 *Aptamers as a sandbox for lab-in-the-loop agentic molecular discovery*；文件头日期 2026-10-12；版权行 © 2026 Google |
| Epitope excision paper `.../epitope_excision.pdf` | 200，18 页 | 可公开下载；标题 *Agent-guided epitope excision enables economical de novo binder design*；文件头日期 2026-10-23；版权行 © 2026 Google |
| PyLabRobot 文档 `https://docs.pylabrobot.org/` | 200 | 开源框架；`git ls-remote https://github.com/PyLabRobot/pylabrobot` 可达，仓库 LICENSE 为 MIT（博客本身只链了文档站） |
| RCSB 3RGK（myoglobin）、1PPB（α-thrombin）、1Z68（FAP） | 均 200 | 公共结构库 |
| DeepMind Frontier Safety Framework、bioresilience 两篇 | 均 200（前者重定向到 `deepmind.google/blog/updating-the-frontier-safety-framework/`） | 公开网页 |
| Wikipedia：In silico、Digital twin、Aptamer、G-quadruplex、Cell sorting | 均 200 | 公开 |

- 博客**没有**链接任何 GitHub 仓库、数据集或模型权重；三篇 PDF 的全文文本中也检索不到 github / 代码或数据可得性声明（对 VERID 与表位切除两篇检索了 `github`、`availab`、`code`，未命中任何发布声明；适配体一篇的 `availab` 命中仅是方法 / 补充表述）。结论：截至 2026-11-07，**论文可读，但 VERID、适配体流水线、表位切除流水线的代码与数据均未见公开入口**（宣称未见，不是「确认不会开放」）。

## 核心摘录（归纳，非全文；数字均为博客作者表述）

### 动机与两大挑战

- 一个 100 个氨基酸的蛋白至少有 20^100 种序列（页面里渲染成「20100」，是上标丢失）。
- 愿景：AI agent 不只构思与分析，还要「做湿实验检验假说、从结果学习、自我改进」，且「一切由人类科学能动性驱动」。
- 挑战一：前沿模型对三维几何与物理状态推理不可靠，而液体处理机器人要搬试剂、记体积、执行分子生物学流程。
- 挑战二：agent 的假说生成与 in silico 设计进展很快，但需要反复真实反馈来验证，湿实验没跟上，读出成为更大的瓶颈。

### 「构件」（building blocks）

博客**没有**列一张正式的「构件清单」。题目与结语里的「building blocks」指三类工作，结语原话：从「把 agent 接地到物理实验室机器人」，到「跑快速闭环适配体循环、让 agent 学会探索序列空间」，再到「发现个位数纳摩尔蛋白结合剂」，「这些工作构成人类指导的自主发现的构件」。引言列出了两条要点（VERID；适配体与 de novo 蛋白设计的闭环原型），而句首写的是「三个研究项目」，也就是正文三小节。

### 小节一：VERID（Grounding agents in physical reality）

- VERID = VERified Infrastructure for Discovery Laboratories，描述为实验台工作站的**语义数字孪生**：有状态的 3D 体积实体树，跟踪从机械臂几何到单孔体积、试剂身份、枪头库存。
- 基于开源 PyLabRobot；四项能力：实时 3D 状态跟踪、快速虚拟 dry run（称「几秒」）、硬件安全检查（挡掉无效指令）、同步视频与动作日志。
- 指令是否安全取决于实时物理上下文而不只是代码；三类问题：物理上不可能（8 通道移液头放进太短的窄槽）、逻辑无效（从耗尽的孔吸液）、科学上被未建模流体效应破坏。
- 案例（博客转述）：同步视频与执行轨迹发现靠近孔底分液造成染料分层，agent 改进混合策略；持久孔态让 agent 在板读数仪故障后无需重新移液即可恢复；模拟器拒绝 8 通道进窄槽，agent 即时重规划，并设计出针对枪头壁残留的析因诊断；闭环自博弈中 agent 从顶部相机图像重建盲配染料混合物，「识别自己的光学与流体伪影，使准确度提升超过五倍」。
- 可复现执行与完整轨迹日志（停留时间、蒸发、板布局等）为「将来 agent 区分生物信号与实验噪声」打基础。

### 小节二：适配体沙盒（Learning to learn with aptamers）

- 完整一轮设计—测试—分析循环「只要 7–10 天」，因为 DNA 可成池化学合成并直接测序读出；每轮 agent 设计候选序列 → 在磁珠上对靶蛋白筛选 → 测序，返回定量数据。
- 3 轮、2 个靶点（myoglobin、α-thrombin）；确认的靶上命中率升至 myoglobin 33.5%、α-thrombin 42.1%。
- α-thrombin：第 1 轮 0%，博客称因固定引物突出端使 48% 序列折叠被破坏；第 2 轮加夹持茎（clamped flanking stems）到 26.0%，第 3 轮多样化到 42.1%，特异性 98.5%。
- myoglobin：第 2 轮 agent 偏重原始结合力，G-四链体（G4）motif 占设计的 63%，pull-down 上升，但多数是对脱靶蛋白与裸磁珠的「粘性」结合，特异性 75.0% → 19.1%，珠校正命中率 10.2% → 3.7%；agent 在第 3 轮自主改规则（限制混杂 motif、扣除珠结合物、转向非 G4 骨架），特异性回到 72.4%，靶上命中率「九倍」增至 33.5%。
- 另称论文还考察了 agent 如何像科学家推理、以及有时「非典型」地探索结构生物学家不会去的序列空间。

### 小节三：表位切除（Accelerating de novo protein binder design）

- 蛋白需生物表达，每一圈多出细胞 / 无细胞筛选、表达、折叠步骤，序列空间更大，计算与湿实验两侧都有瓶颈。
- 靶点：成纤维细胞活化蛋白 α（FAP），一种在实体瘤中过表达的大受体。「仅几次实验运行」，最好的设计得到个位数纳摩尔亲和力结合剂（博客称这通常要大量实验室优化才能达到）。
- 计算侧：重折叠（re-folding）消耗绝大部分 GPU 算力、随靶点长度二次增长；只切出局部结合口袋来折叠，「每个循环加速超过八倍且不损结构精度」；图注写「结构预测加速 8.6x」。
- 同等算力下，科学家定策略 + agent 做战术执行的人机混合，使 in silico 命中率「近乎翻倍」，并覆盖更宽的靶位点范围。
- 湿实验侧：酵母表面展示 + 荧光激活细胞分选（FACS）并行筛选大型文库。
- 下一步：把 VERID 的全程可追溯自动化与蛋白发现的闭环学习合并，让 agent 直接在湿实验里探索「崎岖」的生物搜索空间。

### Safety 小节

- 依据 Google 的 Frontier Safety Framework 与 bioresilience 做法，计算侧与湿实验侧都设保障：
  - 安全试验床与筛查：选用受控、非感染、无功能增益风险的生物系统；每条设计序列都经过生物安全筛查。
  - 先模拟再执行：通过 VERID，agent 从不把未验证代码直接发给液体处理机器人；每个协议先 dry run，再对数字孪生验证。
  - 人类监督：人类授权并审阅设计序列、核验湿实验协议、监督机器人运行。
- 开放问题：确保 agent 优化的是真结合而非实验捷径、验证多步湿实验协议、持续监测 agent 的推理与执行轨迹。

### 三张图的图注（页内原文要点）

1. 图（表位切除）：表位切除使结构预测加速 8.6x（左）；人机混合流水线使 in silico 命中率近乎翻倍（右上）；在湿实验发现个位数纳摩尔 FAP 结合剂（右下）。alt：「结构模型、命中率与结合验证曲线」。
2. 图（适配体）：agent 在闭环中迭代设计、合成、选择、测序适配体；alt：多轮闭环 AI 流水线示意。
3. 图（适配体结果）：三轮中 agent 从湿实验读出学习，恢复靶标特异性并提升 myoglobin（左）与 α-thrombin（右）的确认 de novo 命中率；alt：两张折线图。
- 另有一段 VERID 的循环视频（页面里的 looping video，无文字图注，仅写「通过 VERID，AI agent 在液体处理机器人上规划、模拟并执行液体处理协议，结合同步多相机画面与实时状态跟踪，跨循环改进实验」）。

## 与三篇论文的口径对照（只读摘要 / 引言 / 针对性段落）

| 项 | 博客 | 论文侧 | 判断 |
|---|---|---|---|
| VERID 论文名 | 「VERID paper / report」 | 标题为 *Semantic Representation for Agentic Self-Play in Automated Laboratories*，作者含 Delley、Pawlosky、Sirkovic、Rao 等（Google Research / Cloud / DeepMind） | 博客不写题名，容易与库里按题名检索脱节 |
| VERID 能力命名 | 实时 3D 状态跟踪 / 虚拟 dry run / 硬件安全检查 / 同步视频与日志 | 四项服务：queryable introspection、time-compressed simulation（Simulation Gate）、trace persistence、safe execution | 实质对应；名称不同 |
| dry run「几秒」 | 「Simulate protocols in seconds」 | 摘要称 time-compressed；全文文本检索未找到「几秒」的定量数字 | 博客的时长表述未在 PDF 文本中核到（作者表述，未核） |
| 「准确度提升超过五倍」 | accuracy improved more than fivefold | 论文度量为 millinats 平均损失：第二周期 58 mnats，较 MC-2A「超五倍改进」，另有 MC-1 85 / SC-1 205 mnats 等对照 | 博客把「损失下降」说成「准确度提升」，且未给基线；读法见节点页 |
| VERID 使用的模型与硬件 | 未提 | 论文：Hamilton Microlab STAR 液体处理机 + Byonoy 读板仪 + UFACTORY Lite 6 机械臂，案例 1–3 用内部预览检查点 Gemini 4 Argon；全部用染料与去离子水 | 博客没说案例只用染料水，容易被读成「生物实验」 |
| 适配体周期 | 7–10 天 | 摘要：1–2 周验证周期、每候选 1–3 美元；图 1：约 10 天、3 个台面步骤 | 区间略异，量级一致 |
| 适配体命中率 | 33.5% / 42.1%（第 3 轮）；特异性 75.0% → 19.1% → 72.4%；「九倍」 | 与论文摘要 / 结果一致：myoglobin 10.2% → 3.7% → 33.5%（9.1 倍），α-thrombin 0.0% → 26.0% → 42.1%，特异性 98.5% | 一致 |
| 「48% 序列折叠被破坏」「63% G4」 | 博客写 | 在论文可抽取文本中**检索不到**这两个数；论文写 myoglobin 第 1–2 轮 G4 占 58.2%–60.7%，α-thrombin 第 1 轮 0/167 因引物—核心杂交 | 这两个数疑出自图 / 补充材料或不同分母，**未核到**，页面按「博客作者表述」标注 |
| 适配体亲和力 | 博客只写命中率与特异性 | 论文：8 个全长 α-thrombin 候选的单克隆滴定，文献来源 motif 的表观 KD = 24.6–133 nM，纯 de novo 骨架 KD > 1 μM（初步） | 博客省略了「de novo 骨架亲和力只有微摩尔级」这一重要限定 |
| 表位切除加速 | 「每个循环加速超过八倍」「8.6x」 | 摘要：在一个表位上把 RFdiffusion 与 AF3 GPU 时间减少约 88%；图 5b：完整 FAP + binder 约 779 token 落入 1024 桶（59.3 s / fold），裁剪 85–130 残基落入 256 桶（6.9 s / fold），约 88% 的**单次折叠**降幅（59.3/6.9 = 8.6 倍，为令牌桶模型预期） | 8.6x 是单次 AF3 折叠、令牌桶模型下的预期值，不等于整个设计循环；「不损结构精度」的论文依据是 15 靶 3,666 设计的中位 Cα RMSD 0.37 Å，且论文自述小于约 85 残基的裁剪对部分靶点不可靠 |
| 「近乎翻倍」 | 人机混合 vs 科学家单干 | 5.06% → 9.33%（+84%）、约 58% GPU 预算、23.2 vs 12.6 hits/GPU-h；还有全自主（HOOTL）臂，博客未提 | 一致；博客省略了全自主臂与 Gemini 3.1 Pro 是 agent |
| 「个位数纳摩尔」 | 「few experimental runs」「drug-like」 | 论文：四个富集克隆的表观 KD = 3.0–9.1 nM，来自酵母展示克隆结合实验，三轮 FACS（300/150/75 nM）；可溶表达的结合表征、FAP 域特异性与抑制实验写为未来工作 | 博客强度偏高：是酵母展示表观 KD、4 个克隆；「几次实验」对应 3 轮 FACS |
| 三者如何串成一条栈 | 结语说「下一步」把 VERID 与蛋白闭环合并 | 三篇各自独立评估；论文间没有联合运行 | 博客的「栈」是路线图，不是已运行的一体系统 |

## 对 wiki 的映射

- [google-grounding-ai-agents-wet-labs](../../wiki/entities/google-grounding-ai-agents-wet-labs.md) — 博客节点（本归档为其 `original`）。
- 三篇论文各自的独立节点由其他条目维护；本节点只引用其摘要级口径，未复制其正文。
