# The Imitator Game（意图级模仿基准）

> 来源归档（ingest）

- **标题：** The Imitator Game: Benchmarking Robot Imitative Ability Beyond Action Prediction
- **类型：** paper
- **原始链接：** <https://arxiv.org/abs/2608.22301>（PDF：<https://arxiv.org/pdf/2608.22301v1>，v1，2026-08-23，cs.RO）
- **机构：** 香港大学（HKU）；超忆（TranscEngram）；复旦大学（Fudan）；浙江大学（ZJU）
- **作者：** Xunzhe Zhou（项目负责）、Yiyang Cai、Fengyi Wang、Ran Ju（以上共同一作）、Hanxiang Ren、Ruizhe Liu、Yu Zhang、Qian Luo、Feng Chen、Pei Zhou、Yi Ma、Yanchao Yang（通讯）
- **项目页：** <https://imitator-game.github.io/> — 归档见 [`sources/sites/imitator-game-github-io.md`](../sites/imitator-game-github-io.md)
- **代码：** <https://github.com/imitator-game/The-Imitator-Game>（MIT）— 归档见 [`sources/repos/the-imitator-game.md`](../repos/the-imitator-game.md)
- **数据：** <https://huggingface.co/datasets/imitator-game/IG-10K-Dataset> — 归档见 [`sources/datasets/ig-10k.md`](../datasets/ig-10k.md)
- **入库日期：** 2026-08-30
- **再核日期：** 2026-09-02；全文通读核对：2026-11-12
- **一句话说明：** L0–L3 四级基准把人类示范与机器人现场的差距逐步拉大，用目标等价而非动作相似衡量模仿；配套 IG-10K 与盲测 Arena。
- **核对范围：** arXiv v1 PDF 全文：摘要、正文 §1–§5（含 Limitations）、附录 A（场景分解与四级细节）、B（数据采集与任务套件 Table 6、语料划分 Table 7、Arena 打分流程与指标定义 Table 8、基线适配）、C（C.1–C.8 的补充实验，含 Table 9、Table 10 与 Figure 8–13 的文字解读）、D–G（掩码、MANO、基线超参、失败模式）；Table 1–5、Table 7–10 的数字逐项核对。未逐格核对：附录 H 的 Table 11（逐任务逐级的仿真成功率，仅读了 G 节对它的文字解读）、Table 6 里每个任务的 ✓/– 逐格（只核对了各域任务数与 L3 缺项说明）、Figure 8–13 中柱状/曲线的读数（以正文与表格数字为准）。项目页、仓库与数据集页的内容来自各自归档，未在本次重新抓取。
- **通读：** 已完成

## 核心摘录

### 1) 轨迹复现 ≠ 意图模仿（§1、§3.1.1）

- **摘录要点：** 作者自述：现有策略学观测到动作的映射，对人类视频多在近似场景复现轨迹；真正的模仿是推断目标并用手头物体完成（§1）。形式化：策略 \(\pi(a_t\mid o_t,V)\)，任务只由示范视频 V 隐式给出，没有任务描述、目标谓词或符号规格；目标谓词只在评测平台内部用于仿真打分，从不作为策略输入（§3.1.1）。VLA 经平台侧固定 captioner \(T(V)\) 才看到语言；层级标签不告诉策略（§3.2.2）。
- **对 wiki 的映射：**
  - [Imitator Game](../../wiki/entities/paper-imitator-game.md)
  - [模仿学习](../../wiki/methods/imitation-learning.md)
  - [VLA](../../wiki/methods/vla.md)

### 2) 场景分解与四级保真（§3.1、附录 A）

- **摘录要点：** 场景 \(S=(\{O_i=(A_i,G_i,S_i)\},P)\)，A 外观、G 几何、S 语义类别、P 空间配置；操作性假设 \(S\)变 ⇒ \(G\)变 ⇒ \(A\)变，反向不成立（§3.1.1）。Table 1：L0 保 P/A,G/S（轨迹匹配）；L1 改 P（物体终态）；L2 改 A 或 G、保 S（语义任务）；L3 连 S 也换（功能替代，如茶包换抹茶粉改为搅拌而非浸泡，Figure 2）。P 只在两场景物体集相同时才可比，故 L2/L3 的 P 列写 n/a 而非 ×（附录 A.1）。四级是作者自己的综合，借用发展心理学构念作标签，明确声明不主张机器人具有心智状态（附录 A.2）。同一人类 clip 服务四级，只改机器人侧场景（§3.2.1）。附录 A.2 自述该层级与 RHyME 等具身差距层级正交：本基准固定具身差距、改场景。
- **对 wiki 的映射：**
  - [Imitator Game](../../wiki/entities/paper-imitator-game.md)
  - [Manipulation](../../wiki/tasks/manipulation.md)

### 3) IG-10K + Arena 协议（§3.2、附录 B）

- **摘录要点：** 20,000+ 配对（真机 11.7K Realman 双臂 VR 遥操作 + 仿真 10K 双臂 Franka / ManiSkill3，OMPL 规划手工路点轨迹），50+ 基础任务、6 域（Household 12、Supermarket 7、Restaurant 17、Logistics 4、Hospital 6、Laboratory 7），200+ 任务变体，LeRobot-0.5.0 格式（§3.2.1，Figure 3）。仿真配对复用同任务的真机人类 clip，仅换机器人侧场景；MANO 仅真机人类 clip 有（附录 B.2）。每个任务–级别只用一种替换模式（§3.2.1）。少数任务缺 L3（危害或语义不明确，Table 6 注记）。语言标注三层抽象，人工撰写并用 Qwen3-VL 扩充；MANO 由 WiLoR 估计，掩码由 Grounded-SAM-2 生成，均为离线自动标注（附录 B.1、D、E）。
- Arena：评估者看示范与两条匿名随机顺序 rollout，回答完成度（SUCCESS/PARTIAL/FAIL）、模仿质量 q∈[0,10]（Table 8 五锚点）、成对偏好；得 SR_human（仅 SUCCESS）、Q、WR。10 名志愿评估者，每屏一人；总计 15,000 对仿真 + 5,000 对真机，基于 36,000 仿真 rollout 与 4,800 真机 rollout；每个示范–rollout 对被不同评估者判 3–5 次；未报告评估者间一致性统计，作者列为计划扩展（附录 B.3）。仿真自动指标 SR（所有目标谓词成立）与 Sub-SR（有序子目标峰值子奖励过阈值的比例），逐任务手工编写（附录 B.3，Pour Kettle 示例）。自动 SR 与 Arena 人类判断在同一仿真 rollout 上 r=0.858（SR vs SR_human）与 r=0.861（Sub-SR vs Q）（附录 C.7，Figure 13）。
- **对 wiki 的映射：**
  - [Imitator Game](../../wiki/entities/paper-imitator-game.md)
  - [具身评测基准枢纽](../../wiki/overview/hub-embodied-eval-benchmark.md)
  - [LeRobot](../../wiki/entities/lerobot.md)

### 4) 九模型横评数字（§3.2.3、§4、附录 B.4、C、F）

- **摘录要点：** 三族共 9 个模型 15 个训练变体：VLA（OpenVLA、RDT-1B、GR00T-N1.6、π0.5）用固定 captioner；技能族（XSkill、UniSkill）；视频条件视觉动作（ACT/DP/VQ-BeT × DINOv2/SigLIP2/VideoMAE）。视频编码器或 VLM 骨干冻结，只训动作模块，并缓存骨干激活（§3.2.3）。作者在附录 B.4 称早期试验解冻后两侧收益相当，但未给数字。评测 10 任务（5 seen + 5 unseen）× L0–L3，仿真每对 10 次、真机 5 次；预训练语料 15⊂30⊂45 任务，每任务 50 条；few-shot 为 10 条机器人示范（§4.1）。
- 仿真（Table 3，三个语料规模均值）：seen SR——ACT/DINOv2 0.81、XSkill 0.79、ACT/SigLIP2 0.79、UniSkill 0.75、π0.5 0.73、GR00T 0.51、OpenVLA 0.29、RDT-1B 0.43；零样本最高 0.13（VQ-BeT/DINOv2），π0.5 0.09；Scr. 最高 π0.5 0.80，P+FT 最高 π0.5 0.85；P+FT 相对 Scr. 有 12/15 变体为正（RDT-1B、VQ-BeT/DINOv2、VQ-BeT/SigLIP2 为负）。从 15 到 45 任务，仿真 14/15、全部 19 个变体中 18 个 P+FT 上升（§4.3、附录 C.3）。
- 真机（Table 4，四个代表模型，SR 为人类判断）：XSkill seen 0.63 / ZS 0.29 / Scr. 0.28 / P+FT 0.49，P+FT 的 WR 0.95；π0.5 seen 0.51、ZS 0.04、P+FT 0.36。真机零样本随语料 15→45：XSkill 0.24→0.33，ACT/DINOv2 0.22→0.30，DP/DINOv2 0.19→0.25，π0.5 约 0.04 不变（附录 C.3）。
- 分级（Table 5，真机 P+FT，四模型均值）：SR L0/L1/L2/L3 = 0.42/0.42/0.39/0.29；Q 6.26/6.22/6.17/5.62；XSkill 0.53/0.57/0.56/0.29。附录 C.5：真机 seen 任务 SR 0.54/0.55/0.54→0.39；但仿真并非 L3 最低——仿真 seen 的 SR 曲线在 L2 凹陷后 L3 回升（0.62/0.57/0.48/0.63），仿真未见任务 P+FT 为 0.58/0.55→0.37→0.54（作者归因于仿真 L2 同时换了执行手臂，且仿真资产有限、规则规划使替换物体仍可沿用类似策略），因此正文对仿真汇总不按级拆分。
- 编码器：DINOv2/SigLIP2 优于 VideoMAE，影响随动作头而异（DP seen 0.67→0.14，ACT 0.81→0.72）（附录 C.2）。示范互换检验（Table 10，仿真 seen，45 任务检查点）：换成相似任务视频时 ACT/DINOv2 0.81→0.31、DP 0.67→0.03、π0.5 0.73→0.15、XSkill 0.79→0.31；换成无关任务视频时 0.30/0.00/0.01/0.23（附录 C.6）。captioner 检验：用纯人工标注替换 VLM 增强标注，π0.5 seen SR 变化 0.01±0.05（附录 B.4）。
- **对 wiki 的映射：**
  - [Imitator Game](../../wiki/entities/paper-imitator-game.md)
  - [模仿学习](../../wiki/methods/imitation-learning.md)

### 5) 作者自述的结论与局限（§5、附录 G）

- **摘录要点：** 作者结论：意图级模仿不太可能只靠扩大语料或动作头自然出现，应让示范作为「关于目的的证据」保留（affordance 感知的物体落地、显式目标推断、能跨物体替换存活的中间表示）。局限（§5）：IG-10K 有限且人工设计；每个任务–级别单一替换模式、不测替换方差；层级只分级场景失配、固定具身差距；有限资产和规则规划使仿真 L3 结果有偏；被评策略是适配到本游戏而非为意图模仿设计。附录 G：未见任务应读作部分泛化而非解决，真机少样本数据仍太小，难以校准抓取点、释放时机与分级 affordance 替换。
- **对 wiki 的映射：**
  - [Imitator Game](../../wiki/entities/paper-imitator-game.md)

### 6) 开源状态（截至 2026-09-04 再核仍成立）

- **摘录要点：** **已开源（MIT）**。官方仓含 ManiSkill/SAPIEN 仿真、采集与九套基线训练/评测入口；HF 发布 IG-10K-Dataset 与 IG-10K-Assets（ModelScope 镜像 `Zhouxunzhe/*`）。Arena / 文档 / 数据页在项目站。真机部署评测走社区申请，仓内无上传接口。2026-08-30 浅入库时训练仓尚未公开，本次复核已改判。（此条来自项目页 / 仓库归档，非论文正文。）

## 论文内部不一致

- 摘要与正文称「每个模型零样本低于 13%」，但 Table 3 里 VQ-BeT/DINOv2 零样本 SR 为 0.13，附录 C.4 又写仿真零样本 SR ≤ 0.14。
- Figure 4 的未见任务示意（扔水果入垃圾桶、扫描果汁、封箱、倒水等）与附录 Table 7 的未见任务名单（PickRemoteControl、ScanMilkBox、PourKettle、PickFood、FoldBox）不完全对应。
- 正文 §4.1 把 seen 任务之一写作「pick up a paper」，附录 B.2 写作「filing a folder」，Figure 4 写作「Open the folder and place files」。

## 当前提炼状态

- [x] 项目页与 arXiv 全文对齐（2026-09-02 再核）
- [x] 步骤 2.5：GitHub + HF 数据/资产已发布
- [x] wiki 映射：`wiki/entities/paper-imitator-game.md` 深化
- [x] 全文通读与逐项核对（2026-11-12，范围见上）
