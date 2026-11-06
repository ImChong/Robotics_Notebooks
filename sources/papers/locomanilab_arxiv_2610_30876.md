# LocoManiLab: A Unified Testbed for Data, Policies, Controllers, and Deployment in Humanoid Loco-Manipulation

> 来源归档（paper）；2026-10-26 依据 arXiv v2 摘要页与项目页入库，2026-11-06 依据 arXiv v4 PDF 全文、项目页、官方仓库与 Hugging Face 数据集复核。

- **作者：** Dewei Wang、Jinrui Han、Jiyuan Shi、Chenyun Zhang、Xianbo Lu、Jiabao Wu、Jiqin Feng、Jipeng Kong、Xinzhe Liu、Ping Luo、Yuchen Xiao、Xingxing Wang、Chenjia Bai、Xuelong Li（v4 作者表；v2 无 Xinzhe Liu、Yuchen Xiao、Xingxing Wang）
- **机构：** 中国电信人工智能研究院（TeleAI）、中国科学技术大学、香港大学、浙江大学、上海交通大学、上海科技大学、宇树科技（Unitree Robotics，v4 新增）、Gamma Robotics
- **版本/日期：** arXiv:2610.30876v4，cs.RO；v1 2026-10-22，v2 10-23，v3 10-26，v4 10-28。v4 与 v2 PDF 逐词比对，差异仅为作者与机构增补及日期行，正文、表格和数值未变（v3 未单独比对）。
- **论文：** <https://arxiv.org/abs/2610.30876>
- **PDF：** <https://arxiv.org/pdf/2610.30876>
- **项目页：** <https://locomanilab.github.io/>
- **代码：** <https://github.com/Gamma-Robot/LocoManiLab>（自有代码 MIT；首个提交 v1.0.0 为 2026-11-02，至 2026-11-05 共 3 次提交）
- **数据 / 权重：** <https://huggingface.co/datasets/GammaRobotics/LocoManiLab>（公开、未设访问门槛；2026-10-30 创建，2026-11-02 最近更新）
- **核对范围：** 2026-11-06 通读 arXiv:2610.30876v4 PDF 全文 27 页：正文 §1–6（表 1–6、图 1–5 及图注）、参考文献、附录 A–E（任务清单与评测切分、DR 预设、12 种策略的适配与训练配置、控制器接口、指标与聚合、示范规模 / 数据源 / Kimodo 微调与评测条件、仿真与真机协议；表 7、图 6–10）；与 v2 PDF 逐词比对；另核对项目页、官方仓库 README 与 docs（models、hf_resources、THIRD_PARTY）、CHANGELOG 及 HF 数据集卡与文件清单。未运行代码。
- **通读：** 已完成
- **入库日期：** 2026-10-26

## 摘要式摘录

LocoManiLab 将人形移动操作的数据采集与生成、策略训练、评测、全身控制和实机部署放进同一测试床。任务集覆盖 5 类家庭场景中的 234 个原子与组合任务，论文报告 51.5 小时模拟遥操作数据。作者比较共享控制器下的 12 种策略、5 种控制器下的策略表现，并分析示范规模、数据来源混合、未见任务/场景和视觉扰动对泛化与鲁棒性的影响。

论文还微调一个人形动作生成模型以生成移动操作参考动作，研究合成动作与任务执行要求的对齐情况，以及它在数据生成链路中的作用。三个仿真任务与现实机器人设置相对应，用于跨仿真和真机比较策略表现。

## 项目页可核验信息

- **任务与场景：** 102 个原子任务、132 个组合任务、13 个场景实例，涵盖厨房、客厅、卧室、浴室和办公室；另有两个 Real2Sim 重建场景。
- **遥操作数据：** 项目页列出 14,160 段 episode、50 小时以上数据；策略执行频率展示为 50 Hz。机器人接口展示 Unitree G1 与 Dex3 灵巧手。论文附录 A 写明 14,160 段即五种控制器合计的 51.5 h 模拟遥操作；项目页 SONIC 原子 / 组合拆分写作 5,258 / 6,801，论文为 5,207 / 6,852，合计一致。
- **系统比较：** 12 种策略与 5 种全身控制器由 Loco-Mani Arena 汇总比较。论文摘要指出，未见任务与场景成功率下降，视觉扰动通常也会降低成功率，且不同策略的泛化和鲁棒性存在差异。
- **动作生成：** 项目页给出 Kimodo 在 30 个 task-scene 配对、每个模型 10 次试验上的总体任务成功率：预训练 39.7%、LoRA 微调 71.3%、全量微调 90.7%，与论文表 5(a) 一致。这是该特定对比设置的结果，不应外推为全部生成动作的总体成功率。
- **仿真到真机：** 官网提供开冰箱、货架到桌面搬运、传送带抓放三个任务的仿真与真机视频对照；论文用这三项任务检验仿真评测与实机表现的一致性。

## 全文复核要点（2026-11-06）

- **系统与接口：** 场景在 MuJoCo 中运行，物体来自 RoboCasa 与 Objaverse，房间布局来自 MolmoSpaces 的 iTHOR 衍生场景与 Holodeck 办公室。主评测输入为一路 320×240 头部 RGB 与 43 维关节状态（GR00T N1.7 另加 3 维重力投影，StarVLA-α 不用本体感知），输出 64 维 SONIC motion token 加 14 维手指目标；共享驱动以 50 Hz 下发命令，每 20 个控制步同步查询一次策略。五种控制器动作维度为 AMO 36、SONIC 78、HEFT 50、ALMI 52、ScaleBFM 50（§3.2、附录 A、B-B）。
- **数据：** 三类示范来源为 VR 遥操作（仿真与真机）、动作合成和轨迹增强。14,160 段 / 51.5 h 模拟遥操作中，SONIC 主语料 12,059 段，其余四种控制器在 10 个共享浴室任务上共 2,101 段；另有 567 段、约 9.0 h 真机遥操作，覆盖开冰箱、搬杯、抓瓶和传送带四类任务（§3.4、附录 A、E）。
- **策略与训练：** 12 种策略为 10 个预训练 VLA、Fast-WAM 和 Diffusion Policy 基线，原子与组合任务各用 50 任务清单分别训练。各策略保留原有预处理与 action chunk 长度，训练预算约为两轮名义遍历，不代表等算力；每个配置只训练一次，取预算末 checkpoint，不按验证或测试分数挑选（§4.1、附录 B-A）。
- **主结果（表 2，SONIC、无 DR）：** 切分为 seen 100、unseen task 123、unseen scene 11 个任务，按任务等权平均。XR-1 总平均 47.9% 居首，seen 80.2% 降到 unseen task 25.2%；unseen scene 最高为 DM0.5 的 15.5%；Diffusion Policy 为 7.1%。原子任务表现相近的策略在组合任务上会拉开差距。
- **视觉扰动（表 3）：** Low / Mid / High 三档只在 seen 任务上测试，每档每任务 20 回合。DM0.5 总平均 70.2% 居首；XR-1 在 Low 领先，Mid / High 被 DM0.5 超过。High 的干扰物只在 100 个任务中的 38 个放得下（附录 A-A）。
- **多控制器与示范规模（图 3、图 4）：** GR00T N1.7 与 π0.5 在 5 个原子、5 个组合浴室任务上，按各控制器命令格式和对应数据分别训练。SR 最高的组合不一定运动质量最好：GR00T N1.7 原子任务上 ALMI 的 SR 最高，但成功回合跌倒多于 SONIC；SONIC 的机械输出功率最低。示范由每任务 10 条增至 50 条时，GR00T N1.7 原子任务的五控制器平均 SR 由 33.0% 升至 69.8%；训练步数随数据同比增加，并非等算力比较。SONIC 加到 100 条时只有 π0.5 原子任务继续提升（附录 D-A）。
- **数据源（表 4）：** 20 个任务上遥操作单源最好（GR00T N1.7 82.0%、π0.5 89.2%），两种生成源单独使用都不如遥操作。GR00T N1.7 加入轨迹增强后，组合任务由 56.2% 升至 68.8%，总分 82.5% 基本持平；三源混合则降到 46.2%。混合未均衡来源数量与采样概率；轨迹增强的 1,000 条有效回合来自 1,606 次尝试，本实验未用在线 LLM 规划（附录 D-B、D-C）。
- **Kimodo 微调（表 5、附录 D-C/D-D）：** Kimodo 以动作文本、已执行动作前缀和根部 / 腕部目标为条件生成 3 秒、30 Hz 的 G1 参考动作，由冻结 SONIC 执行。作者按扩散模型强化学习的做法，用 PPO 截断目标微调去噪器，奖励来自仿真中的控制器执行结果，不对仿真器求导。比较对象为 LoRA（rank 16）与全量微调。30 个 task-scene 对中有 13 个与微调任务重叠；目标误差测试的提示词和目标范围与训练重叠，属分布内评测。
- **仿真与真机（表 6、附录 E）：** GR00T N1.7、Fast-WAM、XR-1 在每个任务上分别训练：仿真用 50 条模拟遥操作、评测 20 回合；真机用 50 条真机示范、试验 10 次，由人工判定成功。平均成功率仿真为 XR-1 76.7%、GR00T N1.7 53.3%、Fast-WAM 50.0%，真机为 70.0% / 53.3% / 46.7%，平均排序一致；但 GR00T N1.7 与 Fast-WAM 在搬杯任务上的先后顺序在两域相反。仿真与真机场景只是近似对应，并非几何配对；传送带任务的判据也不同（仿真须三箱都放上桌，真机只需从传送带取起一箱）。真机用头部 ZED Mini 的左目 RGB，仿真用 G1 自带头部相机。
- **作者自述局限（§5）：** 基准主要覆盖室内家庭与办公任务；硬件研究只有 3 种策略 × 3 个任务 × 10 次试验，尚需更多模型、活动、环境、重复训练和实机试验；机载算力受限时依赖边缘 / 云推理带来的通信延迟与控制超时尚未评估。

## 开放状态

**2026-11-06 复核：** 项目页的 **Code** 与 **Dataset** 已分别链接官方 GitHub 仓库与 HF 数据集；2026-10-26 入库时两者仍标 “Coming soon”。

- **仓库：** `lab` CLI 覆盖 collect → prepare → convert → train → evaluate → deploy；`src/locomanilab/benchmark/tasks.yaml` 记录 234 个任务的评测设置，`lab eval --task / --task-set` 可在仿真中闭环评测。仓库含 12 个模型 ID（ACT、DP、DM0.5、GR00T、FastWAM、starVLA、PRTS、π0.5、Ψ0、InternVLA-A1.5、DreamZero、XR-1；ACT 与 DP 共用一个适配模块）。G1 控制器 SONIC、HEFT、AMO、ScaleBFM、ALMI 经 `g1_backend` 子模块集成，仿真宿主为 `humanoid_benchmark` 子模块。仓库模型集合与论文 12 种策略不一致：不含论文评测的 SmolVLA、X-VLA，另含 ACT、DreamZero。
- **README 自述边界：** “Model integrations are retained for research and have not all been revalidated end to end”。CPU 样例 “is not a robot demonstration or a training benchmark”。`docs/models.md` 标明各集成均未重新做端到端验证，且仓库不含精度数字。
- **HF 数据集：** `dataset/` 为 LeRobot v2.1 格式（33 GB），含 `main/`（234 个基准任务）、`data_generation/`（human / gpt_aug / kimodo）、`multi_controller/`（五种控制器）、`real/` 与 `real2sim/`（三个 TeleLab 任务）；`asset/` 为仿真资产清单，第三方资产从原站下载；`checkpoint/` 含 GR00T N1.7（train50，step 10000）、DM0.5 与 XR-1（step 50000）的推理权重，没有优化器状态、不能续训，另有五种控制器权重。
- **许可：** 仓库自有代码 MIT；HF 上的自录数据、微调权重与 Gamma Robotics 自有资产 MIT。依赖、基座模型、控制器权重（SONIC 为 NVIDIA Open Model License）与仿真资产各守原条款；`docs/THIRD_PARTY.md` 将 PRTS（CC BY-NC 4.0）与 InternVLA-A 系列（CC BY-NC-SA 4.0）列为受限集成；任务加载的 2 个 Objaverse 物体为 CC BY-NC 4.0。
- **判定：** 已开源（代码、基准数据、部分策略与控制器权重）。论文 12 策略主表不能只凭公开权重直接复现，集成也未全部端到端复验。

## 对 wiki 的映射

- [LocoManiLab 项目实体](../../wiki/entities/paper-locomanilab.md)
- [移动操作任务页](../../wiki/tasks/loco-manipulation.md)
