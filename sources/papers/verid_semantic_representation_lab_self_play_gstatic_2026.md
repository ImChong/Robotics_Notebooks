# Semantic Representation for Agentic Self-Play in Automated Laboratories（VERID）

> 来源归档（paper / Google Research 技术报告）

- **标题：** Semantic Representation for Agentic Self-Play in Automated Laboratories
- **作者：** Yves Delley, John Coller, Zachary Cutts, Alexander Daryin, Mathias Bellaiche, Jun W Kim, Oscar Donosa, Srinidhi B Seshadri, Tanner Clary, Sandy Klemm, Avinatan Hassidim, Yossi Matias, Pooja Rao, Petar Sirkovic, Annalisa Pawlosky（通讯：Delley / Sirkovic / Pawlosky；Sirkovic 与 Pawlosky 为共同资深作者）
- **机构：** Google Research、Google Cloud、Google DeepMind
- **PDF：** <https://www.gstatic.com/science-ai/verid_semantic_lab.pdf>（页眉日期 2026-10-22，PDF 元数据创建时间 2026-10-22，Typst 排版，A4，16 页，约 2.3 MB）
- **arXiv：** [arXiv:2610.32755v1](https://arxiv.org/abs/2610.32755)（2026-10-23 提交，cs.RO / cs.AI，备注 16 pages, 4 figures；DOI 10.48550/arXiv.2610.32755）
- **许可：** 论文页脚 CC BY 4.0
- **版本核查：** arXiv 检索到 v1（摘要措辞比 PDF 略精简：PDF 摘要多一句 "Together, these studies demonstrate..." 并写 "robust structured multimodal experimental data collection"，arXiv 摘要写 "VERID also captures structured multimodal experimental data"）；Crossref 标题检索未见期刊版本；Google Research publications 页检索未发现该条目（未登录检索，不代表不存在）。
- **代码 / 数据：** 全文无代码 / 数据可用性声明，无仓库链接；arXiv 摘要页无代码链接。GitHub 搜索接口 502，未能用搜索确认，故结论为「论文未声明开放」。
- **入库日期：** 2026-11-07
- **核对范围：** 通读 PDF 全部 16 页：摘要、§1 引言、§2 VERID 架构与四项服务、§3 三个案例（图 1-4、全部正文数字）、§4 讨论（4.1-4.5 含安全与生物安全）、§5 结论、§6 方法（6.1 工作站与试剂、6.2 软件架构与 agent 接口、6.3 案例 3 评分公式 1-5）、致谢与作者贡献、参考文献 [1]-[37]；该文无附录。参考文献仅核对与正文引用的对应，被引论文本身（如 PyLabRobot [25]、Safe-SDL [35]）未阅读。
- **通读：** 已完成

## 核心摘录

### 1) 问题与定位

- 位置：§1。主张自主科学的两个问题（搜索巨大空间 + 高保真物理执行）被分开处理；现有框架「把已编译的单元操作当作理想的，或把状态绑在单次临时脚本执行上」，没有跨命令块与硬件故障持久存在的数字孪生。引用 PyLabRobot 提供程序化硬件控制与内存中的体积 / 吸头追踪。
- 对 wiki 的映射：[VERID 实体页](../../wiki/entities/paper-verid-semantic-representation-lab-self-play.md)

### 2) 语义数字孪生与四项服务

- 位置：§2、图 1。孪生是长期存在、由服务端管理的实体树，记录工作台上每个物件（机械臂、读板器、孔、吸头）的空间坐标、容器几何、运行状态与语义内容（试剂身份、体积、液面高度）；本体层次形式化「包含」与「空间」关系（孔属于板、盖子阻挡孔）。四项服务：Introspection、Simulation Gate、Safe Execution、Trace Dossier。每个命令块必须先过仿真门再物理下发；仿真克隆当前孪生状态执行，追踪质量守恒、吸头状态、台面几何与耗时，孵育可快进。
- 对 wiki 的映射：同上

### 3) 案例 1-3 数字

- 案例 1（图 2）：11 步 2 倍系列稀释、四种染料；初版 z = 0 mm，实测步稀释倍数仅约 1.4x-1.5x（红 / 绿 / 黄）、1.6x-1.8x（蓝），每步携带 68-70% 染料；改为 1.5 mm 底部间隙并在 1.5 mm 吸、4.8 mm 打的垂直对流混匀后，平均步稀释 1.92x-1.99x，管与槽之间差异 ≤0.02x。
- 案例 2（图 3）：8 通道同时入槽需要至少 72.0 mm，超过 71.2 mm 槽长，被仿真拒绝；改 7 通道混匀（63.0 mm）；中途通信故障后从孪生保存的状态续跑；事后发现通道 8 吸光度下降 28-47%，干吸头基线平坦（1.58 OD，CV = 1.4%），反向分配后缺口转移到通道 1，归因于 7 吸头槽内混匀的内壁液膜携带 +4.9 µL（+40.6%），同时解释 24 µL 与 12 µL 两个稀释水平之间 14.5% 偏差。
- 案例 3（图 4）：Challenger 配制四个盲二元染料混合物 + 四个纯参考；Solver 只有一块 96 孔板、受限吸头数、俯视 RGB 相机；Oracle 为 Byonoy Absorbance 96 四波长读板器；评分为回火 KL 散度（mnats，越低越好）。MC-1 85 mnats vs SC-1 205 mnats；SC-2A 142 到 SC-2B 162 mnats 停滞；MC-2A 从 247 退化到 307 mnats；继承「样本盲」经验后 MC-2B 平均 58 mnats（文中称比 MC-2A 提升 5 倍以上），其 Cycle 1 为 137 mnats。
- 对 wiki 的映射：[VERID 实体页](../../wiki/entities/paper-verid-semantic-representation-lab-self-play.md) 的「实验与评测」节

### 4) 工作站与软件（§6.1-6.2）

- Hamilton Microlab STAR 液体处理机（96 通道头 + 8 通道独立头 Span-8）；Byonoy Absorbance 96 四波长读板器；UFACTORY Lite 6 六自由度机械臂在液体处理机与读板器间转运板；OBSBOT Tiny 2 相机装在液体处理机龙门上；试剂为去离子水与零售食用色素。VERID 通过基于开源 PyLabRobot 的 Python 设备驱动与自定义仪器接口控制硬件；agent 经 client-server 结构化工具调用访问（工作站检视、命令校验、命令下发、状态 / 完成轮询）。案例 1-3 使用 Gemini 4 Argon 的内部预览检查点。
- 对 wiki 的映射：[Gemini 4 Argon](../../wiki/entities/gemini-4-argon.md)

### 5) 局限与未做之事（作者表述）

- 仅用染料与去离子水验证，未涉及生物试剂；对「重大生物学突破」的验证是下一步；跨已归档 trace dossier 的大规模索引与知识抽取及其对下游活动的基准评测为 future work；自动生物安全网关（SecureDNA 类筛查）、成分谱系追踪为设想（§4.5.3）。

## 对 wiki 的映射

- 主节点：[paper-verid-semantic-representation-lab-self-play](../../wiki/entities/paper-verid-semantic-representation-lab-self-play.md)
