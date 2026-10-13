# If, Then, Otherwise: Diagnosing Conditional Branching in Vision-Language Navigation（arXiv:2608.17318）

> 来源归档（ingest）

- **标题：** If, Then, Otherwise: Diagnosing Conditional Branching in Vision-Language Navigation
- **类型：** paper / vision-language-navigation / benchmark / conditional-reasoning / embodied-ai
- **arXiv：** <https://arxiv.org/abs/2608.17318>
- **HTML：** <https://arxiv.org/html/2608.17318v1>
- **PDF：** <https://arxiv.org/pdf/2608.17318>
- **作者：** Seoyoung Lee、Neel P. Bhatt、Pranay Samineni、Cong Liu、S P Sharan、Timothy Barclay、Gregory M. Wagner、Daniel Milan、Sandeep Chinchali、Ufuk Topcu、Atlas Wang
- **机构：** 德克萨斯大学奥斯汀分校（The University of Texas at Austin）；柯林斯航空航天公司（Collins Aerospace）
- **发表日期：** 2026-08-18（arXiv v1）
- **基准：** CondVLN
- **项目与代码/数据入口：** [官方项目页](https://condvln.github.io/)；页面的 Code / Dataset 按钮分别指向匿名 4open artifact，入口与核验范围见[项目页核查](../sites/condvln-project-page.md)
- **入库日期：** 2026-10-13
- **一句话说明：** 将可验证的场景图谓词转成不同嵌套深度和分支链长度的条件导航指令，并用 Branch Selection Accuracy 与 Conditional Success Rate 区分“走得像”与“按场景条件选对分支并走完”。

## 核心摘录（面向 wiki 编译）

### 1. 问题与 benchmark 设计

传统 VLN 常以成功率（SR）、路径长度或 SPL 衡量“是否到达目标”，但这些指标不一定能识别智能体是否遵守了指令里的 if/else 条件。CondVLN 把条件分支作为可控变量，重点诊断条件接地、逻辑分支选择和后续路径执行。

论文从 AI2-THOR、Matterport3D（MP3D）、Gibson 与 ReplicaCAD 的室内 3D 场景结构出发，统一成“房间 → 对象实例”的层级。对象记录语义标签和三维中心；数据可用时还保留尺寸、半径、楼层面积或轴对齐包围盒（AABB）。

### 2. 场景图如何产生可验证条件

- 对同房间对象对构造有向关系边；空间距离优先使用 AABB 表面距离，其次包围球表面距离，最后回退到对象中心距离，以适应不同数据源的几何标注。
- 方向谓词把方位离散成 8 个水平象限，并与 up / level / down 组合；另以固定几何阈值产生 near、far、above、below 等语义关系。
- 每条关系保留对象、谓词、目标对象和生成谓词时的度量信息，使条件真值能回溯到场景图。
- 条件对象既可指向房间中存在的对象，也可指向该房间没有、但场景其他位置有的类别，生成真 / 假条件。遇到同类多实例时使用空间限定词消歧；仍无法唯一指代的对象会被排除。

### 3. 指令复杂度与 episode

指令由场景图谓词模板生成，基本形式为“IF 条件 THEN 目标 A ELSE 目标 B”，也可嵌套或串联条件。论文按逻辑嵌套深度与分支链长度组织六类复杂度：Simple、Nested、Deep Nested、Chain、Long Chain、Nested Chain。

生成指令随后被转换成 Habitat / VLN-CE 兼容 episode：把起点和所选分支目标转换到 Habitat 坐标系，用 Habitat-Sim pathfinder 求测地最短路径，过滤寻路失败项，并保存参考路径、测地距离和场景元数据。论文当前实例报告 **超过 11,500 条**条件指令，覆盖上述四类环境；该数字是论文描述的生成指令规模，不应自动理解为每条都能成功转为可执行 episode 或数据已可下载。

### 4. 诊断指标

- **Branch Selection Accuracy（BSA）：** 根据正确分支中的有序子目标，计算智能体按顺序完成的最长前缀占比；允许部分得分，不要求最终目标一定完成。
- **Conditional Success Rate（CSR）：** 严格指标，要求完整完成分支子目标并达到标准 VLN 成功条件。
- **标准指标：** 同时报告 SR、Oracle SR、到目标距离和 SPL 等，但这些只反映到达及效率，不能单独确认分支逻辑是否正确。

BSA / CSR 对无子目标 episode 采用论文约定：成功样本的 BSA 记为 1，失败的无子目标样本不纳入该指标的样本集合。读表时应注意这个纳入规则。

### 5. 实验设置与结果读法

论文评估 VLN-Zero、NaVid、NaVILA、Open-Nav 四类现有 VLN agent；原则上不重新训练，以观察已有能力在条件指令下的表现。评测统一基于 VLN-CE 兼容接口，条件分支只通过自然语言提示给到 agent。

论文报告 NaVid、NaVILA 的 CSR 整体偏低且不稳定，Open-Nav 与 VLN-Zero 相对更强；但所有方法在更深嵌套或更长条件链上都会遭遇退化。关键结论不是某个 agent 的单一总分，而是 SR / SPL 看起来尚可时，BSA / CSR 仍可能暴露“走到一个合理地点，却选错条件分支”的错误。

作者另构造一个轻量神经符号分支选择对照：利用生成数据时记录的真值条件，先把条件句改写为线性导航指令，再交给不变的 VLN-Zero 执行。它用于隔离“正确选分支”对下游导航的影响，使用了构造时的 oracle 场景图信息；因此是诊断上界/对照，不等价于部署时能自行看懂场景的模型。摘要中的“2x”应按具体难度与指标理解，论文正文说明其优势并非在所有聚合结果上都一致领先。

## 对 wiki 的映射

- 新建统一项目实体：[CondVLN](../../wiki/entities/paper-condvln-conditional-branching.md)。
- 交叉：[Vision-Language Navigation](../../wiki/tasks/vision-language-navigation.md)、[VLA](../../wiki/methods/vla.md)。
- 项目页与代码/数据入口：[CondVLN project page audit](../sites/condvln-project-page.md)；项目页指向的匿名 artifact 见[代码/数据入口核查](../repos/condvln-benchmark-artifact.md)。
