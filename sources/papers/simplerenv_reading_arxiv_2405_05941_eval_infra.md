# SIMPLER 论文通读（仿真评测基础设施视角）

> 来源归档（原文通读 · 支撑 `wiki/concepts/simulation-evaluation-infrastructure.md`）

- **标题：** Evaluating Real-World Robot Manipulation Policies in Simulation
- **作者：** Xuanlin Li, Kyle Hsu, Jiayuan Gu 等（UCSD / Stanford / UC Berkeley / Google DeepMind）
- **类型：** paper / evaluation / real-to-sim
- **原文：** <https://arxiv.org/abs/2405.05941>（2026-11 通过 arXiv PDF 获取全文并用 pdftotext 通读）
- **一句话说明：** 提出 SIMPLER：不追求全保真数字孪生，只要仿真「足够真实」使策略 sim 成绩与真机成绩相关；以 MMRV + Pearson r 衡量，并通过系统辨识（控制 gap）与 green-screen / 纹理匹配（视觉 gap）实现强相关。
- **核对范围：** 通读正文 §I–§VII（含问题定义、MMRV 指标、控制/视觉 gap 缓解、主实验、分布偏移、消融、局限）及附录 A–G 中控制器算法、资产流程、Variant Aggregation、Kruskal-Wallis 与单任务策略附加实验的文字部分；附录各结果表（Table IV–XIV）的逐格数值仅抽查，未逐项核对。Genesis World 1.0 博客页面为前端渲染，抓取不到正文，本归档不覆盖该博客。
- **通读：** 已完成

## 核心摘录（均出自原文）

- **问题表述（§III-A）：** 目标不是 1:1 复现真机行为，而是仿真中的**相对策略表现**与真机强相关；仿真永远是不完美代理，不完全取代真机评测。
- **指标（§III-B，式 1–2）：** Pearson r 只衡量线性拟合，且对真机成绩落在窄带时的噪声敏感；Spearman 等纯排名指标忽略差距幅度。提出 MMRV（Mean Maximum Rank Violation，0–1，越低越好）：RankViolation(i,j)=|Ri−Rj|·1[sim 排序≠real 排序]，对每个策略取最坏违例再平均。
- **控制 gap（§IV-A，式 3–5）：** 用开环回放离线数据中的动作，最小化末端 6D 位姿平移+旋转误差，以模拟退火（3 轮）辨识 PD 参数；直接套用真机控制器 PD 值会导致抓取失败。
- **视觉 gap（§IV-B）：** green-screen（真机背景 inpaint 后与仿真前景按掩码合成：I′=M⊙Isim+(1−M)⊙Ireal）+ 物体纹理投影/机械臂颜色调节（Visual Matching）；对照方案 Variant Aggregation 为多变体（背景/光照/干扰物/桌面纹理）求平均。
- **套件（§V）：** 基于 SAPIEN，覆盖 Google Robot 与 BridgeData V2 WidowX；单环境 4090 上 3.5k 仿真步/秒，500Hz 下约 7× 实时；可复现于 Isaac Sim。
- **主结果（§VI-B，表 I）：** Google Robot 三任务平均 MMRV：验证集 action MSE 0.375，VarAgg 0.143，VisMatch 0.056；Pearson r：MSE 0.308，VarAgg 0.778，VisMatch 0.924。即 open-loop 验证 MSE 不是真机表现的好代理。
- **分布偏移（§VI-C）：** 沿背景/光照/干扰物/桌面纹理/相机位姿五轴扫描，sim 复现真机上「相机位姿与桌面纹理影响大、光照与干扰物影响小」，也复现数据增强的效果；Octo-Base 对机械臂纹理更敏感的 sim 预测被真机（包礼品纸改变臂纹理）验证。
- **消融（§VI-D，表 II–III、X）：** 系统辨识扰动后 MMRV 变差；绿幕+物体+机器人纹理需**联合**使用才有收益，只改其中一部分无改善；物体质量/摩擦在合理范围内变化时结论稳健（成功率变动 ≤15%）；SAPIEN 与 Isaac Sim 均可得到强相关（MMRV 0.082 / 0.058）。
- **附录补充：** Kruskal-Wallis 检验用于衡量绝对成绩分布是否与真机显著不同，与 MMRV（相对排序）互补；需启用 Temporal Gauss-Seidel 求解器避免 Google Robot 抓取穿模。
- **局限（§VII）：** 仅限刚体操作；green-screen 仅适用于固定相机且不能处理阴影；环境构建仍需人工整理资产与拼场景（关节物体耗时最多），自动化大规模建环境留作未来工作。

## 对 wiki 的映射

- [`wiki/concepts/simulation-evaluation-infrastructure.md`](../../wiki/concepts/simulation-evaluation-infrastructure.md)
- 既有入库归档（摘录式，非全文通读）：[simplerenv_arxiv_2405_05941](simplerenv_arxiv_2405_05941.md)

## 参考来源

- 论文：<https://arxiv.org/abs/2405.05941>
