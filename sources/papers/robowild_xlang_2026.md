# RoboWild: Can Frontier Models Generalize to Robot Use in the Wild?（论文原文核查）

- **类型：** paper
- **英文标题：** RoboWild: Can Frontier Models Generalize to Robot Use in the Wild?
- **中文标题：** RoboWild：前沿视觉语言模型能否泛化到复杂真实感场景的机器人操作？
- **作者：** XLANG Lab（详见官方仓库 README 中 BibTeX 的完整作者名单）
- **年份：** 2026
- **官方 PDF：** https://huggingface.co/datasets/xlangai/RoboWild/resolve/main/RoboWild.pdf
- **版本：** 2026-11-20（论文首页日期），51 页；官方未提供 arXiv ID
- **机构：** 香港大学 XLANG Lab；Tao Yu 为通讯作者（附录 A）
- **项目主页：** https://robowild.xlang.ai/
- **源码：** https://github.com/xlang-ai/RoboWild
- **Harness：** https://github.com/xlang-ai/RoboWild-Harness
- **数据：** https://huggingface.co/datasets/xlangai/RoboWild
- **收录及核对日期：** 2026-11-22
- **核对范围：** 已通过 GitHub Actions 临时只读核查任务从官方直链取得原始 PDF，使用 pypdf 逐页核对全部 51 页正文、表格、公式、图注和附录 A–G；联合核对官网、官方 README、评测文档和相关源码。图片像素细节未逐图目视验证，所引图例仅采用原文图注与旁述。
- **通读：** 已完成

## 官方已公开且可验证的关键信息

1. **任务规模与复杂性：** 官网 38 个仿真场景、4,480 件功能/物理可交互物体、341 个任务–本体评测；README 进一步拆为 239 项唯一 Dexmate 任务设置和 34×3 项跨本体复用设置。涉及导航、抓取、功能物体操作、双手协作等。
2. **模型表现：** 项目官网报告 Claude Fable 5.2 weighted success 13.28%（榜单四舍五入 13.3%），其余在官网同协议榜单依次 GPT-6 Astra 11.6%、Gemini 4 Carbon 10.6%、Claude Opus 5.5 6.5%、GPT-6.1 Sol 4.6%；官网标题为 ReAct at highest thinking level。
3. **失败归因：** 官方 Results Analysis 标记 188 个失败运行样本，Grasp Failure 42.6%、Motion/Execution 28.7%、Functional Object Operation 16.0%、Goal/Target Selection Error 6.9%、Other 5.9%。
4. **恢复与安全：** 局部错误恢复率 Interface 78.7%、Re-anchor 45.5%、Contact 25.7%；无关物体扰动 87%，碰撞停止 78%，目标物品滑落/倾倒 75%，均来自官网可见结果页。
5. **协议边界：** GitHub README 指明默认 RGB-only ReAct + react_move、冻结仿真、实体动作工具，调用预算 Short 100 / Long 200。成绩不代表真实家庭部署或通用 VLA 本体控制能力。

## 已核论文正文与补充实验

### 任务规模、定义和评价（§2；附录 B/C/E.5）

- 239 个独立任务设置（127 Short、112 Long），205 个 Dexmate-only；其余 34 个设置在 Dexmate、Mobile ALOHA、Franka Mobile、Unitree G1 各测一次。共 205 + 34×4 = **341 个 setup–embodiment 实例**，不是 341 个彼此不同的指令。
- 每个 setup 同权：34 个跨本体设置先平均四个二值成功指标，再和 205 个 Dexmate-only 设置一起平均，**分母为 239**（§3.1、附录 E.5）。
- 任务标签：视觉目标消歧 58.6%、视觉推理 14.6%、OCR 13.4%、关节/材料交互 53.6%、精细操作 48.5%、导航 28.5%；85.4% 含多个完成里程碑、65.7% 存在顺序依赖（§2.2）。
- 场景覆盖 23 种空间类型、平均物体密度 3.60 个/平方米、24 类功能交互及 88 类任务操作模式（§2.5、Table 1），指标遵循作者在附录 D 所定统计标准。
- 任务评估读取物体位置、关节、接触、粒子与行为状态，对过程约束以逐步 `observe()` 追踪事件并在 `evaluate()` 结合最终状态判断。烧烤夹翻牛排必须验证“用夹具而非直接用手夹”的过程；人工标注的成功/失败/边界样例由独立验证器检查，历史事件用 live run 核验，**论文不报告一个所有评价器的汇总准确率**（§2.3，附录 C.1）。
- 部分“真实感”交互是脚本状态机和视觉近似（如点火、烹饪），不等同真实热传递、燃烧或完整粘附力学；物体碰撞与关节参数在 Isaac Sim 中配置（附录 B.2–B.3）。

### 主榜及资源成本（§3.2，Table 2，p.7）

| 模型（最高测试 reasoning effort） | Setup 加权成功率 | Short | Long | 平均 Robot calls | 输出 tokens/任务 | 估计 USD/任务 |
|---|---:|---:|---:|---:|---:|---:|
| GPT-6.1 Sol | 4.60% | 6.69% | 2.23% | 133.9 | 183K | 6.3 |
| Claude Opus 5.5 | 6.49% | 8.27% | 4.46% | 129.8 | 1.37M | 42.9 |
| Gemini 4 Carbon | 10.56% | 12.99% | 7.81% | 127.1 | 634K | 35.3 |
| GPT-6 Astra | 11.61% | 15.16% | 7.59% | 124.1 | 111K | 29.3 |
| Claude Fable 5.2 | **13.28%** | 14.37% | 12.05% | 127.3 | 2.26M | 125.34 |

同一模型 reasoning effort 的收益和成本差异显著：Fable low/high/max 成功率 2.7/8.9/13.3%，对应 131K/470K/2.26M 输出 tokens/任务（附录 F.4，Table 20）。报告的平均任务时间是**根据 output tokens、第三方参考推理速度和固定 0.27 小时额外开销推算**，不是全部直接测得的端到端真机时延：Fable max 6.03h、Astra max 1.04h（§4.4，Table 21）。

### 跨本体和 Harness 消融（§4.1–4.2，pp.8–9）

在共享的 34 个任务设置上，论文明确写明 **5 个模型在 Unitree G1 上均无任务成功**。G1 采用 ScaleBFM 全身跟踪控制器；作者指出上层动作工具与低层跟踪行为的匹配问题。此结论只对应论文中的 G1 + ScaleBFM + Harness 设置，并非普遍性的真机能力判断。

同为 Claude Opus 5.5 max，控制 Dexmate 的 **239 个设置**作四种 Harness 消融（Table 3；与主榜混合本体 6.49% 的加权口径不同）：

| Harness | 成功率 | 平均 Calls | 输出 tokens/任务 | 估计 USD/任务 |
|---|---:|---:|---:|---:|
| ReAct RGB | 6.28% | 131.3 | 1.37M | 43.0 |
| ReAct + 深度 | 6.69% | 136.0 | 740K | 25.4 |
| Claude Code RGB | 12.13% | 97.9 | 730K | 57.1 |
| Claude Code + 深度 | **14.64%** | 110.6 | 610K | 50.3 |

ReAct 加深度仅 +0.41 个百分点，Claude Code 模式和深度可进一步提升，但 Coding Agent 更换了软件接口、动作规划和代码执行能力，不能只归因于“开放代码功能”。深度感知使用带噪声、边缘 dropout 和外参扰动的 processed preset，而非完美真值（附录 E.5）。`move_robot` 默认仅检验关节限位和自碰撞，环境避障依靠代理规划；Coding 模式可结合深度体素障碍世界，但未观测区域按空闲处理（附录 E.2–E.3）。

### 失败、局部恢复与安全（§5；附录 F.2–F.3）

- 从 4 个模型（Opus、Sol、Astra、Carbon；**不包括 Fable**）对同一批 50 个任务各试一次，共 200 轨迹，其中 12 成功、188 失败；样本任务分布 34 Dexmate、4 ALOHA、4 Franka、8 G1。以“首个未解决且阻碍完成的问题”给每条失败轨迹归一个主因。
- 失败数量依次为：抓取 **80/188（42.6%）**；运动/执行 **54（28.7%）**；功能操作 **30（16.0%）**；目标选择 **13（6.9%）**；其他 **11（5.9%）**。G1 的 32 条失败中 15 条主因是运动/执行，11 条抓取，6 条功能物件（附录 F.2）。
- 200 条轨迹上 4,710 个恢复 episode，均值每轨迹 23.6。Interface 每轨迹 12.1 次、局部修复率 **78.7%**；Re-anchor 3.7 次、**45.5%**；Contact 7.7 次、**25.7%**。接口参数修复的 pooled 局部率 96.8%，同策略重抓 32.6%，精细位姿调整 27.0%；**局部修复不能当作最终任务成功**（§5.2、附录 F.3）。
- **78%** 轨迹触发固定障碍碰撞安全停止（合计 1,527 次），**87%** 轨迹扰动非目标物体（合计 679 起事件）；48% 轨迹让非目标物体翻倒、37% 使其跌落。任务完成的 evaluator 不总是惩罚无关物体的扰动（§5.3），所以“成功”不等于“安全”。
- 代表案例：附录 G.1（pp.48–49）眼镜折叠成功 75/100 calls、先将水瓶移至开放台面后双手旋盖成功 49/100；附录 G.2（pp.50–51）狗粮料盒被误判为空却继续倒料，以及胡椒罐未释放连锁碰倒炉旁盐罐并漏关空气炸锅。

### 论文边界（§7、附录 F.1）

仿真与实机的传感、物理机制、执行偏差仍可能改变结果。任务通常按每设置每模型/推理级别一次尝试，不能据此推断稳定的真实世界成功概率或给出未统计的不确定性。能力标签与成功率的相关性并未隔离标签共现的混杂因素；模型、Harness 和动作规划器对结果均有影响。

## 对 wiki 的映射

- [论文详情](../../wiki/entities/paper-robowild.md)
- [官方主页核对](../sites/robowild-xlang-ai.md)
- [官方源码核对](../repos/robowild.md)
