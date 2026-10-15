# When Do Discrete Actions Make Good Continuous Controllers?（arXiv:2610.20429）

> 来源归档（ingest）

- **标题：** When Do Discrete Actions Make Good Continuous Controllers?
- **缩写 / 框架：** 比较 CR / CF / DR / DF 四类动作头；离散流（DF，discrete flow）与无分类器引导（CFG）为核心修正手段
- **类型：** paper / action representation / VLA / discrete tokens / flow matching / empirical study
- **arXiv：** <https://arxiv.org/abs/2610.20429>（提交日期 2026-10-14；HTML：<https://arxiv.org/html/2610.20429v1>）
- **项目页：** <https://iscoyizj.github.io/discrete-action-website/>（BibTeX 注明 2026；页面 Code 按钮为 “Code (soon)”）
- **代码：** 未发布。项目页与 BibTeX 旁注写明 “Code and model checkpoints will be released upon publication.”
- **作者：** Zeji Yi、Soham Dasgupta、Xinyang Zhang、Koushil Sreenath、Guannan Qu、Xi Chen、Max Simchowitz、Guanya Shi
- **机构（据项目页）：** Carnegie Mellon University；Amazon FAR（Frontier AI & Robotics）；部分作者为 Amazon FAR 实习期间完成
- **入库日期：** 2026-10-15
- **一句话说明：** 系统比较连续与离散动作参数化在单任务从零训练与多任务 VLA 训练中的表现，发现最朴素的均匀分箱量化也能匹配或超过连续策略；离散策略的瓶颈不在动作精度，而在跨时间与动作维度的因子化误差（factorization error）。

## 开源状态（步骤 2.5）

- **项目页核查（2026-10-15）：** Code 按钮为 “Code (soon)”；页面无 GitHub、Hugging Face 等链接。
- **论文 / 页面声明：** “Code and model checkpoints will be released upon publication.”
- **结论：** **待发布**（宣称将开源，尚无 URL）。

## 摘录 1：问题与主张（Abstract，§1）

- **问题：** 机器人基础模型普遍用连续向量表示动作，离散 token 是否只能作为语言模型的接口，还是能与连续动作竞争。
- **主张：** 在多种设定下比较 single-step 回归与多步 flow 的连续与离散变体；离散动作预测即使是 uniform-bin 量化也能匹配甚至超过连续策略。
- **关键发现：** (1) 离散策略的精度并非瓶颈，它们对动作块中的时间变化与观测变化更敏感（jerk ratio、state responsiveness）；(2) 瓶颈是 token 的乘积分布造成的因子化误差；(3) 迭代离散流或 CFG 可以缓解；(4) 显式注入几何信息或结构化 tokenizer 没有稳定收益。

**对 wiki 的映射：** 升格为 [`wiki/entities/paper-discrete-actions-continuous-control.md`](../../wiki/entities/paper-discrete-actions-continuous-control.md)；与 VLA、action chunking、扩散策略、flow matching 页面互链。

## 摘录 2：方法与设定（§2、§4、附录 B–C）

- **动作表示：** 末端位姿，动作块长度 H，每步 M 维；离散化为 B = 512 个 bin，展平为 L = H·M 个 token；解码取 bin center。
- **四种头：** CR（回归）、CF（连续 flow matching）、DR（逐位置交叉熵，单次前向）、DF（离散流，多步迭代，每步按概率替换 token）。
- **CFG：** 训练时随机替换条件为空条件，推理时线性组合有条件与无条件的 logits，改变 token 偏好。
- **评测环境：** RoboMimic（单任务从零）；VLA 规模的 RoboCasa、LIBERO、DROID（离线诊断）；真机多任务（论文称三项真机任务，Franka 臂相对末端命令）。

## 摘录 3：实验要点（§3–§5，Fig. 1–4，Table 1）

| 发现 | 论文数字 / 要点 |
|------|-----------------|
| 离散头在 RoboCasa 上优于连续头 | 论文称 DR 在 RoboCasa 上 27.1%（无 CFG），CFG 后 49.2%；DF 53.9%、CF 52.7%、CR 38.8% |
| 因子化误差 | 对动作块的时间与维度做乘积分布假设会产生不一致组合；DF 与 CFG 分别通过迭代与改变 token 偏好降低误差 |
| 几何与结构化 tokenizer | 显式几何（metric-path DF）为 52.8%，对比 mixture-path DF 的 53.9%；替代 tokenizer 对部分配置有小幅提升，但未超过最佳的均匀分箱 DF 结果 |
| 真机 | 论文图 1(b) 给出三项真机 VLA 任务结果；本次未从文本中提取逐项数字，记为未核实 |

**对 wiki 的映射：** 写入 `## 实验与评测`；论文图表中未逐项核对的数字不写入正文。

## 对 wiki 的建议动作

- 新增 [`wiki/entities/paper-discrete-actions-continuous-control.md`](../../wiki/entities/paper-discrete-actions-continuous-control.md)，开源状态写「待发布」，不写 `code:` 字段。

## 相关链接

- arXiv：<https://arxiv.org/abs/2610.20429>
- 项目页：<https://iscoyizj.github.io/discrete-action-website/>
- 文章摘录来源：[`sources/raw/wechat_embodied_station_robot_31_papers_2026-10-15.md`](../raw/wechat_embodied_station_robot_31_papers_2026-10-15.md)
