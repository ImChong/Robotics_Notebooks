# When Is a World Model Worth Consulting? Task, Capacity, and Horizon in Simulated Control

> 来源归档（ingest）

- **类型：** paper
- **论文：** <https://github.com/VasPug/robot-world-model-study-/blob/main/paper.pdf>
- **项目 / 代码：** <https://github.com/VasPug/robot-world-model-study->
- **作者：** Aravind Kanappan、Vasanth Pugalenthi
- **入库日期：** 2026-10-12
- **一句话说明：** 在模拟导航与 Push-T 中对照策略直执行、预测未来后选择和直接候选评分，并分析任务、容量、前视长度与推理代价。
- **沉淀到 wiki：** [实体页](../../wiki/entities/paper-world-model-consultation-study.md)

## 摘要级结论

预测式选择在一些结构化导航条件中有成功率点估计增益，冻结的 episode-level gate 改善成功率—成本折衷。Push-T 主设定约 1M 参数、32 步视野，predictor 与 scorer 成功率依次为 easy alignment 43.3%/55.3%、recovery 30.0%/42.7%、friction shift 33.7%/37.0%、damping shift 0.7%/0.3%。但 36 项模型—评分器对比的 95% 区间下界均未严格大于零；damping 条件近零成功，结论不确定。

实用读法：世界模型必须与同候选动作的直接评分器比较，并一并考察任务收益、状态条件和延迟。增大容量或 horizon 未呈现稳定的普遍收益。

## 评测范围与限制

实验均为模拟控制。导航包括开阔、绕行、钥匙/门和未见控制条件；Push-T 是冻结策略与候选生成方案下的扩展实验。逐项区间未做多重比较校正。延迟独立测量并以串行估算 episode cost，不能当作普适部署常数。

## 代码与复现

仓库提供 Python 3.11 + uv 锁定环境、CPU 导航与 Push-T 入口、GPU 扩展脚本和参考结果校验。GPU 扩展需要 CUDA；CPU 校验不等于重跑整个扩展实验。细节见[仓库归档](../repos/robot-world-model-study.md)。

## 对 wiki 的映射

- [实体页](../../wiki/entities/paper-world-model-consultation-study.md)
- [Model-Based RL](../../wiki/methods/model-based-rl.md)
- [World Action Models](../../wiki/concepts/world-action-models.md)
