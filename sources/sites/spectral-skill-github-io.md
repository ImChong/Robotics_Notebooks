# Spectral Skills — 官方项目主页

- **类别：** 官方研究项目站点 / Spectral Skills
- **项目地址：** https://spectral-skill.github.io/
- **论文：** https://arxiv.org/abs/2609.37677
- **论文 HTML 全文：** https://arxiv.org/html/2609.37677
- **论文归档：** [spectral_skills_arxiv_2609_37677.md](../papers/spectral_skills_arxiv_2609_37677.md)
- **知识节点：** [Spectral Skills](../../wiki/entities/paper-spectral-skills-motion-representation.md)
- **机构：** Georgia Institute of Technology（佐治亚理工学院）
- **项目状态：** ICLR 2027 under review（据项目主页；非录用确认）
- **核验日期：** 2026-11-08
- **核对范围：** 项目首页、Method、Tracking、Skill space、Composition、Planning、Ablations、Try it、G1 真实视频及公开代码入口；与 arXiv v1 的 §3–5、附录核对。
- **通读：** 已完成

## 站点一手证据

- **方法：** 64 维连续 latent z 表示 0.2 秒动作片段，通过预测后续运动学习；因式化扩散模型约束 z 的仿射进入方式，平均响应 Gram 矩阵分解出正交谱方向。
- **单控制器：** 冻结的 50 Hz PPO 全身控制器接收编码参考、语言 planner 生成的技能或基础技能＋谱方向，用于跟踪／拼接／组合。
- **实验：** 与 SONIC v1.1 的 4096 条动作仿真对照，MPJPE-G 187.86→70.51 mm；SR 98.88%→98.27%；真机页面提供 G1 各项操作演示，但不公布等价的真机 MPJPE 指标。
- **浏览器演示：** 项目页“Try it”通过 ONNX Runtime Web 和 MuJoCo WebAssembly 交互运行；官方表述是演示而非 benchmark。

## 代码、数据、模型开放核查

- **官方训练/评测代码：** 截至 2026-11-08 **未见公开仓库 URL**。项目页标题区“Code”未提供可访问的官方代码库链接；论文可复现性声明计划在发表时开放。标为 **待发布 / 尚无法按公开源码复现完整训练**。
- **可试用模型：** 官方浏览器交互示例可运行部署策略；无需等同为公开的训练代码仓库、权重完整下载或可审计的部署源码。
- **来源数据：** 论文报告 129,785 条重定向动作；未在项目页发现可直接获取该具体训练集合的独立发布入口。
- **源码运行时序图：** 不适用（未核实到官方训练／推理仓库与实际调用入口）；不凭浏览器 Demo 构造虚假的源码时序。

## 参考来源

- [Spectral Skills 官方主页](https://spectral-skill.github.io/)
- [arXiv:2609.37677v1 正文及附录](https://arxiv.org/html/2609.37677)
