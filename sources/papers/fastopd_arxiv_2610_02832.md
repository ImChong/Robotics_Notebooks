# FastOPD: On-Policy Distillation for Lightweight VLA Deployment

> 来源归档（paper；核对 arXiv HTML v1 与官方项目页；2026-10-06）

- **作者：** Yoojin Oh, Jeongsol Kim, Yeonwoo Seo, Jangho Park, Seonghyun Jin, Sunwoo Park, Youngmin Kim, Youngjun Jun, Kyumin Choi, Jong Chul Ye
- **机构：** KAIST；Sungkyunkwan University
- **论文：** <https://arxiv.org/abs/2610.02832> · [HTML v1](https://arxiv.org/html/2610.02832v1)
- **项目页：** <https://fastopd.github.io/>
- **代码：** 截至 2026-10-06，官方项目页未列 GitHub / 下载仓库，代码待发布；不以论文复现实验描述推断源码已开放。
- **一句话说明：** FastOPD 通过单个学生 on-policy 状态上的教师速度匹配与有限区间自一致性，把大型流策略蒸馏为更小、少步推理的 VLA。

## 核心摘录

1. **效率瓶颈。** 传统 on-policy distillation 沿学生去噪轨迹多次查询教师，教师模型大、推理步多，训练成本高。
2. **方法。** 学生 flow map 从噪声一步跳到采样状态，只在该状态匹配教师速度（OPFD）；再以 midpoint self-consistency 约束一段直接跳转与两个短跳转一致，把局部教师监督延拓到任意区间的少步生成。冻结 VLM 主干，只微调轻量 action expert 和时间投影层。
3. **LIBERO。** 采用约 451M 参数学生，2 次采样平均成功率 81.8%，保留 π0.5 教师结果的 84%，相较 π0.5 推理延迟降低 78.1%（301→66 ms，单 RTX 3090）。
4. **RoboTwin 2.0。** LingBot-VLA 教师时，单步平均成功率从 base SmolVLA 的 35.3% 提至 51.2%；作者报告常规 OPD 达到相近 1-step 成功率耗时约 5.7 倍。
5. **真机。** MolmoAct2（5B）蒸馏为 451M 学生，在 YAM 机器人 pnp-plate 任务以 4 步获得 50% 成功率，base SmolVLA 同步数为 42%；相较 10 步基线完成时间从 19.32s 降至 17.38s。
6. **可复现性。** 项目页列出论文、方法图、benchmark 结果与真机演示，但目前没有代码 URL；部署数据和模型权重也未见公开下载入口，代码待发布。

## 对 Wiki 的映射

- [FastOPD 论文实体](../../wiki/entities/paper-fastopd.md)
- [VLA 方法页](../../wiki/methods/vla.md)
- 与 flow/diffusion 少步推理、策略蒸馏的关系见本页「与其他工作对比」。

## 参考来源（原始）

- arXiv HTML v1：<https://arxiv.org/html/2610.02832v1>
- 官方项目页：<https://fastopd.github.io/>
