# Knowledge Insulating Vision-Language-Action Models（arXiv:2505.23705）

> 来源归档（ingest）

- **标题：** Knowledge Insulating Vision-Language-Action Models: Train Fast, Run Fast, Generalize Better
- **短名：** Knowledge Insulation / KI
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2505.23705>
- **项目页：** <https://www.pi.website/research/knowledge_insulation>
- **机构：** 物理智能（Physical Intelligence）
- **入库日期：** 2026-09-28
- **一句话说明：** 用 FAST 离散动作训练 VLM 骨干，同时让 flow 动作专家出连续动作且梯度不回传骨干。

## 开源状态（步骤 2.5，2026-09-28）

- **部分开源**： [openpi](https://github.com/Physical-Intelligence/openpi) README 写明 π₀.₅ 用 knowledge insulation 预训练，并发布 `pi05_libero` / `pi05_droid` 等权重；同一 README 注明仓库里的 π₀.₅ 训练与推理目前只支持 flow matching 头。
- 维护者在 [openpi#649](https://github.com/Physical-Intelligence/openpi/issues/649) 说明：预训练 π₀.₅ 检查点用了 KI；`train.py` 微调只对 action expert 的 flow matching 损失反传，梯度会进入 VLM 骨干，**未实现** FAST token 与 stop-gradient 的 KI 微调。

## 核心摘录（面向 wiki 编译）

- 直接把连续动作专家的梯度送进预训练 VLM，会拖慢学习并损伤语言跟随。只冻结骨干又让表征不适应控制。
- KI：骨干用 π₀-FAST token（及网页 VLM / 高层规划数据）学习；动作专家用 flow matching 出连续动作，梯度停在专家。推理丢掉离散 token。
- 博客称通才 bussing 上达到与 π₀-FAST 相近的训练步数，约为 π₀ 的 1/7.5，推理速度仍是动作专家；单本体 bussing 上自回归 π₀-FAST 耗时约为一倍。
- **对 wiki 的映射：** [paper-knowledge-insulation](../../wiki/entities/paper-knowledge-insulation.md)
