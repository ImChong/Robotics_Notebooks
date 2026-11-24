# NoPeWorld：非穿透世界模型（Non-penetrating World Models） — sites 来源归档

- **类型：** site
- **标题：** Non-penetrating World Models
- **机构：** 香港理工大学 Learning & Generation Lab
- **原文：** https://nopeworld.github.io/
- **arXiv：** 2611.27963
- **发表日期：** 2026-11-20
- **项目主页：** https://nopeworld.github.io/
- **代码：** https://github.com/yuhangdai427/NoPeWorld
- **开放状态：** Apache-2.0；2026-11-21 发布代码、六种生成器的预测头与基准数据；生成器本体须另行获取。
- **入库日期：** 2026-11-24
- **核对范围：** 已打开项目主页；核查公开资源与结果，尚未完成所有关联原文的阅读。
- **通读：** 未完成
- **对应知识页：** [nopeworld](../../wiki/entities/paper-nopeworld.md)

## 资料要点

- 训练阶段以 Unreal 场景真实几何标注候选运动安全性，冻结生成器，仅学习不足 200 万参数的头。每个 chunk 评估 34 个候选，包含原动作、停止、缩短步幅与转向。预测模式读取上一 chunk 的状态；碰撞模式读当前 chunk 首次去噪状态。保留安全原动作，否则选择尽量保持意图的候选，再闭环生成视频。
- NoPeBench Task 1 在真实历史上测试决策，Task 2 在已知几何的闭环生成中评估，Task 3 用开放域静态及移动障碍图像测试零样本迁移。作者报告六模型 Task 2 碰撞模式违规率平均 74%→16%；LingBot-World 2.0 89%→7%。预测模式不是所有模型均支持，Evoke-Turbo 无跨 chunk 记忆。
- 视频不穿透不等于机器人安全；几何评估依赖重建和分割，开放域依赖 VLM 判断；预测头并未提供形式化安全保证。安装文档要求至少 80 GB 显存，部分模型建议 96 GB。

## 公开资源

- [论文 HTML](https://arxiv.org/html/2611.27963)
- [arXiv](https://arxiv.org/abs/2611.27963)
- [官方项目页](https://nopeworld.github.io/)
- [官方源码](https://github.com/yuhangdai427/NoPeWorld)
- [预测头权重](https://huggingface.co/PeterDAI/NoPeWorld)
- [NoPeBench 数据](https://huggingface.co/datasets/PeterDAI/NoPeBench)

## 对 wiki 的映射

- [nopeworld 独立实体](../../wiki/entities/paper-nopeworld.md)
- [关联归档](../papers/nopeworld_arxiv_2611_27963.md)
- [关联归档](../repos/nopeworld.md)
