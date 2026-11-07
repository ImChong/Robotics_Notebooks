# XMimic: Learning Whole-Body Motion Tracking in Under a Minute

- **类型：** paper
- **作者：** Nico Bohlinger、Jan Peters
- **链接：** <https://github.com/nico-bohlinger/papers/blob/main/2026_xmimic.pdf>
- **版本：** 作者仓库 2026 PDF，2026-11-07 获取；全文 SHA256 见下。
- **代码：** <https://github.com/nico-bohlinger/xmimic>
- **核对范围：** 全文三页：摘要、I Introduction、II A/B 方法、III Experiments、IV Conclusion、致谢与参考文献；结合 README 详细实验边界核对。
- **通读：** 已完成
- **入库日期：** 2026-11-07

## 核心摘录

1. JAX 编译整轮训练、单卡合并 50 轮；重排想象轨迹梯度减少小 GPU 操作。对 wiki 的映射：[核心原理](../../wiki/entities/paper-xmimic-jax.md)。
2. 多卡 actor、世界模型及 critic、策略、失败搜索并行，以陈旧参数换取较短墙钟时间。对 wiki 的映射：[源码时序](../../wiki/entities/paper-xmimic-jax.md)。
3. H100 单卡 2.48 分钟，四卡 0.91 分钟；严格目标连续三点评测达标且无失败，排除编译；没有直接报告 85× SuperMimic 实验。对 wiki 的映射：[评测与结论](../../wiki/entities/paper-xmimic-jax.md)。

## 开放与局限

[源码归档](../repos/xmimic-jax.md)：MIT 代码、公开结果 CSV，数据另下；LAFAN1 衍生物非商业。纯仿真实验，未证明真机与大规模动作泛化。

PDF SHA256：`b94338a254177828ece2f0bfd9eec06e2c6ac41df8ad53b486de13a8eb66b1c3`。
