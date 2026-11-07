# VLA-Tug：校准潜变量探针的 VLA 故障预测与分类

- **标题：** Failure Prediction and Classification in Vision-Language-Action Models Via Calibrated Latent Probing
- **论文：** <https://arxiv.org/abs/2611.08425>
- **全文版本：** <https://arxiv.org/html/2611.08425v1>，2026-11-05 v1。
- **作者：** Michal P. Podolinsky、Yunhao Yang、Daniel Little、Hong Lu、Rohan Siva、Olin Goudey、Neel P. Bhatt、Matthias Scheutz、Ufuk Topcu。
- **机构：** 得克萨斯大学奥斯汀分校、塔夫茨大学、得克萨斯大学达拉斯分校。
- **项目页：** <https://vla-tug.github.io/>
- **代码：** <https://anonymous.4open.science/r/VLA-Tug/>
- **通读：** 已完成
- **核对范围：** 2026-11-07 通读 v1 HTML 正文 §1–7、附录 A–E（探针/标签/校准、数据、基线/指标、完整结果/消融、定性干预）及表 1–11 的文本；图以图注与正文核对，不从曲线估读额外数值。另读官方项目页与匿名源码 README、数据配置及核心训练/评测模块。

## 核心摘录

1. §4 / 附录 A：对 agent、wrist、language token 分别均值池化并拼接为 6144 维；单层 256 隐元 LSTM 接评分、10 类故障模式和 64 维 CLIP 语义头。失败轨迹在 onset 及之前标正，之后排除，成功轨迹全负；CLIP 只作训练监督，推理不用 CLIP 编码器。
2. §4.2 / A.3：LTT 同时约束 episode-level FPR 与 early miss rate；候选阈值无法认证时弃权。正式保证要求校准/测试可交换；主检测表用 max-conformal（仅 FPR），不能把 LTT 保证套给所有表。
3. §5 / B：公开仿真数据 3180 episodes（500 + 500 + 2180），含 28.8 万以上 embedding；真机另有 Kinova Gen3 八任务 399 episodes，未公开。标注由 Gemini 三轮流水线产生，故障 onset 是可观察错误或应接管时刻，不是物理损坏时刻。
4. §6 / C–D：early AUROC 是 FPR 0–0.4 上的面积除以 0.4；LIBERO-X unseen skill 为 39.0±7.0，SAFE-LSTM 30.1±8.5（相对约 +30%，不是 +30 个百分点）。unseen new-task 为 40.2 对 40.4，不能概括为所有任务领先。
5. 表 7：FPR 15% / early TPR 15% 的 LTT 只认证 7/14 runs；unseen FPR 达标 86%、early TPR 达标 71%，两个比例分别报告，不能写成联合达标 86%。FPR 10% 行无人被认证。形式保证不延伸至分布外任务。
6. 表 3 / D.3：4090 GPU 每步 0.12 ms，四 CPU 线程 0.94 ms，26.5 MB；是探针新增延迟，非整个 VLA 推理或机器人闭环。真机 unseen 面积 25.8±11.6，对 SAFE-LSTM 25.1±18.8，优势很小。
7. §7 / D.4 / E：骨干层选择未系统分析；辅助头对检测无一致收益；改变 onset 会影响排名。steering 是启发式演示，没有系统成功率评测，不能据演示宣称可靠自动纠错。

## 对 wiki 的映射

- [VLA-Tug 主节点](../../wiki/entities/paper-vla-tug.md)
- [项目页核查](../sites/vla-tug.md)
- [源码与数据核查](../repos/vla-tug.md)
