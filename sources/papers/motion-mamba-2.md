# Motion Mamba 2.0: Autoregressive Diffusion Mamba for Long-Horizon Humanoid Control｜长程人形控制的自回归扩散 Mamba

- **类型：** paper
- **作者：** Yue Huang、Tianshan Zhang、Zeyu Zhang、Jayeon Kang、Akide Liu、Sungjoon Choi、Dacheng Tao、Hao Tang
- **机构：** 北京大学、高丽大学、蒙纳士大学、南洋理工大学
- **版本：** arXiv:2611.29897v1（2026-11-23）
- **原文：** <https://arxiv.org/html/2611.29897v1> · [PDF](https://arxiv.org/pdf/2611.29897)
- **项目页：** [官方项目](../sites/motion-mamba-2.md)
- **代码：** [官方代码](../repos/motion-mamba-2.md)
- **核对范围：** 2026-11-25 查阅 HTML 原文关键机制、结果及讨论摘录，项目主页和 README；全文与必要附录尚未逐段核完。
- **通读：** 未完成

## 核心摘录（整理）

- 每个窗口为 52 帧、25 FPS；文本、空间目标和已接受历史共同约束根轨迹去噪，根轨迹局部特征再条件化身体去噪。局部注意力读取近期特征，Mamba 循环状态跨窗恢复，PCM 保存可供注意力检索的摘要。去噪过程中历史保持固定，仅接受干净窗口后提交历史状态。生成参考运动与 SONIC 跟踪控制是不同层次。
- 论文 §V–VII：HiPHI 988 帧（39.52 s）条件下 Root CE 7.55 cm、End CE 13.64 cm；RTX 4090 的 52 帧完整窗口 p50 由 1685.35 ms 降至 276.14 ms，配对中位加速 6.138 倍。此为稳态窗口生成而非端到端响应延迟。G1 真机 100 次试验覆盖四种任务；用户提供的 94/100 与无摔倒统计仍待逐项核对表格。
- 论文明确暂未覆盖物体操作；生成器不等价于低层运控。项目页列有代码、模型和数据；外部 SONIC、LLM2Vec 及资产仍受各自许可和安装条件约束。

对 wiki 的映射：[长程人形控制的自回归扩散 Mamba](../../wiki/entities/paper-motion-mamba-2.md)。
