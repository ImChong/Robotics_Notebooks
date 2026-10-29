# VIStream（视觉-惯性流式几何基础模型：核心源码 + 研究网站）

> 来源归档（RP1 总览 README「Related papers」中的第六项；仓库不属于 Roboparty 组织）

- **标题：** VIStream: Bringing Inertial Sensing to Streaming Geometric Foundation Models
- **类型：** repo + 项目页 + 论文 PDF
- **作者归属：** RP1 README 署名 Mingzhe Liu、投稿场合写 ICRA；项目页的论文 PDF 为匿名稿（“Anonymous Authors”），**未在仓库中核到 arXiv 编号**
- **项目页：** https://cn-mingzheliu.github.io/vistream/
- **代码：** https://github.com/CN-MingZheLiu/vistream（RP1 README 链接；最近提交 2026-10-14 `Link the core code release from the website`）
- **核心源码发布：** https://github.com/CN-MingZheLiu/vistream/releases/tag/core-code-20261014（`core/` 目录为同一发布的源码）
- **入库日期：** 2026-10-27
- **许可：** 仓库根目录未见 LICENSE 文件；`core/licenses/README.md` 声明保留各第三方组件的原许可，不授予统一许可；模型权重另行获取
- **一句话说明：** 在冻结的 LingBot-Map 流式视觉几何模型上，接入已有视觉-惯性估计器（实现中为 ORB-SLAM3）的标定运动，输出带持久度量尺度的相机轨迹；无需重训任何上游模型。
- **沉淀到 wiki：** [`wiki/entities/vistream.md`](../../wiki/entities/vistream.md)

---

## 开源状态核查（2026-10-27）

- **部分开源（仅源码）**：`core/` 含 VIStream Python 包（`vistream/`：`checkpoint.py`、`config.py`、`losses.py`、`metrics.py`）、固定实验配置（`configs/*.json|yaml`）、推理/融合/评测脚本、冻结视觉骨干 `lingbot-map` 源码、DPVO / DROID-SLAM / PyPose 源码、ORB-SLAM3 + OpenCV 4.5.5 + Pangolin 的构建源压缩包。
- **不含**：模型权重、数据集、预测轨迹、编译产物与 6.54 GB 离线环境；发布说明明确“源码发布不是自包含推理安装”。
- **复现边界**：此前完整运行包在 KITTI 00–10 上完成 132 个阶段，平均 Sim(3) ATE 为 LingBot-MAP 32.49 m、VIStream 回环关 15.25 m / 开 9.10 m，**未能精确复现**更早修复参考值 15.08 / 10.41 m（`verification: NOT_REPRODUCED`）；该配置不同于论文原始队列。本次发布未做新的 GPU 验证。
- **Web 演示**：回放预录轨迹，不运行模型推理。

## 对 wiki 的映射

- [vistream](../../wiki/entities/vistream.md)
- 上游骨干：[paper-lingbot-map](../../wiki/entities/paper-lingbot-map.md)
- 总览：[roboparty-rp1](../../wiki/entities/roboparty-rp1.md)、[roboparty_rp1.md](roboparty_rp1.md)
