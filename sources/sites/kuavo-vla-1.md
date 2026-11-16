# KUAVO-VLA-1.0（乐聚官方项目页）

> 来源归档（site）

- **标题：** KUAVO-VLA-1.0
- **类型：** 项目页 / 垂域视觉-语言-动作模型
- **官方项目页：** <https://model.lejurobot.com/kuavo-vla-1/>
- **代码：** <https://github.com/LejuRobotics/LeTools-Learning>
- **技术报告（PDF，24 页）：** <https://model.lejurobot.com/kuavo-vla-1/kuavo-vla.pdf>
- **模型卡：** <https://huggingface.co/LejuRobotics/LET-KUAVO-VLA-1.0-models> · <https://www.modelscope.cn/models/lejurobot/LET-KUAVO-VLA-1.0-models>
- **数据集卡：** <https://huggingface.co/datasets/LejuRobotics/LET-KUAVO-VLA-1.0-Dataset> · <https://www.modelscope.cn/datasets/lejurobot/LET-KUAVO-VLA-1.0-Dataset>
- **社区入口：** <https://openlet.openatom.tech/tools/KUAVO-VLA-1.0>
- **来源核查日期：** 2026-11-14（初版归档 2026-10-03）
- **核对范围：** 官方项目页（`model.lejurobot.com/kuavo-vla-1/` 全文与逐任务结果）、官方技术报告 `kuavo-vla.pdf`（全文 24 页：Sec. 1–8、Appendix A–E、Tables 1–8）、Hugging Face 模型/数据卡页面及 HF Hub API 元数据（`LejuRobotics/LET-KUAVO-VLA-1.0-models`、`LejuRobotics/LET-KUAVO-VLA-1.0-Dataset`）
- **通读：** 已完成
- **一句话说明：** 乐聚智能（Leju Robotics / Lejoin Intelligence）基于 LingBot-VLA-2.0 在 Kuavo4pro 双臂人形机器人 680 小时 / 100 项工业真机数据上做两阶段垂域化训练的 DSFM；代码与工具链公开在 LeTools-Learning，技术报告与项目页已全文公开，模型权重与数据集在 HF/ModelScope/OpenLET 上采取审核制分发。

## 项目与数据状态（实测核查）

| 资源 | 当前可见状态 | 备注 |
|------|-------------|------|
| 官方项目页与技术报告 PDF | **已公开，免登录全文可读**（`kuavo-vla.pdf`，24 页） | 包含完整 25 任务列表、5 类技能拆分、Table 6 超参、5 类失败模式及 Appendix A–E 逐任务 SR/PS 与评分细则 |
| `LeTools-Learning` 代码 | **已公开（GPL-3.0）** | 含 Rosbag→LeRobot 数据转换、`kuavo_model/train.py` 训练编排、`kuavo_server` ZMQ adapter（含 `lingbot_vla_v2`）与 `kuavo_deploy` 评测部署脚本 |
| `LET-KUAVO-VLA-1.0-models` 权重 | **需申请审核**（HF API `gated: "auto"`、`usedStorage: ~12.76 GB`） | 公开文件树含 `LET-KUAVO-VLA-1.0-models/model/hf_ckpt/`（3 个 `.safetensors` 分片）、`LET-KUAVO-VLA-1.0.json` 与 `.yaml`；未登录直接读取 `README.md` 返回 HTTP 401，具体权重许可条款以获批后显示为准 |
| `LET-KUAVO-VLA-1.0-Dataset` 数据集 | **需申请审核**（HF API `gated: "auto"`、`usedStorage: ~1.39 TB`、9,650 个文件） | HF 元数据含 `license:apache-2.0`，采用 LeRobot 格式按任务分目录发布（`tasks/001-place-dumplings-into-tray` 至 `tasks/100-weigh-and-pack-calibration-weights`）；报告 Sec. 8 说明全套 100 任务按打包进度分批释出 |
| OpenLET / ModelScope | OpenLET 与 ModelScope 均挂出同名入口 | 与 HF 分发同一套权重和数据集文件，均需登录与社区接入流程（HF 模型卡另列出 QQ 群与 Discord 支持） |

## 官方项目页与技术报告核心事实（核对旧版截图口径）

2026-10-03 初版归档仅依据厂商发布截图，2026-11-14 直接通读官方项目页与 `kuavo-vla.pdf` 后确认以下精确口径与差异：

1. **语料规模与格式（Sec. 3）：** **100 项工业操作任务**、**58,780 条示教 episode**、**24,471,172 同步帧（约 24.5M）**，在 **10 Hz** 下共计 **680 小时**真机遥操作（旧截图写作笼统的「600+ 小时」）。任务分布为抓取放置 38、分拣 21、装配插接 16、包装 15、工具类 10；观测含 1 路头部 + 2 路手腕 RGB（`848×480`、10 Hz、AV1 编码），状态与动作共享 `16D` 交错向量 `[q1..7^L, g^L, q1..7^R, g^R]`（零填充至 `55D` 统一槽位以兼容 `which_arm` 与 `leju_claw` / `qiangnao` 末端）。经「采集 → 标注 → 复核 → 验收」四级人工质检与自动轨迹过滤后转入 LeRobot 等格式。
2. **骨干与两阶段训练（Sec. 4 & Table 6）：** 未提出新骨干，而是以 **LingBot-VLA-2.0**（`Qwen3-VL-4B-Instruct` 视觉-语言骨干 + 36 层 MoE 动作专家，每层 32 路由专家 top-4 + 1 共享专家，共约 **6B** 参数；图像缩至 `256×256`、`16×16` patch、`2×2` merge；10 步 flow matching 预测 50 步 action chunk）为起点：
   - **Stage 1（Domain-specific retraining → DSFM）：** 在 100 任务 680 小时语料上用 `32 × NVIDIA H100`（FSDP2、全局 batch 1024）训 10 epoch（238,970 步），Muon 优化器恒定 LR `5e-5`（视觉编码器未冻结，LR 降 50 倍为 `1e-6`），保留 `0.004` 权重的深度与未来帧辅助损失。
   - **Stage 2（Task-specific post-training）：** 从 Stage 1 第 10 epoch（不做离线挑 ckpt）出发，在单个任务上继续微调 **20 epoch**。
3. **25 任务真机基准与基线（Sec. 5 & Appendix A–E）：** 在 Kuavo4pro 上统一评测 `N=15` 次真机 rollout（最长 3 分钟，3 次连续子任务失败或碰撞即终止），所有模型均在同一 20-epoch 后训练预算与同一 RTX 4090 `bf16` + 10 Hz / 50-step open-loop chunk（每 5s 重规划）插值到 100 Hz WBC 下运行：
   - **总体（25 任务）：** KUAVO-VLA-1.0 **48.3% SR / 74.5% PS**（旧截图 48.27% / 74.51% 与此四舍五入口径一致），对比 `LingBot-VLA-2.0` 16.3% / 42.1%、`π0.5` 12.8% / 32.1%、`GR00T N1.7 (3B)` 6.7% / 21.2%。
   - **20 个工业任务（Seen）：** 49.7% SR / 74.1% PS vs LingBot-VLA-2.0 14.0% / 40.1%（注：这 20 任务的 Stage 2 示教即 Stage 1 数据子集 135.9h，KUAVO-VLA-1.0 实际多过了一轮 10 epoch）。
   - **5 个 GM100 未见任务（Unseen in-domain，130 episodes/task，平均 2.0h）：** 42.7% SR / 76.2% PS vs LingBot-VLA-2.0 25.3% / 49.7%；PS 在全部 5 项上均领先（1.34×–1.78×），但 `n=5` SR 的 Wilcoxon 检验未达显著（`p=0.63`，仅总体 `p=2.2e-4`、工业组 `p=5.2e-4` 显著），且优势主要由 BM-45 零食分拣（73.3% vs 0.0%）贡献。
   - **薄弱类别：** 装配/插接均值仅 **6.7% SR / 52.8% PS**（Task 16 圆柱孔轴装配所有模型 SR 均为 0.0%，KUAVO-VLA-1.0 仅 PS 领先 50.6% vs 5.0%）。
4. **未证实的营销口径：** 2026-10-03 截图所称「新任务算力与数据成本下降 **50%+**、训练周期从数周压缩到数天」**未出现在**官方项目页正文与 24 页技术报告 `kuavo-vla.pdf` 中，不可当作已验证实验结论引用。

## 对 wiki 的映射

- 项目知识页：[KUAVO-VLA-1.0](../../wiki/entities/kuavo-vla-1.md)
- 通用训练与部署栈：[LeTools](../../wiki/entities/letools.md)
- 代码归档：[letools-learning.md](../repos/letools-learning.md)
- 数据社区：[OpenLET](../../wiki/entities/openlet.md)

## 核验入口

- [KUAVO-VLA-1.0 官方项目页](https://model.lejurobot.com/kuavo-vla-1/) 与 [技术报告 PDF](https://model.lejurobot.com/kuavo-vla-1/kuavo-vla.pdf)
- [LeTools-Learning README](https://github.com/LejuRobotics/LeTools-Learning)
- [Hugging Face 模型卡](https://huggingface.co/LejuRobotics/LET-KUAVO-VLA-1.0-models) 与 [数据集卡](https://huggingface.co/datasets/LejuRobotics/LET-KUAVO-VLA-1.0-Dataset)
