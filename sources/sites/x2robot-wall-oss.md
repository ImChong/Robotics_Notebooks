# WALL-OSS / WALL-OSS-0.5 官方页面（自变量机器人）

> 来源归档

- **标题：** WALL-OSS — Igniting VLMs toward the Embodied Space；Wall-OSS-0.5 — Pretrain Once, Act Anywhere
- **类型：** site / research post（官方研究页 ×2）
- **机构：** 自变量机器人（X Square Robot）
- **入库日期：** 2026-10-09
- **研究列表页：** <https://x2robot.com/research>（英文 <https://x2robot.com/en/research>）——时间线两条：**WALL-OSS-0.5 · 2026.05.28**（"fully open-source, trained with gradient-bridged co-training, and deployable on real robots straight from pretraining."，链接 `/oss`）；**WALL-OSS · 2025.09.08**（链接 `/research/68bc2cde8497d7f238dde690`）
- **代码：** <https://github.com/X-Square-Robot/wall-x>（两代共用同一仓库，见 [wall-x 仓库归档](../repos/wall-x.md)）
- **Hugging Face 组织：** <https://huggingface.co/x-square-robot>
- **一句话说明：** 两个官方页分别对应 WALL-OSS（2025-09）与 WALL-OSS-0.5（2026-05）两代具身基础模型；两代各有独立 arXiv 技术报告。站点为 Next.js 前端渲染，静态 curl 只能拿到 RSC 负载；正文用 Playwright 渲染核对。

## 页面 1：WALL-OSS（2025-09-08）

- **URL：** <https://x2robot.com/research/68bc2cde8497d7f238dde690>（英文 <https://x2robot.com/en/research/68bc2cde8497d7f238dde690>）
- **论文：** [arXiv:2509.11766](https://arxiv.org/abs/2509.11766)《Igniting VLMs toward the Embodied Space》，v1 提交 2025-09-15（cs.RO）；PDF 首页写 *Date: September 8, 2025*，20 位作者，均署 X Square Robot
- **页面按钮：** Paper（arXiv PDF）· Code（wall-x）· Flow-Model（HF `wall-oss-flow`）· Fast-Model（HF `wall-oss-fast`）· ModelScope（<https://modelscope.cn/organization/X-Square-Robot>，未逐个核对 ModelScope 上的文件）
- **页面要点：**
  - 主干 **Qwen2.5-VL-3B**；输入为第一人称视角 + 臂载相机画面加文本指令，不同训练阶段输出不同
  - **Unified Cross-Level CoT**：在一个可微框架里依次完成指令推理、子目标分解和细粒度动作生成
  - 数据分三类：自采机器人动作数据、开源动作数据、多模态 VQA；页面写规模为 "tens of thousands of hours"（论文 §4 写 "exceeds 10,000 hours"，两处口径不一致）
  - 两阶段训练：**Inspiration**（具身 VQA 加 FAST 离散动作先验）→ **Integration**（flow matching 连续控制；先单训动作分支，再与 VLM 联合优化）
  - 六个操作任务评测；set-table、tidy-bedroom、place-by-color 三项预训练时未见

## 页面 2：WALL-OSS-0.5（2026-05-28）

- **URL：** <https://x2robot.com/oss>（英文 <https://x2robot.com/en/oss>；HF 卡片链接 `#resources` 锚点）
- **论文：** [arXiv:2605.30877](https://arxiv.org/abs/2605.30877)《Wall-OSS-0.5 Technical Report》，v1 提交 2026-05-29，v2 2026-06-01（cs.RO），27 位作者。官方 PDF 另挂 <https://x2robot.com/api/files/file/WALL-OSS_0.5.pdf>。各处标题不统一：项目页 BibTeX 写 *Pretrain Once, Act Anywhere*，GitHub README 写 *A Deployment-Ready VLA with Gradient-Bridged Pretraining*
- **页面头部：** "OPEN-SOURCE VISION-LANGUAGE-ACTION MODEL"；Date May 2026；Model: Open Source；Technical report；**"CODE COMING SOON"**（头部按钮）。但同页 **Open Source** 区块写 "We release model weights and training code"，并链到 GitHub wall-x 和 HF `wall-oss-0.5`，**两处说法互相矛盾**；以仓库实际内容为准（见下表）
- **页面关键数字（与 arXiv 报告一致）：**
  - Competence **4/17**：不经微调时任务进度 ≥ 80% 的任务数（页面副文案写"3 unseen tasks"）
  - Efficient Adaptation **+17.5 pp**：15 任务微调平均 60.5 对 π0.5 的 43.0
  - Understanding **+21.8 pp**：Embodied Grounding 提升（页面写 "general VL capability preserved"；报告 §4.3 实际显示 RealWorldQA −15.0、ERQA −5.5，见 wiki 页说明）
- **方法要点：** gradient bridge（action-token CE 是 VLM 主干的主要动作梯度来源，flow matching 对主干更新的占比约 5%）；MoT（VL Expert + Action Expert）；Vision-Aligned RVQ Action Tokenizer 取代 FAST；Action-Space Supervision（等效于 \((1-\tau)^2\) 加权，收敛快约 2×）；单阶段三路共训
- **数据：** 20+ 本体；每 epoch 1M+ 轨迹（约 60% 自采、40% 开源）；90M 多模态样本，其中 12M 为 embodied bridge 样本
- **参数量口径：** 摘要为 "4B VLA built upon a 3B VLM backbone"；同页另一处写 "open-source 3B Vision-Language-Action model"，**口径不一致**，以报告 §2.1.1（"over 4B parameters"）为准

## 开源核查（2026-10-09）

| 入口 | WALL-OSS（2025-09） | WALL-OSS-0.5（2026-05） |
|------|---------------------|--------------------------|
| **论文** | arXiv:2509.11766（v1 2025-09-15） | arXiv:2605.30877（v1 2026-05-29 / v2 2026-06-01）+ 官网 PDF |
| **代码** | **已开源**：wall-x（Apache-2.0）。README 指明 FLOW/FAST 用法需回退到提交 `97406f2ab5de414c79b091873f946c112d105c72` | **已开源**：wall-x **1.1.0**（README 写 2026-06 发布）。`workspace/README.md` 明确"本次开源面向 Wall-OSS-0.5"，含 FSDP 微调、LIBERO 评测、WebSocket 服务与开环评测；默认训练配置依赖 `X-Square-Robot/dmuon`（未单独核对该仓） |
| **权重** | **已开源**：HF `x-square-robot/wall-oss-flow`、`wall-oss-fast`（均创建于 2025-09-06）；`wall-oss-flow-0.1`（2026-01-30，为架构升级版：动作 token 与非动作 token 的 QKV/输出投影不共享参数）；另有 ModelScope 组织页 | **已开源**：HF `x-square-robot/wall-oss-0.5`（创建于 2026-05-28，最后修改 2026-07-07；`model.safetensors` + 动作/本体归一化器 + `config.yml`；未设 gated） |
| **权重许可** | HF 卡片元数据**未声明** license；仓库 LICENSE 为 Apache-2.0 | 同左 |
| **预训练数据** | **未公开**：自采机器人数据与具身 VQA 均未发布 | **未公开**：自采数据、12M bridge 样本均未发布。HF 组织下有 `libero_all`（2026-01-30，LeRobot 格式 LIBERO）和 `XRZero-G0-3K`（2026-05-16）两个数据集，但都不是报告中的预训练语料本身 |
| **评测资产** | 真机六任务与 Embodied VQA 基准未发布 | 17 任务零样本套件、15 任务微调套件未发布；仓库只带 LIBERO 与开环评测脚本 |

## 页面与仓库的交叉核对

- GitHub README 的 News：2025-09 WALL-OSS → 2026-05 Wall-OSS-0.5（arXiv 2605.30877v2）→ 2026-05 WALL-WM（arXiv:2606.01955）→ 2026-06 Wall-X 1.1.0
- README 的 Models 段列出四个权重：WALL-OSS-0.5 / WALL-OSS-FLOW-0.1 / WALL-OSS-FLOW / WALL-OSS-FAST
- HF `wall-oss-0.5` 卡片的 H1 仍写 "WALL-OSS"，Cite 段引用 2025 白皮书 `wall_oss.pdf`，而不是 0.5 报告——卡片没有随新版本同步更新

## 对 wiki 的映射

- WALL-OSS 与 WALL-X 代码主节点：[`wiki/entities/cn-os-wall-x.md`](../../wiki/entities/cn-os-wall-x.md)
- WALL-OSS-0.5 技术报告节点：[`wiki/entities/paper-wall-oss-0-5.md`](../../wiki/entities/paper-wall-oss-0-5.md)
- 代码归档：[`sources/repos/wall-x.md`](../repos/wall-x.md)
