# 新智具身智能（NeoteAI · www.neoteai.com）

- **类型：** 公司 / 视触觉具身智能（产品站）
- **收录日期：** 2026-07-26
- **主站：** <https://www.neoteai.com>
- **研究站：** <https://research.neoteai.com>（归档见 [research-neoteai-com.md](./research-neoteai-com.md)）
- **说明：** 上海新智具身智能科技有限公司（NeoteAI），源自 **复旦大学可信具身智能研究院（TEAI）**；使命「让机器感知世界，让智能触手可及」。以触觉为原生模态，产品线覆盖 **视触觉传感器 → 具身数据平台 → VTLA / 触觉世界模型**。

## 一句话

**把工业级视触觉硬件、大规模视触觉数据采集与 N 系列 VTLA / 触觉世界模型串成「传感器—数据—模型」闭环，服务接触丰富精细操作落地。**

## 为什么值得保留

- **产业侧触觉原生栈样本**：与纯视觉 VLA / 视频 WAM 对照，官网明确走 **视觉+触觉双中心**，并公开 **InTac** 传感器规格与数据平台叙事。
- **与 research.neoteai.com 三件套对齐**：官网「N 系列」对应研究站 **𝒩₀-Foundation / 𝒩₀-VTLA / 𝒩₀-TWAM**（2026-07-25）。
- **融资与场景**：宣称 **近亿元天使轮**；应用场景含智能制造、家纺服装、智能物流。

## 项目页核查（步骤 2.5 · 2026-07-26）

| 核查项 | 结论 |
|--------|------|
| **官网导航** | 产品中心（传感器 / 数据平台 / 大模型 / 软件）、应用场景、新闻、关于我们、**开源主页**（链至 research.neoteai.com）、联系我们 |
| **软件中心** | 链至 GitCode [`neoteai/neoteai-release`](https://gitcode.com/neoteai/neoteai-release)（视触觉传感器 SDK / Studio / 文档） |
| **研究开源** | 见 [research-neoteai-com.md](./research-neoteai-com.md)：OpenNeoData **已开放**；模型代码/权重仓 **占位**，roadmap 写明 **2026-07-31** 前发布 |
| **开放程度** | **部分开源** — 传感器 SDK（GitCode）+ OpenNeoData（HF/ModelScope，门禁）；𝒩₀ 模型训练/推理代码与 checkpoint **截至入库日未落仓** |

- **代码（传感器）：** <https://gitcode.com/neoteai/neoteai-release>
- **代码（研究）：** <https://github.com/neoteai>（N0-Foundation / N0-VTLA / N0-TWAM，见 `sources/repos/`）
- **数据集：** OpenNeoData — <https://huggingface.co/datasets/NeoteAIEmbodied/OpenNeoData>（ModelScope 镜像）
- **模型 checkpoint：** 截至入库日 **未发布**（仓库 roadmap：By July 31, 2026）

## 公开信息要点

### 产品线

| 组件 | 角色 |
|------|------|
| **InTac M1 / S1** | 视触觉指尖传感器；M1 适配平动夹爪（~30 μm/px、法向 0–30 N、30 fps）；S1 紧凑型（~20 μm/px、法向 0–10 N、**120 fps**、40 g） |
| **InTac G1 / F1** | G1：感知驱动一体化；F1：超小指尖形态，面向灵巧手 |
| **具身数据平台** | 真机遥操 + UMI/无本体手持双轨；宣称近百工作台、7×24 采集 |
| **N 系列大模型** | VTLA + 触觉世界模型 + 触觉强化学习后训练叙事 |

### 公司与团队（About）

- **主体：** 上海新智具身智能科技有限公司
- **渊源：** 复旦大学可信具身智能研究院（[teai.fudan.edu.cn](https://teai.fudan.edu.cn/)）
- **融资：** 近亿元天使轮（官网口径）
- **地址：** 上海市杨浦区莱蒙国际中心 B 座 1205 室
- **联系：** 021-65809368 · mkshan@neoteai.com

### 近期动态（新闻摘录）

| 日期 | 事件 |
|------|------|
| 2026-07-16 | 预告 WAIC 2026 参展 |
| 2026-05-29 | 上海数智未来峰会视触觉精细操作 Demo |
| 2026-04-20 | CEO 赵世豪出席 GEIA Asia 2026 灵巧手论坛 |
| 2026-03-05 | 静安区「数智领航」企业 |

## 交叉链接

- 研究站：[research-neoteai-com.md](./research-neoteai-com.md)
- 论文归档：[n0_foundation.md](../papers/n0_foundation.md) · [n0_vtla.md](../papers/n0_vtla.md) · [n0_twam.md](../papers/n0_twam.md)
- 仓库：[n0-foundation.md](../repos/n0-foundation.md) · [n0-vtla.md](../repos/n0-vtla.md) · [n0-twam.md](../repos/n0-twam.md) · [neoteai-release.md](../repos/neoteai-release.md)
- Wiki：[neoteai.md](../../wiki/entities/neoteai.md)
