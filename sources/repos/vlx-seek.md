# VLX-Seek 官方仓库

> 来源归档（repo）

- **标题：** VLX-Seek — Fine-Grained Perception VLM: From Coordinate Generation to Region Reference
- **类型：** repo / vlm / visual-grounding / open-vocabulary-detection / edge-inference
- **机构：** [Om AI Lab](https://github.com/om-ai-lab)（杭州联汇科技 OmAI）
- **代码：** <https://github.com/om-ai-lab/VLX-Seek>（**已开源**，Apache-2.0）
- **项目页 / 博客：** <https://om-ai-lab.github.io/2026_07_06_vlx_seek_1_5_en.html> · [HF Blog](https://huggingface.co/blog/omlab/vlx-seek)
- **在线 Demo：** <https://om-agent.com/#/front>
- **权重：** [omlab/VLX-Seek-1.5-10B](https://huggingface.co/omlab/VLX-Seek-1.5-10B) · [ModelScope Om_AI_Lab/VLX-Seek-1.5-10B](https://modelscope.cn/models/Om_AI_Lab/VLX-Seek-1.5-10B)
- **入库日期：** 2026-09-30
- **一句话说明：** 面向边缘具身场景的细粒度感知 VLM：把定位从「LLM 直接生成坐标串」改为「候选区域 token + 语言检索/引用」，支持开放词汇检测、指代表达、区域 OCR/VQA 与显式拒识。

## 开源状态（步骤 2.5）

| 项 | 核查（2026-09-30） |
|----|-------------------|
| **GitHub** | 公开仓 [om-ai-lab/VLX-Seek](https://github.com/om-ai-lab/VLX-Seek)；含 `inference.py`、`vlx_seek_worker.py`、`vlx_seek/` 包 |
| **权重** | **已发布** VLX-Seek 1.5-10B（HF / ModelScope）；家族规划含 0.6B / 3B / 10B，当前仓以 10B 为主 |
| **区域 proposal** | 博客内训 OPN **未开源**；仓内默认集成开源 **WeDetect-Base-Uni**（`fushh7/WeDetect`），可 `--bbox-list` 跳过或换检测器 |
| **训练代码** | README 未提供完整训练栈；**推理路径可复现** |
| **结论** | **已开源（推理 + 10B 权重 + 可替换 proposal）**；训练与内部 OPN 为部分开放 |

## 仓库入口（README 归纳）

| 组件 | 说明 |
|------|------|
| `inference.py` | CLI：检测 / grounding 等任务；`--model-path omlab/VLX-Seek-1.5-10B` |
| `vlx_seek_worker.py` | `VLXSeekWorker` Python API |
| `vlx_seek/` | 模型结构与 prompt 格式化 |
| `detect_tools/` | 与 WeDetect 等区域 proposal 集成 |
| `demo/` | 示例图像与用法 |

## 技术要点（对 wiki 的映射）

1. **Region reference 而非坐标生成**：候选框编码为 `<obj*>` 区域 token，LLM 输出区域引用再映射回 bbox——解码更短、解析更稳。
2. **HFRE（Hybrid Fine-Grained Region Encoder）**：语义通路 + 细节通路，把 proposal 变成 LLM 可读实体。
3. **两阶段训练叙事**：区域–语言对齐 → 感知指令微调；混一般 VLM 数据防遗忘 + hard-negative 拒识。
4. **能力**：开放词汇检测、REC、区域 caption/OCR/VQA、计数、全图 VQA（可无 proposal）。
5. **技术谱系**：同团队 [OmDet](https://github.com/om-ai-lab/OmDet)、[VLM-R1](https://github.com/om-ai-lab/VLM-R1)、[VLM-FO1](https://github.com/om-ai-lab/VLM-FO1)。

## 对 wiki 的映射

- 主实体：[VLX-Seek](../../wiki/entities/vlx-seek.md)
- 站点归档：[vlx-seek.md](../sites/vlx-seek.md)
- 交叉：[机器人视觉感知栈选型闭环](../../wiki/queries/robot-perception-stack-selection-loop.md)、[五大具身模型 VLX 家族（Vision-Language-X）](../../wiki/comparisons/vlm-vln-vla-vlx-world-model-taxonomy.md)（**命名不同**：本仓 VLX-Seek 为联汇产品系列名，非 taxonomy 里「一体化多任务 VLX」专指）
