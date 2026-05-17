# Tencent-Hunyuan / HY-Motion-1.0

- **URL（仓库）**: https://github.com/Tencent-Hunyuan/HY-Motion-1.0
- **URL（模型卡）**: https://huggingface.co/tencent/HY-Motion-1.0
- **维护方**: Tencent Hunyuan 3D Digital Human Team
- **定位**: 文本（+ 期望时长）→ **SMPL-H** 系 **3D 人体运动** 的 **DiT + Flow Matching** 开源实现与权重发布仓；论文 **arXiv:2512.23464**（HY-Motion 1.0）

## 仓库要点（维护者速览）

- **推理栈**：与论文一致的 **HY-Motion DiT** 生成主干 + **时长 / 提示改写 LLM** 模块（以 README 中的依赖与脚本为准）。
- **变体**：README 中区分全尺寸与 **Lite** 等配置；显存与步数以官方说明为准。
- **许可**：以仓库根目录 `License.txt` 及附带声明为准（商用前务必人工复核）。

## 对 wiki 的映射

- 与论文摘录同批入库：[hy-motion-1](../../wiki/methods/hy-motion-1.md)
- 原始论文来源：[hy_motion_arxiv_2512_23464.md](../papers/hy_motion_arxiv_2512_23464.md)
