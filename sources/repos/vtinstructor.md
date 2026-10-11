
# VTInstructor（TidalHarley/VTInstructor）

> 来源归档（repo）

- **标题：** VTInstructor — Visual Trajectory Prompting for Navigation Instruction Generation in Continuous Environments
- **类型：** repo / vision-language-navigation / instruction-generation
- **来源：** TidalHarley（GitHub）
- **链接：** <https://github.com/TidalHarley/VTInstructor>
- **论文：** [arXiv:2608.15284](https://arxiv.org/abs/2608.15284) · ACM MM 2026
- **项目页：** <https://tidalharley.github.io/VTInstructor/>
- **模型：** <https://huggingface.co/TidalYang/VTInstructor-8b>
- **许可证：** Apache-2.0（代码仓库）
- **入库日期：** 2026-10-11
- **一句话说明：** 官方代码覆盖 Habitat 轨迹预处理、VTP 掩码、VP-Adapter SFT、VP-GRPO、评测与指令生成。
- **开源状态：** **部分开源** — README 列出主要流水线，但声明 EDTC 中 DPC 子模块未提供；权重另在 Hugging Face 发布。

## 入口与执行顺序

| 仓库路径 | 职责 |
|---|---|
| preprocess/rendering/ | 沿 R2R-CE / RxR-CE 轨迹渲染观测 |
| visual_prompt/render_masks.py | 生成路径、转向、终点三通道掩码 |
| vtmod/ | 编码掩码并注入视觉骨干 |
| train/sft.py、train/grpo.py | 监督微调与 VP-GRPO |
| eval/infer.py、metrics/evaluate.py | 验证集推理与 NLG 评分 |
| generate/generate_instructions.py | 为下游 VLN follower 生成增强指令 |
| run_pipeline.sh | 串联主要预处理与训练阶段 |

README 提供环境、训练和评测脚本；模型卡说明权重基于 Qwen3-VL-8B-Instruct，模型推理需正确挂接 VP 模块。运行时还需准备对应 Habitat 场景与 VLN-CE 数据。

## 许可注意
代码仓为 Apache-2.0。HF 模型卡也标注该许可，同时提示训练数据遵循 R2R/RxR、Matterport3D 各自条款，并称 checkpoint 面向非商业研究；Qwen3-VL 底座许可同样适用。

## 对 wiki 的映射
- [VTInstructor 论文实体](../../wiki/entities/paper-vtinstructor.md)
- [项目页归档](../sites/vtinstructor.md)
- [论文归档](../papers/vtinstructor_arxiv_2608_15284.md)
