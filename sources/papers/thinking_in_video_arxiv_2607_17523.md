# Thinking in Video: Can Video Generators Really Reason About the Real World?（arXiv:2607.17523）

> 来源归档（ingest）

- **标题：** Thinking in Video: Can Video Generators Really Reason About the Real World?
- **类型：** paper / video generation / world-model consistency / CGDJ / Perception-Prediction Gap
- **arXiv：** <https://arxiv.org/abs/2607.17523>（PDF：<https://arxiv.org/pdf/2607.17523.pdf>）
- **作者：** Yongheng Zhang、Guang Yang、Ruihan Hou、Qiguang Chen 等（中南大学 / 腾讯 / 清华大学）
- **机构：** 中南大学（Central South University）、腾讯（Tencent）、清华大学（Tsinghua）
- **代码：** <https://github.com/BRZ911/Thinking-in-Video>
- **数据集：** <https://huggingface.co/datasets/BRZ911/Thinking-in-Video-Data>
- **入库日期：** 2026-07-27
- **一句话说明：** 把「用视频生成模型做因果推演」重定义为 **Thinking in Video**；提出 **Causal-Generative Dual-Judge（CGDJ）**——**Flatten Temporal Video** 测显式因果感知，参考视频测隐式生成预测——揭示开源模型 **近零感知但仍能出像样动力学** 的 **Perception-Prediction Gap**。

## 开源状态（核查，2026-07-27）

- **已开源：** [BRZ911/Thinking-in-Video](https://github.com/BRZ911/Thinking-in-Video) 含 `Perception/` 与 `Prediction/` 两套流水线脚本；[HF 数据集](https://huggingface.co/datasets/BRZ911/Thinking-in-Video-Data) 公开（非 gated）。
- **可运行入口：** `Perception/01_generate_video_veo3.py` → Whisper 转写 → Gemini-3-Pro 判定；`Prediction/01_generate_video_veo3.py` → Gemini 视频质量打分。
- **边界：** 评测依赖 **Gemini / Veo / Whisper** API 与凭证；仓库 LICENSE 文件未见（截至核查日 API 返回 `license: null`），以 README 使用说明为准；Stage-1 脚本以 Veo 3.1 为示例，可替换为其他生成器输出目录。

## 摘要级要点

- **范式：** 视频不只是产物，而是构造 / 延展 / 验证因果思维的介质。
- **CGDJ：** Explicit Causal Perception（扁平时空 VQA）+ Implicit Generative Prediction（因果后果视频）。
- **Flatten Temporal Video：** \(N{=}70\) 帧铺成 \(7{\times}10\) 网格 + 查询条带 → \(1280{\times}720\) 合成图；隐式轨用 \(N{=}7\) motion-anchor。
- **基准规模：** 约 **1500** 视频（Video-MME 900 + 成对因果前后 600）。
- **主发现：** Perception-Prediction Gap；开源（Wan-2.2-14B、HunyuanVideo-1.5）显式感知近塌但仍出中等续写；闭源（Sora-2、Veo-3.1）对齐更好但仍有限；另有 **音画错位**（口头因果对、画面不对）。

## 核心论文摘录（MVP）

### 1) Thinking in Video 范式

- **链接：** §1；Fig.1
- **摘录要点：** 现有指标把视觉保真与语义逻辑切开；需要同时审计「读得懂」与「演得出」。
- **对 wiki 的映射：**
  - [Thinking in Video](../../wiki/entities/paper-thinking-in-video.md)
  - [Generative World Models](../../wiki/methods/generative-world-models.md)

### 2) Flatten Temporal Video + Dual-Judge

- **链接：** §2；README Method
- **摘录要点：** 解决多数视频生成器只吃静态图条件的接口问题；双轨 Gemini-3-Pro 评判。
- **对 wiki 的映射：**
  - [Thinking in Video](../../wiki/entities/paper-thinking-in-video.md)
  - [`sources/repos/thinking-in-video.md`](../repos/thinking-in-video.md)

### 3) Perception-Prediction Gap

- **链接：** §3 Experiments
- **摘录要点：** 画面连续 ≠ 因果理解；挑战「世界模拟器」叙事。
- **对 wiki 的映射：**
  - [物理保真度输出轴](../../wiki/overview/world-model-physics-fidelity-outputs.md)
  - [Imagined Rollouts…](../../wiki/entities/paper-imagined-rollouts-kinematic-not-dynamic.md)

## BibTeX

```bibtex
@misc{zhang2026thinkingvideovideogenerators,
  title         = {Thinking in Video: Can Video Generators Really Reason About the Real World?},
  author        = {Zhang, Yongheng and Yang, Guang and Hou, Ruihan and Chen, Qiguang and
                   Liu, Ziang and Liu, Xiaolong and Zhang, Manman and Hao, Yanchao and
                   Wei, Zheng and Wu, Hao and Qin, Libo and Dai, Peishan and Li, Yinghui and
                   Yin, Di and Sun, Xing},
  year          = {2026},
  eprint        = {2607.17523},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV}
}
```

## 对 wiki 的映射

- 主实体页：[`wiki/entities/paper-thinking-in-video.md`](../../wiki/entities/paper-thinking-in-video.md)
- 代码归档：[`sources/repos/thinking-in-video.md`](../repos/thinking-in-video.md)
- 互链：[物理保真度输出轴](../../wiki/overview/world-model-physics-fidelity-outputs.md)、[Generative World Models](../../wiki/methods/generative-world-models.md)、[KineBench](../../wiki/entities/paper-kinebench.md)、[Imagined Rollouts…](../../wiki/entities/paper-imagined-rollouts-kinematic-not-dynamic.md)
- 策展入口：[微信 · 世界模型物理保真度](../blogs/wechat_embodied_ai_lab_world_model_physics_fidelity.md)
