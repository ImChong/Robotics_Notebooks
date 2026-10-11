
# VTInstructor（官方项目页）

> 来源归档（project page）

- **类型：** site / project-page
- **官方入口：** <https://tidalharley.github.io/VTInstructor/>
- **论文：** <https://arxiv.org/abs/2608.15284>
- **代码：** <https://github.com/TidalHarley/VTInstructor>
- **模型：** <https://huggingface.co/TidalYang/VTInstructor-8b>
- **作者：** Haolin Yang、Yuxing Long、Zihan Yang、Hao Dong
- **机构：** CFCS, School of Computer Science, Peking University；PrimeBot
- **入库日期：** 2026-10-11
- **一句话说明：** 官方主页概述连续环境 VLN 指令生成方法、评测结果，并提供论文、代码与模型入口。
- **开源核查（2026-10-11）：** **部分开源** — 主训练、推理、评测和生成脚本及模型权重公开；官方仓库注明 EDTC 内的 DPC 子模块尚未发布。

## 页面要点
- 输入为连续环境中的第一人称 RGB 轨迹，输出自然语言导航指令。
- 方法依次包括 EDTC 关键帧压缩、VTP 路径视觉提示、VTMod 特征注入和 VT-GRPO 奖励校准。
- R2R-CE、RxR-CE Val Unseen 评测同时包含 NLG 指标、follower 下游效用与数据增强实验。
- 作者机构以项目页列示为准：北京大学计算机学院（CFCS）及 PrimeBot。

## 对 wiki 的映射
- [VTInstructor 论文实体](../../wiki/entities/paper-vtinstructor.md)
- [论文归档](../papers/vtinstructor_arxiv_2608_15284.md)
- [官方源码归档](../repos/vtinstructor.md)
