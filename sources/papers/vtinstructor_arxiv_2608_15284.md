# VTInstructor（arXiv:2608.15284）

> 来源归档（paper）

- **标题：** VTInstructor: Visual Trajectory Prompting for Navigation Instruction Generation in Continuous Environments
- **类型：** paper / vision-language-navigation / instruction-generation
- **论文：** <https://arxiv.org/abs/2608.15284>（v1，2026-08-15；arXiv 页面标注 ACM Multimedia 2026 accepted）
- **项目页：** <https://tidalharley.github.io/VTInstructor/>
- **代码：** <https://github.com/TidalHarley/VTInstructor>
- **模型：** <https://huggingface.co/TidalYang/VTInstructor-8b>
- **作者：** Haolin Yang、Yuxing Long、Zihan Yang、Hao Dong（前两位共同一作；Hao Dong 为通讯作者）
- **机构：** CFCS, School of Computer Science, Peking University；PrimeBot（项目页列示）
- **入库日期：** 2026-10-11
- **一句话说明：** 将连续第一人称 RGB 轨迹压缩为导航关键帧，并把路径、转向、终点视觉化，让视觉语言模型生成可执行的导航指令。

## 核心论文摘录（MVP）

### 1）问题与贡献（摘要）
连续环境中的导航指令生成以密集 RGB 视频为输入，路线结构不像离散视点图那样显式。VTInstructor 将隐式轨迹几何变为显式视觉轨迹提示，目标是在无需导航图、预建地图或场景重建的条件下，从 ego-centric RGB 视频生成 VLN 指令。
- **对 wiki 的映射：** [论文实体](../../wiki/entities/paper-vtinstructor.md)；[VLN 任务](../../wiki/tasks/vision-language-navigation.md)。

### 2）方法（项目页 Method Overview）
EDTC 通过动作游程编码、小步组合和视觉–几何优先级选择，将长轨迹压缩为最多 32 个导航关键帧；VTP 在关键帧叠加路径带、转向箭头和终点标记，并编码为三通道二值掩码；VTMod 将掩码特征注入视觉编码器；之后用 SFT 与 VT-GRPO 校准语言生成和空间注入门控。
- **源码边界：** 仓库称主流程代码已开放，但 EDTC 中的 DPC 子模块尚未发布。
- **对 wiki 的映射：** [论文实体](../../wiki/entities/paper-vtinstructor.md)；[Habitat-Sim](../../wiki/entities/habitat-sim.md)。

### 3）评测与下游效用（论文 / 项目页）
在 R2R-CE、RxR-CE Val Unseen 上，论文报告标准 NLG 指标达到新高；相对最强基线 CIDEr 分别提高 0.357、0.109。冻结 CorrectNav follower 使用其生成指令时，R2R-CE 成功率为 63.3%，比最强竞争指令来源高 14.7 个百分点；将生成指令加入下游训练数据后，两个基准的 SR 均约提升 3 个百分点。
- **对 wiki 的映射：** [论文实体](../../wiki/entities/paper-vtinstructor.md) — 实验、结论与局限；[VLN 任务](../../wiki/tasks/vision-language-navigation.md)。

### 4）复现与使用限制
官方仓库公开渲染、提示掩码构造、SFT、GRPO、评测和生成入口，HF 提供 VTInstructor-8B 权重。代码仓标 Apache-2.0；模型卡亦标 Apache-2.0，但指出训练数据受 R2R/RxR、Matterport3D 等原始条款约束，并将 checkpoint 定位为非商业研究用途，底座模型许可也继续适用。请勿把代码许可等同于数据或模型的无限制许可。
- **对 wiki 的映射：** [论文实体](../../wiki/entities/paper-vtinstructor.md)；[源码归档](../repos/vtinstructor.md)。
