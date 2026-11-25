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
- **通读：** 已完成
- **一句话说明：** 将连续第一人称 RGB 轨迹压缩为导航关键帧，并把路径、转向、终点视觉化，通过 ViT 第 7 层近零门控注入 Qwen3-VL-8B 后经 SFT + VT-GRPO 微调，无导航图或 3D 重建即可生成可执行 VLN 指令。

## 原文通读摘录（arXiv:2608.15284）

### 1）问题与贡献（§1）
连续环境中的导航指令生成以密集 RGB 视频为输入，路线结构不像离散视点图那样显式。VTInstructor 将隐式轨迹几何变为显式视觉轨迹提示，在无需导航图、预建地图或场景重建的条件下从 ego-centric RGB 视频生成 VLN 指令。
- **对 wiki 的映射：** [论文实体](../../wiki/entities/paper-vtinstructor.md)；[VLN 任务](../../wiki/tasks/vision-language-navigation.md)。

### 2）方法四组件（§3）
1. **EDTC（事件驱动轨迹压缩）：** RLE 动作压缩 → Combo 小步合并 → 视觉-几何动态打分选出最多 `F_max = 32` 关键帧（注：仓库中 DPC 子模块未开放）。
2. **VTP（视觉轨迹提示）与质量过滤：** 渲染 3 通道二值掩码叠画路径带、旋转弧与终点标记，并结合 GPT 质量过滤构建图文对。
3. **VTMod（视觉轨迹调制器）：** 用卷积 + 自适应池化编码掩码，在 Qwen3-VL-8B 视觉编码器 **第 7 层** 进行逐 patch 调制，并采用接近零初始化的可学习门控 `g` 以免破坏预训练视觉特征。
4. **SFT + VT-GRPO 两阶段训练：** 8×H200 上先做 SFT，再以 `G = 8` 采样组的加权 B1/B4/METEOR/ROUGE-L/CIDEr 奖励叠加门控对比损失执行 VT-GRPO 优化。
- **对 wiki 的映射：** [论文实体](../../wiki/entities/paper-vtinstructor.md)；[Habitat-Sim](../../wiki/entities/habitat-sim.md)。

### 3）评测与下游效用（§4）
在 R2R-CE Val Unseen 上达到 CIDEr **0.560**（相对 Gemini-3.1-Pro 提升 +0.357），RxR-CE Val Unseen CIDEr 达 **0.142**；冻结 CorrectNav follower 使用其生成指令时，R2R-CE 成功率为 **63.3%**，比最强竞争指令来源高 14.7 个百分点；将生成指令加入下游训练数据后，两个基准的 SR 均约提升 3 个百分点。
- **对 wiki 的映射：** [论文实体](../../wiki/entities/paper-vtinstructor.md) — 实验、结论与局限；[VLN 任务](../../wiki/tasks/vision-language-navigation.md)。

### 4）复现与使用限制
官方仓库公开渲染、提示掩码构造、SFT、GRPO、评测和生成入口，HF 提供 VTInstructor-8B 权重。代码仓标 Apache-2.0；模型卡亦标 Apache-2.0，但指出训练数据受 R2R/RxR、Matterport3D 等原始条款约束，并将 checkpoint 定位为非商业研究用途，底座模型许可也继续适用。请勿把代码许可等同于数据或模型的无限制许可。
- **对 wiki 的映射：** [论文实体](../../wiki/entities/paper-vtinstructor.md)；[源码归档](../repos/vtinstructor.md)。
