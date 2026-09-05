# 4DAnyone（arXiv:2608.20335）

> 来源归档（ingest）

- **标题：** 4DAnyone: Create Anyone in 4D from a Casual Monocular Video
- **简称：** 4DAnyone
- **类型：** paper / 4dgs / multiview-video / hmr / human-avatar
- **venue：** SIGGRAPH Asia 2026
- **arXiv：** <https://arxiv.org/abs/2608.20335>
- **PDF：** <https://arxiv.org/pdf/2608.20335>
- **HTML：** <https://arxiv.org/html/2608.20335>
- **项目页：** <https://4danyone.github.io/> — 归档见 [`sources/sites/4danyone-github-io.md`](../sites/4danyone-github-io.md)
- **代码：** <https://github.com/ant-research/4DAnyone> — 归档见 [`sources/repos/4danyone.md`](../repos/4danyone.md)
- **权重：** <https://huggingface.co/AntResearch/4DAnyone>
- **机构：** 浙江大学、蚂蚁灵波科技（Robbyant）、蚂蚁集团、香港科技大学、香港中文大学
- **作者：** Yudong Jin\*、Tao Xie\*、Qihang Zhang、Zehong Shen、Zhen Xu、Yujun Shen、Hujun Bao、Xiaowei Zhou†、Yinghao Xu†
- **入库日期：** 2026-09-05
- **一句话说明：** 不标定的单目视频先生成「重建级」多视角人体视频，再抬到 4DGS；RCP 把参考上下文压成 \(O(1)\)，TCR 在去噪分组间传结构。

## 开源状态（步骤 2.5，2026-09-05）

| 组件 | 状态 |
|------|------|
| 项目页 | 已上线；论文写明 source code 入口 |
| GitHub | **已开源** Apache-2.0；`inference.py` + GVHMR 子模块 |
| HF | `AntResearch/4DAnyone`（权重条款分项） |
| 4DGS 重建 | 论文用 FreeTimeGS；仓 Roadmap：**开源 4DGS 方法仍待接**；nerfstudio **3DGS** 已写 |

**结论：已开源** — 多视角生成可复现；论文 4DGS 数字绑 FreeTimeGS，不要写成「仓内一键 4DGS」。

## 核心摘录

### 摘录 1：失败点是注意力上下文，不是「再加一个相机条件」

- 4DGS 要几十路目标视角；单次 DiT 装不下就把视角分组。
- 参考上下文按已生成视角线性涨（\(O(N)\)），外观引导变弱。
- 各组目标上下文不相交 → 跨组结构漂移。
- RCP：多尺度 pack，参考复杂度 \(O(1)\)。TCR：高噪声轮换分组传结构，低噪声固定邻接组修细节。

**对 wiki 的映射：** [paper-4danyone](../../wiki/entities/paper-4danyone.md)

### 摘录 2：精度优先于密度

- 野外视频上稠密度量深度 / 相机参数不可靠。
- 用 [GVHMR](../../wiki/entities/gvhmr.md) 出 3D 骨架，深度缓冲渲染，消掉 2D 骨架前后歧义。
- 从 Goliath 308 点里只用 **40** 个（17 身 + 6 足 + 10 掌级手 + 7 辅助）；脸和细手指交给源视频，不喂噪声关键点。

**对 wiki 的映射：** [paper-4danyone](../../wiki/entities/paper-4danyone.md)、[gvhmr](../../wiki/entities/gvhmr.md)

### 摘录 3：数字与配方

- 骨干 Wan2.2-TI2V-5B；三阶段课 128×H20，约 0.5 + 1 + 1.5 天；推理 20 步；\(t_s/T=0.2\)。
- 训练数据：MVGameHuman 38k / SynCamVideo 34k / DNA-Rendering 51k / TedTalk 42k / Pexels 20k。
- DNA-Rendering 4DGS PSNR **24.15**（ReCamMaster† 20.55）；生成一致性 **24.33**。
- DyMVHumans（OOD）4DGS PSNR **23.28**。
- 消融（一致性）：去掉 RCP+TCR 21.09；Sliding 路由 22.63。Random/Strided 路由没有增益。

**对 wiki 的映射：** [paper-4danyone](../../wiki/entities/paper-4danyone.md)

## 当前提炼状态

- [x] 项目页 + GitHub + HF 核查（2026-09-05）
- [x] arXiv HTML 方法 / Table 2–3 / 附录训练细节
- [x] wiki 映射：`wiki/entities/paper-4danyone.md`
