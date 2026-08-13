# wam_realtime_async_arxiv_2608_01880

> 来源归档（ingest）

- **标题：** World Action Models in Real Time: An Empirical Study of Smooth Execution via Asynchronous Deployment
- **短名：** WAM Real-Time Async / Beyond Stalls
- **类型：** paper
- **来源：** arXiv abs / PDF；官方博客
- **原始链接：**
  - <https://arxiv.org/abs/2608.01880>
  - <https://arxiv.org/pdf/2608.01880>
- **博客：** <https://www.motubrain.com/zh/research/beyond-stalls-deploying-world-action-models/> — [`sources/blogs/motubrain_beyond_stalls_wam_async.md`](../blogs/motubrain_beyond_stalls_wam_async.md)
- **项目页：** <https://www.motubrain.com/zh/research> — [`sources/sites/motubrain-com.md`](../sites/motubrain-com.md)
- **平台仓：** <https://github.com/shengshu-ai/Motubrain> — [`sources/repos/motubrain.md`](../repos/motubrain.md)
- **作者：** Motubrain Team（Mengchen Cai 发起；Jiangfeng Liu 项目负责人；Yinze Rong 参与）；致谢 Jun Zhu
- **机构：** 生数科技（Shengshu Technology）；清华大学（致谢）
- **版本：** arXiv:2608.01880（2026-08）
- **入库日期：** 2026-08-13
- **一句话说明：** 在双臂 10 Hz WAM（\(H=24\)）上对照六种异步 chunk 部署：时序对齐是前提；async+blend 是最低可用基线；去噪内加权伤精度；推理时 RTC（infer）压不住 delay 区；前缀条件训练（train）综合最好。官方仓无本实验可运行代码。

## 核心摘录

### 1) 问题
- WAM 对 video–action 做迭代去噪，端到端延迟到秒级。同步 chunk 执行会在边界停顿，闭环频率被推理绑死，动态目标跟不上。
- 异步推理让计算与执行重叠，但相邻 chunk 在 overlap 上不一致；硬切同样抖。
- 四族方法：纯异步切换 / 直接动作加权 / 推理时引导扩散（RTC infer）/ 训练时前缀条件（Training-Time RTC）。

### 2) 时序量
- \(H=24\)，执行地平线 \(s=4\)，\(d_{\mathrm{est}}=8\)（端到端延迟中位数），overlap \(=20\)（delay 8 + remaining 12）。
- delay 区：chunk \(n\) 已承诺、\(n{+}1\) 生成期间仍在执行；理想上两 chunk 在此应一致。
- \(d>d_{\mathrm{est}}\) 时切换落错帧，位置跳变**任何融合都救不回来**。博客强调硬件时间戳对齐。

### 3) 六种策略
| 代号 | 机制 | 是否重训 |
|------|------|----------|
| sync | 等推理完硬切 | 否 |
| async | \(d_{\mathrm{est}}\) 硬切、不融合 | 否 |
| async+blend | 输出层加权（SmolVLA 思路） | 否 |
| simple | 去噪逐步加权（HoloBrain-0 SimpleRTC） | 否 |
| infer | 推理时改去噪速度场（RTC） | 否 |
| train | 训练时注入已承诺前缀（Training-Time RTC） | 是 |

### 4) 实验（论文 / 博客）
- **平台：** 双臂末端控制 10 Hz；WAM 即 Motubrain 路线。
- **离线：** infer 在 delay 区 MAE/max 明显高于 simple/train；async 最差；simple/train 把 delay 区压近零。
- **真机（每格 5 trial）：**
  - 传送带取物：sync/async **20**；async+blend **40**；simple **80**；train **96**（61.24 s）
  - 插块入槽：simple **27.5**（精度被平滑吃掉）；sync **72.5** / train **70**（train 更快：12.13 s vs 19.4 s）
  - 微波炉放食物：train/sync **96**；sync 85.18 s，train 68.9 s，异步普遍 60–66 s
- **局限：** 突发障碍时旧 chunk 前缀变成干扰；平滑 vs 反应仍开放。

### 5) 开源核查（步骤 2.5）
- 博客与 arXiv 互指；GitHub 指向 Motubrain **模型仓**，不是本实验脚本。
- [`shengshu-ai/Motubrain`](https://github.com/shengshu-ai/Motubrain) 仅 LICENSE + PDF + README + figures → **本实证无可运行训练/评测入口**。
- **结论：** 论文+博客可读；复现六策略代码 **未发布**。

## 对 wiki 的映射

- 升格 [WAM 实时异步部署实证](../../wiki/entities/paper-wam-realtime-async.md)
- 配套 [Motubrain](../../wiki/entities/paper-motubrain.md)
- 更新 [Action Chunking](../../wiki/methods/action-chunking.md)、[WAM 概念](../../wiki/concepts/world-action-models.md)、[VLA 部署指南](../../wiki/queries/vla-deployment-guide.md)

## 当前提炼状态

- [x] 六策略 + 真机三任务表 + 开源边界
- [x] wiki 实体与交叉引用
