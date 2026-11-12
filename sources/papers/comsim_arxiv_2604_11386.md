# ComSim：Building Scalable Real-World Robot Data Generation via Compositional Simulation（arXiv:2604.11386）

> 来源归档（paper）

- **标题：** Building Scalable Real-World Robot Data Generation via Compositional Simulation（arXiv 版标题带前缀 "ComSim:"）
- **类型：** paper
- **作者：** Yiran Qin, Jiahua Ma, Li Kang, Wenzhan Li（共同一作）等，通讯 Ruimao Zhang；单位含 CUHK-Shenzhen、中山大学、上海交通大学、USTC、港大、牛津、哈佛
- **arXiv：** <https://arxiv.org/abs/2604.11386>（v1，2026-04-13，cs.RO）
- **PDF：** <https://arxiv.org/pdf/2604.11386v1>
- **项目页：** <https://faceong.github.io/ComSim/>（入库日核对：页面仅链到 arXiv 摘要页，未列 GitHub 链接）
- **代码：** 截至 2026-11-12 论文与项目页均未给出代码仓库
- **入库日期：** 2026-11-12
- **一句话说明：** 用少量真机数据训练「仿真视频 → 真实风格视频」的神经模拟器，把经典模拟器（RoboTwin）里带精确动作的大量轨迹转成伪真实视频，与少量真机数据联合训练 Diffusion Policy。

## 核对范围与通读

- **核对范围：** arXiv v1 PDF 全文（正文第 1–5 章、局限与未来工作、参考文献），以及附录第 6–11 章（LLM 使用声明、任务细节、神经模拟器与 DP 训练细节、评测平台与设置、Real2Sim 对齐细节与相机标定代码、补充可视化的文字说明）；表 1–3 逐格核对，图 1–12 仅看图注与文字描述，未逐帧比对生成视频。未运行任何代码，未读项目页之外的第三方复现。
- **通读：** 已完成

## 核心摘录

1. **方法（§3）：** 先在真机采集 (V_real, A)，在对齐好的 RoboTwin 数字孪生里回放同一动作得到 V_sim，构成三元组 (V_sim, V_real, A)；训练神经模拟器 N 把 V_sim 转成 V_real 风格且保持动作（式 1、式 2）。再用动作原语加 GPT-5 生成代码在 RoboTwin 中批量产出 (V_sim, A)，经 N 转成「Pseudo Real Data」，与少量真实数据按比例 α 混合训练策略（算法 1）。
2. **神经模拟器训练数据（§3.3）：** 10 个任务共 200 对训练数据。
3. **神经模拟器实现（附录 8.1）：** 基于 Stable Diffusion 1.5 微调，固定文本指令 "Change the image style from ... simulated environment to ... DSLR camera"，推理时用 FRESCO 保持时序一致；单张 H200，约 10 小时；损失为去噪损失 1.0 加感知损失 0.2 加像素 MSE 0.1。正文 §3.3 与图 2 称用 DiT 对控制动力学与视觉动力学分数做组合。
4. **视频质量（§4.1，表 1）：** Ours-Full 在 PSNR 19.577、SSIM 0.6484、CLIP 0.8102、LPIPS 0.2647、FID 147.90、FVD 15.765 上均为最优；Ours-CD（仅动作条件）最差；Ours-VD 次优。
5. **真机策略（§4.2，表 2）：** Diffusion Policy，每格 30 次试验。7 个任务变体中，10 Real + 200 Pseudo Real 的分布内成功率均高于 20 Real，如 Shake Bottle 28/30 对 17/30，Move Playing-Card Away 29/30 对 24/30；10 Real + 200 Sim 为 6/30 与 6/30。
6. **泛化（§4.3，表 2、表 3）：** 空间 OOD 下 20 Real 基本为 0 到 3/30，10 Real + 200 Pseudo Real 为 5 到 17/30；新物体：Shake Bottle 15/30、Move Playing-Card Away 21/30，对 20 Real 的 0/30 与 2/30。
7. **局限（第 5 章后）：** 实验仅限桌面操作；未来可用非配对数据。附录 10.3 自述物体位置对齐是「naive」的 5 cm 网格加固定角度目录。

## 对 wiki 的映射

- [paper-comsim](../../wiki/entities/paper-comsim.md)
