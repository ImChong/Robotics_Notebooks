# Google Colab

> 来源归档

- **标题：** Google Colab
- **类型：** site（托管 Notebook GPU 环境）
- **来源：** Google
- **链接：** https://colab.research.google.com/ 、https://colab.research.google.com/signup
- **入库日期：** 2026-07-02
- **一句话说明：** 浏览器内 Jupyter 环境：免费档可用有限 GPU；Colab Pro/Pro+ 提供优先 T4/V100/A100、更长会话与更大数据盘，适合原型验证与小规模实验。

## 为什么值得保留

- **本库大量引用**：MuJoCo MJX、Brax、TSIL、RWM-Lite、dm_control 等均链出 Colab 教程。
- **零摩擦入门**：无 SSH/云账号运维，适合课程与算法验证。
- **与 VM 云分工**：Colab 做实验；确认无误后再上 RunPod/Lambda 长跑。

## 平台要点（公开资料 2026-07 归纳）

| 维度 | 要点 |
|------|------|
| **形态** | 托管 Notebook；非完整 Linux VM |
| **GPU** | 免费随机 GPU；Pro 优先高端卡（型号随配额变化） |
| **会话** | 免费较短；Pro 可达 ~24h、减少空闲断开 |
| **存储** | Google Drive 挂载；非传统 `/data` 盘 |
| **多卡** | 通常单卡；不适合大规模分布式 |
| **TPU** | 可选 TPU runtime（偏 TensorFlow/JAX） |

## 对 wiki 的映射

- 实体页：[google-colab.md](../../wiki/entities/google-colab.md)
- 统一选型：[international-gpu-cloud-platforms.md](../../wiki/comparisons/international-gpu-cloud-platforms.md)
