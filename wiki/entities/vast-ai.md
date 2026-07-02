---
type: entity
tags: [infrastructure, gpu-cloud, training, international, marketplace]
status: complete
updated: 2026-07-02
related:
  - ./runpod.md
  - ./lambda-cloud.md
  - ../comparisons/international-gpu-cloud-platforms.md
  - ./ppf-contact-solver.md
sources:
  - ../../sources/sites/vast-ai.md
summary: "Vast.ai 是 P2P GPU 租赁市场：主机出租闲置算力，用户按报价筛选；价格常为传统云 40–60% 折扣，适合可 checkpoint 的实验，可靠性因主机而异。"
---

# Vast.ai

**Vast.ai**（[vast.ai](https://vast.ai/)）运营 **GPU 算力市场**：分散主机报价，用户按价格、显存、可靠性分数租卡，是国外 **极致低价** 实验路径。

## 一句话定义

像「算力淘宝」一样竞价租 GPU——便宜，但要自己筛主机、做好 checkpoint，并接受偶发中断。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| GPU | Graphics Processing Unit | 市场挂牌资源 |
| P2P | Peer-to-Peer | 主机方分散出租 |
| RL | Reinforcement Learning | 可恢复训练场景 |
| SLA | Service Level Agreement | 市场模式通常无统一 SLA |
| Docker | Docker Container | 常见隔离方式 |
| API | Application Programming Interface | CLI/REST 租卡 |

## 为什么重要

- **成本地板低**：短跑 RL、渲染、超参 sweep 常比 neocloud 再省 40%+。
- **本库已有挂接**：[PPF Contact Solver](./ppf-contact-solver.md) 文档提及 vast.ai 远程求解模板。
- **与 RunPod 分工**：Vast 省钱；RunPod Secure 要稳定。

## 核心结构 / 机制

- **筛选维度**：$/hr、VRAM、DLPerf、reliability、上传带宽
- **计费**：按秒/小时；实例可中断
- **模板**：数百 Docker 模板（市场资料）
- **最佳实践**：定期 `torch.save` / `rsl_rl` checkpoint；避开无 reliability 过滤的极低价主机

## 常见误区或局限

- **生产 API 不推荐**：延迟与可用性不可控。
- **硬件参差**：数据中心卡与家用矿机混排；多卡 NVLink 不保证。
- **传数带宽**：上传大数据集前看主机网络评分。

## 与其他页面的关系

- [国外 GPU 云平台选型](../comparisons/international-gpu-cloud-platforms.md)
- [RunPod](./runpod.md)、[Lambda Cloud](./lambda-cloud.md)

## 推荐继续阅读

- [Vast.ai 文档](https://vast.ai/docs/)

## 参考来源

- [Vast.ai 官方资料](../../sources/sites/vast-ai.md)
