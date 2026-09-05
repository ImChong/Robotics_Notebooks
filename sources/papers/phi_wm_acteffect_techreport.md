# Phi-WM 1.0 ActEffect: From Predictive Foresight to Consequence Feedback in Robot Learning

> 来源归档（ingest）

- **标题：** Phi-WM 1.0 ActEffect: From Predictive Foresight to Consequence Feedback in Robot Learning
- **类型：** tech-report / world-action-model / VLA / consequence feedback
- **PDF：** <https://www.sunrisingai.com/upload/202608/e597763b7e870a6ec5951d487b4b3e7f.pdf>
- **项目页：** <https://www.sunrisingai.com/>
- **作者栏：** SunRisingAI Lab（光象科技）；微信访谈称联合清华大学李升波课题组
- **入库日期：** 2026-09-05
- **一句话说明：** 把受控世界模型从「推理时预见」改成「训练时后果反馈」：MIP 暴露三档完整动作提案，在冻结 DINOv3 隐空间做反事实排序，部署时卸掉 WM。

## 开源状态（2026-09-05）

- 公司页与 PDF **无** GitHub / 权重 / 数据链接。
- 结论：**确认未开源**。源码运行时序图不适用。
- 产品：[Phi-Bot X1](https://www.sunrisingai.com/) 为工业本体，不是本报告实现。

## 核心摘录（MVP）

### 1) Feedback 而非 Foresight

- **链接：** Eq. (1)
- **摘录要点：** WAM 常见 \(p(o_{t+1}|o_t,l)\) 再出动作；ActEffect 先提案再 \(p(z_{t+1}|z_t,a_t)\)。
- **对 wiki 的映射：** [paper-phi-wm-acteffect](../../wiki/entities/paper-phi-wm-acteffect.md)、[WAM](../../wiki/concepts/world-action-models.md)

### 2) 三提案 + 受控 WM

- **链接：** §4；Fig. 1
- **摘录要点：** Qwen3-VL 4B + DiT/MIP；`{a_ff, a0, a1}` 共享当前状态；4 层 768 Transformer 在 DINOv3 上预测后果。
- **对 wiki 的映射：** [VLA](../../wiki/methods/vla.md)

### 3) 数字

- **链接：** Table 1–4
- **摘录要点：** LIBERO **98.8%**、LIBERO-PLUS **80.3%**、RoboCasa-GR1 **67.5%**（4.6B）。去后果损失 → 97.0%；WM 塞回 VLM 空间 → 97.3%。
- **对 wiki 的映射：** 腾讯访谈 [wechat_tencent_world_model_questions_2026-09-05.md](../blogs/wechat_tencent_world_model_questions_2026-09-05.md)
