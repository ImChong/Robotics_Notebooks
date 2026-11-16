# SLIM（Self-supervised Latent Interaction Model）

> 来源归档

- **标题：** SLIM — compact action-grounded latent interaction policy
- **类型：** repo / latent-policy / flow-matching / libero / calvin
- **组织：** Fudan / BAAI / Tsinghua / RUC（论文作者团队）
- **代码：** <https://github.com/kzz1031/SLIM>
- **项目页：** <https://kzz1031.github.io/slim-project-page/> — [`sources/sites/kzz1031-slim-project-page.md`](../sites/kzz1031-slim-project-page.md)
- **论文：** <https://arxiv.org/abs/2608.09771> — [`sources/papers/slim_05b_arxiv_2608_09771.md`](../papers/slim_05b_arxiv_2608_09771.md)
- **权重：** <https://huggingface.co/kzzwang/SLIM-LIBERO> · <https://huggingface.co/kzzwang/SLIM-CALVIN>
- **Stars：** ~2（2026-08-12；新发布）
- **License：** README / 仓内声明为准（API 报 `NOASSERTION`）
- **入库日期：** 2026-08-12
- **一句话说明：** 官方训练与评测栈：Stage-1 掩码轨迹（IDM/FDM）→ Stage-2 flow-matching；`slim.serving.server` + LIBERO/CALVIN 评测 client；HF 权重含 `config.yaml` 与 `action_stats.json`。

## 开源核查（2026-08-12）

- **已开源**：Python 包 `slim`、`scripts/train_stage1_8gpu.sh` / `train_stage2_8gpu.sh`、`scripts/evaluate_all_8gpu.sh`、`slim.serving.server`、LIBERO / CALVIN 评测模块、`configs/libero/` 与消融配置。
- **权重：** public HF，无需鉴权即可 `hf download`。

## 关键复现入口（README）

```bash
git clone https://github.com/kzz1031/SLIM.git && cd SLIM
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .

# Stage 1 / 2（默认 8 GPU）
bash scripts/train_stage1_8gpu.sh
bash scripts/train_stage2_8gpu.sh checkpoints/stage1/<run>/checkpoints/epoch_3_pytorch_model.pt

# 发布权重评测
hf download kzzwang/SLIM-LIBERO --local-dir checkpoints/releases/SLIM-LIBERO
bash scripts/evaluate_all_8gpu.sh \
  checkpoints/releases/SLIM-LIBERO/checkpoints/epoch_40_pytorch_model.pt \
  outputs/slim_libero_release 12000
```

单机 server / client：

```bash
python -m slim.serving.server --checkpoint <ckpt.pt> --port 10093 --bf16
# 另一环境：
python -m slim.evaluation.libero.evaluate --checkpoint <ckpt.pt> \
  --host 127.0.0.1 --port 10093 --task_suite_name libero_10 \
  --num_trials_per_task 50 --action_chunk_size 8 --send-state
```

## 对 wiki 的映射

- 实体：[SLIM-0.5B](../../wiki/entities/paper-slim-05b.md)
- 交叉：[LIBERO](../../wiki/entities/libero-benchmark.md)、[CALVIN](../../wiki/entities/calvin-benchmark.md)、[World Action Models](../../wiki/concepts/world-action-models.md)

## 固定版本源码复核（2026-11-13）

- 版本：[00426cc82a44ca9397d293fcad226e12dc574d2c](https://github.com/kzz1031/SLIM/tree/00426cc82a44ca9397d293fcad226e12dc574d2c)。已读根README、serving/server.py、websocket_server.py、evaluation/libero/environment.py全文；libero/evaluate.py完整eval_libero；slim_model.py构造/forward/predict_action及state提取；slim_transformer.py IDM/FDM分派与predict_action完整函数。
- server载checkpoint并转设备/精度，WebSocket解包调用SLIMModel.predict_action，DINO/T5条件送SLIMTransformer，缓存观测上下文、flow迭代仅更新动作，返回normalized_actions。客户端ModelClient按chunk缓存，q01/q99反归一化前六维，夹爪由仿真入口二值化。
- 环境须按LIBERO/Plus包冲突分开；release的config和action_stats不可丢。README给训练及LIBERO/CALVIN权重链接，未下载/运行。继承的checkpoint加载与训练loop不在本次完整源码核读范围。
