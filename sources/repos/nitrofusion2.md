# NitroFusion2 — 官方仓库及推理调用链

- **类型：** repo / text-to-image / diffusion inference
- **仓库：** https://github.com/ChenDarYen/NitroFusion2
- **用户提供的仓库地址：** https://github.com/chendaryen/nitrofusion2 （GitHub 归一化到上方大小写规范）
- **论文：** https://arxiv.org/abs/2611.05648
- **主页：** https://chendaryen.github.io/NitroFusion2.github.io/
- **权重：** https://huggingface.co/ChenDY/NitroFusion2
- **许可证：** 根目录 `LICENSE` 为 Apache License 2.0；`CITATION.cff` 同样标记 Apache-2.0。基座/权重需另核各自许可。
- **入库日期：** 2026-11-13
- **核对范围：** 通读官方 README、全部 `inference.py`（load、generate、CLI 主函数）、`CITATION.cff`；核对 `LICENSE` 许可头和根目录中 workflows、assets 的可见文件清单及 README 的运行步骤。未逐个执行或复核所有 ComfyUI JSON 内部节点，未运行 GPU 推理。
- **通读：** 已完成
- **开放程度：** 已开源推理代码、ComfyUI 工作流、权重（含量化变体）、在线演示；仓库当前未提供论文 TSAD/ACD 的完整训练程序。

## 已核对运行流程

1. `inference.py` 顶部固定基座 `Tongyi-MAI/Z-Image-Turbo`、checkpoint `ChenDY/NitroFusion2`、LoRA 权重 `nf2-dedistill-lora.safetensors`。
2. `load(repo, dedistill_lora)` 从 checkpoint 的 `transformer` 子目录装载 `ZImageTransformer2DModel`（bf16），在基座 `ZImagePipeline` 中替换 transformer，再按开关加载 LoRA。
3. CLI `__main__` 从 `--prompt`、`--steps`、`--height`、`--width`、`--seed`、`--out` 等读取参数，调用 `pipe.enable_model_cpu_offload()`。
4. `generate()` 限制 steps ∈ {1,2,3,4}，更新 `FlowMatchEulerDiscreteScheduler`；shift 分别是 3.0/6.0/8.0/9.0，sigmas 逐步按 `1-i/steps` 递减。
5. 默认两步 LoRA scales 为 `[0.5, 0.0]`。第一步用 LoRA，`callback_on_step_end` 在步间把 adapter scale 更新到 0.0；采样 `guidance_scale=0.0`，生成 PNG 保存到 `--out`。若 `--no-dedistill-lora` 则全程关闭。
6. README 同时提供 `workflows/nitrofusion2_1step.json`、`nitrofusion2_2step.json`，以及 GGUF 配套流程；2026-11-12 新闻条目指向新的 Union ControlNet（边缘、深度、姿态、灰度、修复等）。

## 可复现入口

```bash
python inference.py --prompt "a cat holding a sign that says hello world" --steps 2 --out cat.png
```

官方未给定本次环境的实际性能数据；依赖安装、显卡与模型下载会影响可运行性。此目录里的代码用来**推理**，不可把 `inference.py` 误标为论文的判别器训练入口。

## 对 wiki 的映射

- [NitroFusion2 项目实体与源码时序图](../../wiki/entities/paper-nitrofusion2.md)
- [论文通读归档](../papers/nitrofusion2_arxiv_2611_05648.md)
- [项目页核对](../sites/nitrofusion2_project.md)
