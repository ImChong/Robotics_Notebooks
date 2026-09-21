# RelateAnything（Maelic/RelateAnything）

- **URL：** <https://github.com/Maelic/RelateAnything>
- **License：** Apache-2.0
- **Python：** 3.12+
- **关联项目页：** [RelateAnything](../sites/relateanything-project.md)
- **关联论文：** [relateanything_arxiv_2609_12552](../papers/relateanything_arxiv_2609_12552.md)
- **HF 模型：** <https://huggingface.co/collections/maelic/relateanything>
- **HF 数据集：** <https://huggingface.co/datasets/maelic/RA-4M>

## 运行入口（README）

- `pip install -e ".[hub]"` → `from relsgg import RelateAnything`
- `RelateAnything.from_pretrained("maelic/relsgg-vits16plus")` → `model.predict(image, boxes_xyxy)`
- `model.set_vocabulary([...])` 推理时换谓词表
- 可选 `deploy/` ONNX 与 `deploy/render_video.py` 视频渲染

## 交叉链接

- [paper-relateanything](../../wiki/entities/paper-relateanything.md)
