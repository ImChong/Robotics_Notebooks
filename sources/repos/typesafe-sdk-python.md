# TypeSafe AI Python SDK（typesafe-ai/typesafe-sdk-python）

- **URL：** <https://github.com/typesafe-ai/typesafe-sdk-python>
- **PyPI：** `typesafe-sdk`（README 示例 `uv add typesafe-sdk`）
- **文档：** <https://docs.typesafe.ai/sdk/python/>
- **许可：** MIT
- **入库日期：** 2026-09-19
- **关联 wiki：** [typesafe-jev.md](../../wiki/entities/typesafe-jev.md)

## 一句话说明

官方 Python SDK：`TypeSafeClient.system_one(state=..., questions={...})`；`Choice` 定义离散选项 schema。

## 运行时入口

```python
from typesafe_sdk import Choice, TypeSafeClient
with TypeSafeClient() as client:
    response = client.system_one(state={...}, questions={...})
```

## 交叉链接

- [typesafe-sdk-js.md](./typesafe-sdk-js.md)
- [typesafe-ai-skills.md](./typesafe-ai-skills.md)
