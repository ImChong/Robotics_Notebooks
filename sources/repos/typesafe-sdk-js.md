# TypeSafe AI JavaScript SDK（typesafe-ai/typesafe-sdk-js）

- **URL：** <https://github.com/typesafe-ai/typesafe-sdk-js>
- **npm：** `@typesafe-ai/sdk`
- **文档：** <https://docs.typesafe.ai/sdk/javascript/>
- **许可：** MIT
- **入库日期：** 2026-09-19
- **关联论文/产品：** [typesafe_ai_introducing_system_one_models_jev.md](../blogs/typesafe_ai_introducing_system_one_models_jev.md)
- **关联 wiki：** [typesafe-jev.md](../../wiki/entities/typesafe-jev.md)

## 一句话说明

官方 JS/TS SDK：`TypeSafeClient.systemOne({ state, questions })` 返回类型推断的结构化 choice/概率答案；需 `TYPESAFE_API_KEY`。

## 运行时入口

```ts
import { choice, TypeSafeClient } from "@typesafe-ai/sdk";
const client = new TypeSafeClient();
await client.systemOne({ state: {...}, questions: { category: choice(...) } });
```

## 交叉链接

- [typesafe-sdk-python.md](./typesafe-sdk-python.md)
- [typesafe-ai-skills.md](./typesafe-ai-skills.md)
- [typesafe-ai 站点](../sites/typesafe-ai.md)
