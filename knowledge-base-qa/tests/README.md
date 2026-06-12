# 测试说明

## 端到端测试

端到端测试脚本 `e2e_test.py` 模拟完整的问答流程，包括：

1. **文档导入测试** - 创建测试文档并导入知识库
2. **多轮问答测试** - 验证答案的准确性和关键词匹配
3. **性能基准测试** - 统计查询耗时（P50/P95/P99）
4. **流式输出测试** - 验证流式生成功能

### 运行方式

```bash
# 使用默认配置（DeepSeek + deepseek-chat）
python tests/e2e_test.py

# 指定 LLM 提供商和模型
python tests/e2e_test.py --provider deepseek --model deepseek-chat

# 使用 Ollama 本地模型
python tests/e2e_test.py --provider ollama --model qwen2:7b

# 使用 OpenAI
python tests/e2e_test.py --provider openai --model gpt-3.5-turbo

# 自定义基准测试次数
python tests/e2e_test.py --benchmark-queries 20

# 调整日志级别
python tests/e2e_test.py --log-level INFO
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--provider` | deepseek | LLM 提供商（ollama/openai/deepseek） |
| `--model` | deepseek-chat | LLM 模型名称 |
| `--benchmark-queries` | 10 | 基准测试查询次数 |
| `--verbose` | True | 详细输出 |
| `--log-level` | WARNING | 日志级别 |

### 测试报告

测试完成后会在 `tests/e2e_report.json` 生成 JSON 格式的测试报告，包含：

- 测试时间戳
- 引擎配置
- 每个测试用例的执行结果
- 性能基准统计数据

### 测试用例

测试用例覆盖以下类别：

1. **基础概念测试** - 验证对核心概念的理解
2. **细节查询测试** - 验证对具体知识点的检索
3. **应用场景测试** - 验证对实际应用的理解
4. **综合分析测试** - 验证跨文档的综合分析能力

### 性能指标

基准测试统计以下指标：

- **总耗时** - 所有查询的总执行时间
- **平均耗时** - 单次查询的平均耗时
- **P50** - 50% 的查询在此时间内完成
- **P95** - 95% 的查询在此时间内完成
- **P99** - 99% 的查询在此时间内完成
- **检索耗时** - RAG 检索阶段的平均耗时
- **LLM 耗时** - LLM 生成阶段的平均耗时
- **首 Token 延迟** - 流式输出首个 Token 的平均延迟
