# 🧠 DeepThink (Agentic LLM Enhancer)

DeepThink 是一个基于 Agent 架构的大模型回答增强系统。它通过 **LangGraph** 编排工作流，引入了“意图澄清”和“多角色评审迭代”机制，能够让普通大模型也能生成逻辑严密、全面且高质量的专业级回答。

## ✨ 核心特性

- **🤖 意图澄清与多轮确认 (Human-in-the-loop)**：当用户输入的问题模糊或缺乏上下文时，AI 分析专家会主动挂起任务，向用户抛出具体的追问。支持最多 3 轮的互动确认。
- **📝 多角色评审机制**：初稿生成后，系统会自动召唤多位“虚拟审查员”（如逻辑审查员、全面性审查员）对草稿从不同维度进行严苛的挑刺。
- **🔄 自我迭代进化**：根据审查员的意见，Reviser 会重新润色草稿，直到所有审查员一致通过（或达到设定的最大迭代次数），杜绝敷衍回答。
- **⚙️ 架构化 Prompt 管理**：所有的角色设定和 Prompt 均独立在 `config/prompts.yaml` 中。您可以随时零代码增加新的审查角色（例如“安全审查专家”或“性能优化师”）。
- **📊 自动化基准测试框架**：内置了一套自动化测试脚本，可以无头运行多个复杂刁钻的用例（甚至自动模拟敷衍的客户来测试系统的抗压能力），一键生成详尽的评估报告。
- **💾 沉淀与持久化**：最终回答会自动转换为格式美观的 HTML 文件永久保存，同时在终端实时统计各节点的 Token 消耗量。

## 📁 项目结构

```text
deepThink/
├── config/
│   └── prompts.yaml        # 所有的系统 Prompt 和角色定义
├── scripts/
│   └── run_benchmark.py    # 自动化能力评估基准测试脚本
├── src/
│   ├── agents/
│   │   ├── graph.py        # LangGraph 工作流路由与图定义
│   │   ├── nodes.py        # 各个 Agent 节点的具体业务逻辑
│   │   └── state.py        # 全局状态 (GraphState) 定义
│   └── models/
│       └── llm_factory.py  # LLM 模型工厂 (支持 OpenAI/OpenRouter 等)
├── main.py                 # CLI 交互式对话主入口
├── requirements.txt        # 项目依赖
└── .env.example            # 环境变量配置模板
```

## 🛠️ 安装与配置

1. **克隆项目并安装依赖**
   ```bash
   git clone <your-repo-url>
   cd deepThink
   pip install -r requirements.txt
   ```

2. **配置环境变量**
   复制 `.env.example` 为 `.env` 文件：
   ```bash
   cp .env.example .env
   ```
   然后在 `.env` 文件中填入您的配置：
   ```ini
   OPENAI_API_KEY=your_api_key_here
   
   # 如果使用 OpenRouter、Ollama 或国内模型等兼容接口，可以设置 BASE
   # OPENAI_API_BASE=https://openrouter.ai/api/v1
   
   # 模型选择
   DEEPTHINK_MODEL_NAME=gpt-3.5-turbo
   
   # 发生 Review 不通过时，最大允许的重写循环次数
   DEEPTHINK_MAX_ITERATIONS=2
   ```

## 🚀 快速开始

### 1. 启动交互式终端 (CLI)
直接运行主程序，开启您的智能对话：
```bash
python main.py
```
> **Tip**: 您可以尝试输入一个极其简短模糊的需求（例如：“我想开发一个AI小程序”），来体验系统的“多轮灵魂追问”和背后的“激烈审查”过程。生成的最终排版结果会自动存入 `outputs/` 目录中。

### 2. 运行自动化基准测试
为了测试当前配置的模型在极端场景下的真实产品化能力，您可以直接运行基准测试脚本：
```bash
python scripts/run_benchmark.py
```
> 脚本会自动处理挂起点，并在运行结束后在 `outputs/` 目录下生成一份 Markdown 格式的详细测评报告，您可以像批改考卷一样直观地查阅系统的各项能力表现。

## 🤝 参与贡献
欢迎根据您的业务场景自由拓展节点！您可以非常方便地：
1. 在 `prompts.yaml` 中增加特定的 Reviewer 角色。
2. 在 `src/agents/nodes.py` 中接入联网搜索节点 (Web Search Node)。
3. 将输出格式对接到您的飞书、钉钉或者网页后端。
