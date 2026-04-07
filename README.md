# BUPT Campus Smart Life Assistant Agent (北邮校园智能生活助手) 🎓

本项目是一个基于 LangGraph 的多智能体 (Multi-Agent) 架构校园生活辅助系统。系统配备前端 UI 界面，提供自然语言交互的统一入口，能自动拆解复杂任务，下发给各大垂直领域的专家 Agent (如：教务专家、生活推荐专家) ，从而为您提供最无缝的大学生活体验。

## 🌟 核心特性
1. **多域意图识别引擎 (Supervisor Agent)**：能解析“复合指令”（例如查教室的同时推荐食堂），利用 CoT (思维链) 机制进行智能拆解与分发。
2. **底层专家 Worker**: 细分的教务 Agent 与 生活 Agent 分别接入标准化的 MCP 外部工具，避免“工具幻觉”与执行拥堵。
3. **记忆化会话上下文管理**: 基于 `LangGraph Checkpointer` 提供无缝的短时跨越多轮对话的状态留存，随时进行代词指代消解。
4. **精美交互界面**: 现代化极简的 Glassmorphism Web 客户端。

## 🛠️ 技术栈
- **Backend**: Python 3.10+, FastAPI, Uvicorn
- **AI/LLM Framework**: LangGraph, LangChain, LangChain-OpenAI
- **Frontend**: HTML5, Vanilla JS, CSS3
- **Tools**: 基于函数的工具库 (可延伸为 Model Context Protocol 独立 Server)

## 📦 快速部署与运行指北

### 1. 配置密钥环境变量
项目根目录下有 `.env.example` 文件，将其复制一份并重命名为 `.env`，然后填入您真实的 API Key：
```env
OPENAI_API_KEY=your_openai_api_key_here
```

### 2. 依赖安装
推荐使用 Python 的虚拟环境，在终端执行以下命令：
```bash
pip install -r requirements.txt
```

### 3. 本地启动服务
由于我们加入了前端界面，在工程根目录执行以下命令，即可同时拉起后端并托管前台：
```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
服务启动完毕后：
- 👉 **访问极致精美的前端聊天界面**：[http://localhost:8000/](http://localhost:8000/)
- 📄 后端调试与 API Swagger 接口文档：[http://localhost:8000/docs](http://localhost:8000/docs)

## 🔐 开源发布与密钥安全说明

本项目适合公开到 GitHub，但请遵守以下规则：

1. **不要提交真实 API Key**
	- 仓库中只保留 `.env.example`。
	- 使用者应复制 `.env.example` 为 `.env` 并填写自己的密钥。

2. **确保 `.env` 与虚拟环境不入库**
	- 项目已提供 `.gitignore`，默认忽略 `.env`、`.venv/` 等本地文件。

3. **调用说明**
	- 克隆项目后，先执行：`cp .env.example .env`（Windows 可手动复制）。
	- 在 `.env` 中填入 `OPENAI_API_KEY=你的密钥`。
	- 再按上文命令安装依赖并启动服务。

4. **如曾泄露密钥请立刻轮换**
	- 如果真实密钥曾被提交到任何远程仓库，请立即在 OpenAI 控制台撤销并生成新 Key。

## 🧪 演练测试示例
在前端界面的输入框中，大胆输入长难复合指令检测它的本领，例如：
> “我在教三上课，帮我看看有没有空闲教室自习一下，顺便看看中午去哪里吃辣的。”

随后，可以不提及主语，触发长期代词消解的测试能力：
> “那不吃辣的呢，换个清淡的。”
