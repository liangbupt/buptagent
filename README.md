# 🌟 BUPT Campus Smart Life Assistant Agent

> **北邮校园智能生活助手** —— 融合 LangGraph 循环状态图、混合双路 RAG、本地化分层记忆治理与 MCP 标准协议的校园级综合性 AI Agent 平台。

## 🎯 项目定位与核心价值

本项目旨在为高校师生提供一个**统一的自然语言交互入口**，解决教务查询、食堂推荐、蹭课二手等校园场景中信息碎片化的问题。从工程角度全面告别“单体黑盒大模型”与“花哨重资产框架”，以**极简、轻量、高可控**为设计理念，落地了首字秒出的流式响应、完全可释的混合检索增强、极低开销的记忆过滤与严格解耦的工具调用网络。

## 🏗️ 核心架构拆解

```mermaid
flowchart TD
    subgraph 接入层 (API & Stream)
        UI[前端交互界面 / Web UI]
        SSE[FastAPI SSE端点<br>astream_events 发动机]
    end

    subgraph 记忆与缓存管控 (Hybrid Memory)
        SCache[(语义缓存池<br>短路拦截)]
        STM[(短期工作记忆<br>Redis滑动窗口)]
        LTM[(长期偏好记忆<br>Chroma哈希查重)]
        Heuristics[规则启发式打分<br>毫秒级过滤提取]
    end

    subgraph 本地化混合 RAG 系统
        RRF[RRF 倒数秩融合重排]
        TFIDF[手写 TF-IDF<br>逆文档匹配]
        BGE[BGE-large-zh<br>余弦相似度检索]
        KB[(高度受控校园 JSON库<br>Bigram分词切割)]
    end

    subgraph 智能决策与控制网 (LangGraph)
        Supervisor[总控路由器<br>带有回路与动态监控]
        W1[Academic Agent]
        W2[Life Agent]
        W3[Interaction Agent]
    end

    subgraph 本地解耦工具层 (FastMCP)
        Client[MCP Stdio Client<br>本地安全子进程通信]
        Server[MCP Server<br>完全脱离大模型绑定]
        Provider[ToolDataProvider<br>兜底防脏机制]
    end

    UI <-->|HTTP /流式 SSE| SSE
    SSE --> SCache
    SCache -- 未命中 --> STM & Heuristics
    Heuristics --> |权重阈值提取| LTM
    STM --> Supervisor
    Supervisor --> |路由派发| W1 & W2 & W3
    W1 & W2 & W3 --> |碰壁观测回流| Supervisor
    W1 & W2 & W3 -.-> |指令请求| Client
    Client <--> Server <--> Provider
    W1 & W2 & W3 <--> RRF
    RRF --> TFIDF & BGE
    TFIDF & BGE --> KB
```

## ✨ 核心亮点特性

1. **极速的 TTFT 体验与流式响应 (LangGraph V2 + SSE)**
   - 利用 FastAPI 的 `StreamingResponse` 搭配 LangGraph `astream_events` 发动机，实现毫秒级首字时延。
   - 提前向客户端推送 RAG 引用的来源（`rag_hits`），极大填补用户的视觉等待空白。
   - 搭载 **Semantic Cache (语义缓存池)** 短路器，遇到高频同义问题即刻拦截免过 LLM 退回缓存应答。

2. **抗污染的轻量分层记忆管理 (Hybrid Memory Manager)**
   - 摒弃极易阻塞超时的 Celery 大模型扫库，改为基于本地加权打分规则 (`score_long_term_signal`) 进行毫秒级长期偏好识别。
   - 长期记忆落盘前依靠 `_text_hash` 精准比对去重，免除同义语义爆炸；短期记忆用 Redis 原生 `ltrim` 切断窗口，免除 OOM 数据灾难。

3. **双路融合可解释检索 (Custom Hybrid RAG Pipeline)**
   - 放弃效果模糊粗暴的文字递归切分与 OCR，转而针对中文与校园习惯编写 **Bigram 双字滑动切分** 和纯结构化管理。
   - 在 BGE 级密度向量基础上加入了自编底层的 **TF-IDF 倒排统分逻辑**，强行压制实体编号错乱引起的向量距离“近义误判”，尾部配合 RRF 平滑融合并追溯 `source_id`。

4. **交叉校核与彻底的工具解耦 (Cross-Check & MCP Protocol)**
   - 在 LangGraph 中搭建具有“打回边 (Feedback Loop)”的拓扑 (`workflow.add_edge("agent", "supervisor")`)。子节点一旦工具请求遇挫，立刻返回将意图推回，由 Supervisor 再次调整线路。
   - 移除传统项目单体绑定的外部工具调用方式，挂载原生 MCP (Model Context Protocol)。子进程化执行，自带本地模拟器 `Provider` 进行错误隔离无缝兜底。

5. **LLM-as-a-Judge 自闭环自动化评测架构**
   - 建立自有 `Golden Test Set`（黄金测评准线）与 `evaluate_agent.py` 测试脚本，通过上帝视角裁量打分。
   - 实现包含 `Recall Score` 与无幻觉指数 `Faithfulness` 的严格度量输出。

## 🚀 快速上手

### 1) 环境与依赖
```bash
pip install -r requirements.txt
cp .env.example .env
```
在 `.env` 中按需填写 `OPENAI_API_KEY`、`OPENAI_BASE_URL` 以及对应的大语言模型名称。

### 2) 启动服务
```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
- **Web UI**: http://localhost:8000/
- **API Docs**: http://localhost:8000/docs
- 支持在前端侧边栏动态配置 API Key 与模型网关参数，实现不同环境的一键接入。

## 🛡️ 开源与工程安全
- 密钥级配置通过 `.env` 与前端本地 `localStorage` 隔离，确保安全传输。
- MCP 及外层路由均做了完善的 Fallback 回落机制，杜绝脏状态传入下级流。
- ChromaDB 等向量底座与 SQlite 被放置为免部署文件级读写，零成本部署。
