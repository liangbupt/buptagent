import sys
import os
import json

# Add parent dir to path so we can import app modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.rag.campus_rag import campus_rag_retriever
from app.core.graph import build_graph
from app.core.config import settings

# 模拟黄金测试集 (Golden Test Set)
GOLDEN_TEST_SET = [
    {
        "query": "教三的空教室一般几点释放？",
        "expected_intent": "教务",
        "expected_facts": ["整点后 10 分钟", "101-305"]
    },
    {
        "query": "预算 30 块钱，中午去哪吃？",
        "expected_intent": "生活",
        "expected_facts": ["馨园一层", "套餐窗口"]
    }
]

def mock_llm_as_a_judge(query, generated_response, expected_facts):
    """
    模拟使用大模型作为裁判 (LLM-as-a-Judge) 来评估忠实度和涵盖率。
    在真实的工业级项目中，这部分会调用 GPT-4 配合 Prompt 输出 JSON 打分。
    """
    score = 100
    missing = []
    for fact in expected_facts:
        if fact not in generated_response:
            score -= 50
            missing.append(fact)
    
    return {
        "faithfulness": 100 if "根据" in generated_response or "知识库" in generated_response else 80, 
        "recall_score": max(0, score),
        "missing_facts": missing
    }

def run_evaluation():
    print("🚀 开始执行自动化评测管线 (LLM-as-a-Judge)...\n")
    
    graph = build_graph(
        api_key=settings.OPENAI_API_KEY, 
        base_url=settings.OPENAI_BASE_URL, 
        model=settings.LLM_MODEL
    )
    
    total_recall = 0
    total_faithfulness = 0

    for idx, test_case in enumerate(GOLDEN_TEST_SET, 1):
        print(f"▶ 正在测试 Case {idx}: {test_case['query']}")
        
        # 1. 测试 RAG 召回
        rag_hits = campus_rag_retriever.retrieve_with_explanations(test_case["query"], top_k=2)
        context = "\n".join([h["content"] for h in rag_hits])
        
        # 2. 测试 Graph 生成
        inputs = {"messages": [("user", test_case["query"])]}
        config = {"configurable": {"thread_id": f"eval_user_{idx}"}}
        result = graph.invoke(inputs, config=config)
        generated_reply = result["messages"][-1].content
        
        # 3. LLM-as-a-Judge 评估
        eval_metrics = mock_llm_as_a_judge(test_case["query"], generated_reply + context, test_case["expected_facts"])
        
        print(f"   [指标] 召回得分 (Recall): {eval_metrics['recall_score']}")
        print(f"   [指标] 忠实度 (Faithfulness): {eval_metrics['faithfulness']}")
        if eval_metrics['missing_facts']:
            print(f"   [警告] 缺失知识点: {eval_metrics['missing_facts']}")
        print("-" * 50)
        
        total_recall += eval_metrics['recall_score']
        total_faithfulness += eval_metrics['faithfulness']
        
    avg_recall = total_recall / len(GOLDEN_TEST_SET)
    avg_faith = total_faithfulness / len(GOLDEN_TEST_SET)
    
    print("✅ 自动化评测完成！")
    print(f"   👉 平均召回率 (Avg Recall): {avg_recall}")
    print(f"   👉 平均忠实度 (Avg Faithfulness): {avg_faith}")

if __name__ == "__main__":
    run_evaluation()