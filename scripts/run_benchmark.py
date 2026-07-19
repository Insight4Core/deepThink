import os
import sys
import uuid
import time
import logging
from dotenv import load_dotenv

# 将项目根目录加入 path，以便导入 src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents.graph import build_graph
from src.models.schemas import ReviewFeedback, JudgeOutput
from src.models.llm_factory import invoke_structured_llm
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

TEST_CASES = [
    {
        "id": 1,
        "query": "我想做一个 AI 产品",
        "criteria": "Clarification 能力；是否能问出关键方向问题；是否避免过度追问；是否真正推进任务"
    },
    {
        "id": 2,
        "query": "我现在用 GPT-4 API 做 Agent，但成本太高了，怎么办？",
        "criteria": "Agent 实战理解；tradeoff；是否能提出真实可落地方案"
    },
    {
        "id": 3,
        "query": "我的网站最近越来越慢，但服务器 CPU 不高",
        "criteria": "系统化排障；工程经验；多因素推理能力"
    },
    {
        "id": 4,
        "query": "为什么很多 AI Agent Demo 很惊艳，但真正产品化后用户不用？",
        "criteria": "深度分析；产品思维；是否具备真正 insight"
    },
    {
        "id": 5,
        "query": "PostgreSQL 和 MySQL 怎么选？",
        "criteria": "Tradeoff 能力；场景化分析；是否像真正专家而不是“百科答案”"
    }
]


def judge_answer(query: str, criteria: str, answer: str, clarification_history: list = None):
    """使用 LLM-as-Judge 对最终答案按 criteria 打分。"""
    clarification_section = ""
    if clarification_history:
        clarification_section = "\n【澄清历史】\n" + "\n".join(
            f"{item['role']}: {item['content']}" for item in clarification_history
        ) + "\n"

    prompt = f"""【原问题】
{query}

【评分标准】
{criteria}
{clarification_section}
【最终回答】
{answer}

注意：如果系统已经主动追问，而用户明确拒绝提供更多信息并要求直接给出方案，则不应因为“缺少澄清”而扣分。请重点评价最终回答是否基于合理假设、覆盖关键场景、可落地执行。
"""
    messages = [
        SystemMessage(content="你是一位严格但公正的答案评分专家。请根据【原问题】、【评分标准】和【澄清历史】（如有），对【最终回答】进行 0-10 分打分并给出理由。"),
        HumanMessage(content=prompt),
    ]
    try:
        parsed, _, _ = invoke_structured_llm(messages, JudgeOutput, temperature=0.3)
        return parsed
    except Exception as e:
        logging.warning(f"LLM-as-Judge 打分失败: {e}")
        return JudgeOutput(score=0, reasoning=f"打分失败: {e}", strengths=[], weaknesses=[])


def run_benchmark():
    if not os.environ.get("OPENAI_API_KEY"):
        print("错误：未找到 OPENAI_API_KEY 环境变量，请在根目录 .env 中配置。")
        sys.exit(1)

    print("========================================")
    print("🚀 开始运行 DeepThink 自动化基准测试")
    print(f"当前模型: {os.environ.get('DEEPTHINK_MODEL_NAME', '默认模型')}")
    print("========================================\n")

    app = build_graph()
    max_iters = int(os.environ.get("DEEPTHINK_MAX_ITERATIONS", 2))

    os.makedirs("outputs", exist_ok=True)
    report_path = f"outputs/benchmark_report_{int(time.time())}.md"

    scores = []

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# 🧪 DeepThink Benchmark Report\n\n")
        f.write(f"**测试时间**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**测试模型**: `{os.environ.get('DEEPTHINK_MODEL_NAME', '默认模型')}`\n")
        f.write(f"**最大迭代次数**: `{max_iters}`\n\n")

    for case in TEST_CASES:
        print(f"[{time.strftime('%H:%M:%S')}] 正在运行 Case {case['id']}: {case['query']}")

        thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}

        initial_state = {
            "original_question": case["query"],
            "messages": [],
            "iteration_count": 0,
            "max_iterations": max_iters,
            "clarification_rounds": 0,
            "clarification_history": []
        }

        # 记录运行切片
        clarification_questions = []
        clarified_question = ""
        initial_draft = ""
        review_comments = {}
        final_draft = ""

        def step(input_data):
            nonlocal clarified_question, initial_draft, review_comments, final_draft
            for output in app.stream(input_data, config=config):
                for node_name, state_update in output.items():
                    if node_name == "clarifier":
                        if state_update.get("needs_clarification"):
                            clarification_questions.append(state_update.get("clarification_message", ""))
                        else:
                            clarified_question = state_update.get("clarified_question", "")
                    elif node_name == "generator":
                        initial_draft = state_update.get("current_draft", "")
                    elif node_name == "reviewer":
                        # 记录 Reviewer 的意见，如果是多次修改，这里会被覆盖为最后一次的意见
                        review_comments = state_update.get("review_feedback", {})
                    elif node_name == "reviser":
                        final_draft = state_update.get("current_draft", "")

                    if "current_draft" in state_update:
                        final_draft = state_update.get("current_draft", "")

        # 开始运行
        step(initial_state)

        # 处理挂起 (Human-in-the-loop)
        while True:
            state_snap = app.get_state(config)
            if not state_snap.next:
                break  # 图执行完毕

            if state_snap.next[0] == "ask_human":
                # AI 抛出了问题，我们用脚本自动喂一个“万金油”回答来测试它的抗压和推进能力
                print(f"  -> [挂起] AI 发起了第 {len(clarification_questions)} 次追问，已自动注入模拟回复。")

                # 模拟用户拒绝回答具体细节，强行要求 AI 自己做 Assumption 并推进
                mock_reply = "我是小白，细节我也不太清楚，请你运用你的专家经验，自己做最合理的假设（可以分场景讨论），不要再问我了，直接给我出完整的方案。"
                msg = state_snap.values.get("clarification_message", "")

                history_update = [
                    {"role": "ai", "content": msg},
                    {"role": "human", "content": mock_reply}
                ]
                app.update_state(config, {"clarification_history": history_update}, as_node="ask_human")

                # 恢复执行
                step(None)

        # 获取最终状态中的澄清历史，用于 Judge 更公正地评估
        final_state = app.get_state(config)
        clarification_history = final_state.values.get("clarification_history", [])

        # LLM-as-Judge 自动打分
        judge_result = judge_answer(case["query"], case["criteria"], final_draft, clarification_history)
        scores.append(judge_result.score)

        # 追加写入报告
        with open(report_path, "a", encoding="utf-8") as f:
            f.write(f"## Case {case['id']}: {case['query']}\n")
            f.write(f"- **核心测试能力**: {case['criteria']}\n")
            f.write(f"- **LLM-as-Judge 评分**: **{judge_result.score}/10**\n")
            f.write(f"- **评分理由**: {judge_result.reasoning}\n\n")

            f.write("### 1. 意图澄清阶段 (Clarification)\n")
            if clarification_questions:
                for i, q in enumerate(clarification_questions):
                    f.write(f"**AI 第 {i+1} 轮追问**:\n```text\n{q}\n```\n")
                    f.write(f"*(模拟用户回复：我是小白，细节我也不太清楚，请你运用你的专家经验...)*\n\n")
            f.write(f"**最终重构给下游的问题**:\n```text\n{clarified_question}\n```\n\n")

            f.write("### 2. 初始草稿阶段 (Generator)\n")
            short_draft = initial_draft[:500] + ("\n... [为保持报告整洁，已截断展示前 500 字]" if len(initial_draft) > 500 else "")
            f.write(f"```text\n{short_draft}\n```\n\n")

            f.write("### 3. 多角色审查阶段 (Reviewer)\n")
            if review_comments:
                for role, fb in review_comments.items():
                    if isinstance(fb, ReviewFeedback):
                        f.write(f"**{role}** (评分: {fb.score}/10 | 通过: {'✅' if fb.passed else '❌'}):\n")
                        f.write(f"```text\n{fb.comments}\n```\n\n")
                    else:
                        f.write(f"**{role}**:\n```text\n{fb}\n```\n\n")
            else:
                f.write("无审查意见。\n\n")

            f.write("### 4. 最终输出 (Final Draft)\n")
            f.write(f"{final_draft}\n\n")

            if judge_result.strengths:
                f.write("**优点**: " + "; ".join(judge_result.strengths) + "\n\n")
            if judge_result.weaknesses:
                f.write("**不足**: " + "; ".join(judge_result.weaknesses) + "\n\n")

            f.write("---\n\n")

        print(f"  -> ✅ Case {case['id']} 运行完毕！Judge 评分: {judge_result.score}/10")

    # 写入汇总
    avg_score = sum(scores) / len(scores) if scores else 0
    with open(report_path, "a", encoding="utf-8") as f:
        f.write("# 📊 综合得分\n\n")
        f.write(f"**平均得分**: {avg_score:.1f}/10\n\n")
        f.write("| Case | 评分 |\n|------|------|\n")
        for case, score in zip(TEST_CASES, scores):
            f.write(f"| {case['id']}. {case['query']} | {score}/10 |\n")
        f.write("\n")

    print(f"\n🎉 所有测试运行完毕！平均得分: {avg_score:.1f}/10")
    print(f"📄 报告已生成至: {report_path}")


if __name__ == "__main__":
    run_benchmark()
