import os
import sys
import time
import uuid
import logging
import markdown
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

from src.agents.graph import build_graph
from src.models.schemas import ReviewFeedback

# 尝试加载环境变量
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

console = Console()

def main():
    console.print(Panel.fit("🧠 欢迎使用 DeepThink (Agentic LLM Enhancer)", style="bold blue"))
    
    if not os.environ.get("OPENAI_API_KEY"):
        console.print("[bold red]错误：未找到 OPENAI_API_KEY 环境变量。[/]")
        console.print("请创建一个 .env 文件并设置 OPENAI_API_KEY (以及可选的 OPENAI_API_BASE 和 DEEPTHINK_MODEL_NAME)。")
        sys.exit(1)
        
    model_name = os.environ.get("DEEPTHINK_MODEL_NAME", "gpt-3.5-turbo")
    console.print(f"当前配置模型: [bold green]{model_name}[/]\n")
    
    # 编译图
    app = build_graph()
    
    while True:
        try:
            user_input = console.input("\n[bold cyan]请输入您的问题 (输入 'q' 退出): [/]")
            if user_input.lower() in ['q', 'quit', 'exit']:
                break
                
            if not user_input.strip():
                continue
                
            # 读取配置的最大迭代次数
            max_iters = int(os.environ.get("DEEPTHINK_MAX_ITERATIONS", 2))
            
            # 使用随机 uuid 生成 thread_id，保证每次图的状态相互隔离
            thread_id = str(uuid.uuid4())
            config = {"configurable": {"thread_id": thread_id}}
            
            # 初始状态
            initial_state = {
                "original_question": user_input,
                "messages": [],
                "iteration_count": 0,
                "max_iterations": max_iters,
                "clarification_rounds": 0,
                "clarification_history": []
            }
            
            console.print("\n[bold yellow]开始处理...[/]")
            
            final_draft = ""
            final_token_usage = {}
            
            def run_stream(input_data):
                """运行或者恢复流的内部函数"""
                nonlocal final_draft, final_token_usage
                try:
                    with console.status("[bold cyan]Agent 正在努力思考和生成中，这可能需要几十秒的时间，请耐心等待...[/]") as status:
                        for output in app.stream(input_data, config=config):
                            for node_name, state_update in output.items():
                                if node_name == "ask_human":
                                    continue # 忽略占位节点
                                    
                                console.print(f"\n[bold magenta]>> 当前节点执行完毕: {node_name} <<[/]")
                                
                                if "current_draft" in state_update:
                                    final_draft = state_update["current_draft"]
                                    
                                if "token_usage" in state_update:
                                    final_token_usage = state_update["token_usage"]
                                    
                                if node_name == "clarifier":
                                    if state_update.get("needs_clarification"):
                                        status.update("[bold yellow]发现问题不够清晰，准备向您提问...[/]")
                                    else:
                                        console.print("[dim]意图澄清完成（问题已足够清晰）：[/dim]")
                                        console.print(state_update.get("clarified_question"))
                                        status.update("[bold green]意图澄清完成！正在撰写初始回答草稿...[/]")
                                        
                                elif node_name == "generator":
                                    console.print("[dim]生成初始草稿完毕。[/dim]")
                                    status.update("[bold yellow]初稿完成！多个 Reviewer 正在从不同角度进行严格审查...[/]")
                                    
                                elif node_name == "reviewer":
                                    console.print("[dim]审查意见:[/dim]")
                                    feedback = state_update.get("review_feedback", {})
                                    for role, fb in feedback.items():
                                        if isinstance(fb, ReviewFeedback):
                                            short_fb = fb.comments[:80] + "..." if len(fb.comments) > 80 else fb.comments
                                            console.print(f"  - {role}: 评分 {fb.score}/10 | {short_fb}")
                                        else:
                                            short_fb = str(fb)[:80] + "..." if len(str(fb)) > 80 else str(fb)
                                            console.print(f"  - {role}: {short_fb}")
                                    if state_update.get("is_approved"):
                                        console.print("[bold green]✅ 所有 Reviewer 均通过！[/]")
                                    else:
                                        console.print("[bold red]❌ 发现问题，发回重写...[/]")
                                        status.update("[bold magenta]Review 不通过！Reviser 正在根据意见重写回答...[/]")
                                        
                                elif node_name == "reviser":
                                    console.print("[dim]根据审查意见重写完毕。[/dim]")
                                    status.update("[bold yellow]重写完成！正在重新提交给 Reviewer 审查...[/]")
                except KeyboardInterrupt:
                    console.print("\n[bold yellow]⚠️ 已手动中断当前的思考和生成任务！[/bold yellow]")
                    return False
                return True

            # 首次执行
            if not run_stream(initial_state):
                continue
                
            # 处理 Human-in-the-loop 的挂起状态
            while True:
                state_snap = app.get_state(config)
                # 如果图没有 next 节点，说明跑完到 END 了
                if not state_snap.next:
                    break
                    
                # 如果是停在了 ask_human，说明等待用户输入
                if state_snap.next[0] == "ask_human":
                    current_state = state_snap.values
                    msg = current_state.get("clarification_message", "")
                    
                    console.print(f"\n[bold yellow]🤖 AI 分析专家请求进一步澄清:[/bold yellow]")
                    console.print(Panel(msg, title="追问详情", border_style="yellow"))
                    
                    user_ans = console.input("\n[bold cyan]请您补充信息 (输入 'q' 取消提问，强行让 AI 生成): [/]")
                    
                    if user_ans.lower() in ['q', 'quit', 'exit']:
                        user_ans = "用户拒绝提供更多信息，请直接根据现有信息生成最好的重构问题。"
                        
                    # 通过 update_state 作为 ask_human 节点，将用户的回答补充进 state 中
                    history_update = [
                        {"role": "ai", "content": msg},
                        {"role": "human", "content": user_ans}
                    ]
                    app.update_state(config, {"clarification_history": history_update}, as_node="ask_human")
                    
                    # 恢复图的执行
                    if not run_stream(None):
                        break
            
            # 如果正常结束并没有被中断，输出最终答案
            if not final_draft:
                continue
                
            console.print("\n[bold cyan]============ 最终回答 =============[/]")
            console.print(Markdown(final_draft))
            console.print("[bold cyan]===================================[/]\n")
            
            if final_token_usage:
                p_tokens = final_token_usage.get('prompt_tokens', 0)
                c_tokens = final_token_usage.get('completion_tokens', 0)
                t_tokens = final_token_usage.get('total_tokens', 0)
                console.print(f"[dim]📊 本轮 Token 消耗: 提示词 {p_tokens} | 生成 {c_tokens} | 总计 [bold green]{t_tokens}[/bold green][/dim]\n")
                
            # 保存为 HTML 文件
            if final_draft:
                html_content = markdown.markdown(final_draft, extensions=['extra', 'codehilite'])
                full_html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>DeepThink Answer</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; line-height: 1.6; max-width: 900px; margin: 0 auto; padding: 2rem; color: #333; }}
        pre {{ background-color: #f6f8fa; padding: 16px; border-radius: 6px; overflow: auto; }}
        code {{ font-family: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, "Liberation Mono", monospace; background-color: #f6f8fa; padding: 0.2em 0.4em; border-radius: 3px; }}
        pre code {{ background-color: transparent; padding: 0; }}
        h1, h2, h3, h4 {{ border-bottom: 1px solid #eaecef; padding-bottom: .3em; margin-top: 24px; }}
        blockquote {{ border-left: .25em solid #dfe2e5; color: #6a737d; padding: 0 1em; margin: 0; }}
        .question-box {{ background-color: #e1f5fe; padding: 15px; border-radius: 8px; margin-bottom: 30px; border-left: 5px solid #03a9f4; }}
    </style>
</head>
<body>
    <div class="question-box">
        <strong>原始问题：</strong><br/>
        {user_input}
    </div>
    {html_content}
</body>
</html>"""
                os.makedirs("outputs", exist_ok=True)
                timestamp = int(time.time())
                filename = f"outputs/answer_{timestamp}.html"
                with open(filename, "w", encoding="utf-8") as f:
                    f.write(full_html)
                console.print(f"💾 [bold green]本次生成的答案已永久保存至：[underline]{filename}[/underline][/bold green]\n")
                
        except KeyboardInterrupt:
            console.print("\n[dim]退出程序，再见！[/dim]")
            break
        except Exception as e:
            console.print(f"[bold red]发生错误: {e}[/]")

if __name__ == "__main__":
    main()
