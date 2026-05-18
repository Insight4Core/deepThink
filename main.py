import os
import sys
import time
import markdown
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

from src.agents.graph import build_graph

# 尝试加载环境变量
load_dotenv()

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
                
            # 读取配置的最大迭代次数（默认改为 2，避免等太久）
            max_iters = int(os.environ.get("DEEPTHINK_MAX_ITERATIONS", 2))
            
            # 初始状态
            initial_state = {
                "original_question": user_input,
                "messages": [],
                "iteration_count": 0,
                "max_iterations": max_iters
            }
            
            console.print("\n[bold yellow]开始处理...[/]")
            
            final_draft = ""
            final_token_usage = {}
            
            # 使用 stream 来展示各个节点的进度
            try:
                with console.status("[bold cyan]Agent 正在努力思考和生成中，由于大模型需要生成大量内容，这可能需要几十秒的时间，请耐心等待...[/]") as status:
                    for output in app.stream(initial_state):
                        for node_name, state_update in output.items():
                            console.print(f"\n[bold magenta]>> 当前节点执行完毕: {node_name} <<[/]")
                            
                            if "current_draft" in state_update:
                                final_draft = state_update["current_draft"]
                                
                            if "token_usage" in state_update:
                                final_token_usage = state_update["token_usage"]
                                
                            if node_name == "clarifier":
                                console.print("[dim]意图澄清结果:[/dim]")
                                console.print(state_update.get("clarified_question"))
                                status.update("[bold green]意图澄清完成！正在撰写初始回答草稿...[/]")
                                
                            elif node_name == "generator":
                                console.print("[dim]生成初始草稿完毕。[/dim]")
                                status.update("[bold yellow]初稿完成！多个 Reviewer 正在从不同角度进行严格审查...[/]")
                                
                            elif node_name == "reviewer":
                                console.print("[dim]审查意见:[/dim]")
                                feedback = state_update.get("review_feedback", {})
                                for role, fb in feedback.items():
                                    # 截断展示，避免太长
                                    short_fb = fb[:100] + "..." if len(fb) > 100 else fb
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
            break
        except Exception as e:
            console.print(f"[bold red]发生错误: {e}[/]")

if __name__ == "__main__":
    main()
