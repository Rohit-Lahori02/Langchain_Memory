"""
LangChain Long-Term Memory Agent - Main Entry Point
Interactive terminal-based chat with persistent memory across sessions.
"""

import os
import sys
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

from agent import MemoryAgent


def print_welcome():
    """Print welcome message and instructions."""
    console = Console()
    
    welcome_text = """
# 🤖 LangChain Long-Term Memory Agent

Welcome! This AI agent demonstrates **persistent memory** across chat sessions.

## Features
- 💬 **Short-term memory**: Maintains current conversation context
- 💾 **Long-term memory**: Persists conversations across sessions
- 🔍 **Semantic retrieval**: Recalls relevant past interactions
- 🧠 **Context-aware**: References previous conversations when relevant

## Commands
- `/mem` or `/memory` - Inspect current memory state
- `/clear` - Clear short-term memory (current session)
- `/clear long` - Clear long-term memory (persistent)
- `/clear all` - Clear all memory
- `/help` - Show this help message
- `exit` or `quit` - Exit the program

## Usage
Simply type your message and press Enter. The agent will:
1. Retrieve relevant past memories
2. Use them to provide contextual responses
3. Store the interaction for future sessions

---
**Tip**: Try asking the agent to remember something, then restart the program
and ask about it again to see long-term memory in action!
"""
    
    console.print(Panel(Markdown(welcome_text), border_style="bright_blue", padding=(1, 2)))


def print_help():
    """Print help information."""
    console = Console()
    help_text = """
## Available Commands

- `/mem` or `/memory` - Inspect current memory state
- `/clear` - Clear short-term memory (current session)
- `/clear long` - Clear long-term memory (persistent)
- `/clear all` - Clear all memory
- `/help` - Show this help message
- `exit` or `quit` - Exit the program
"""
    console.print(Panel(Markdown(help_text), title="Help", border_style="yellow"))


def main():
    """Main entry point for the interactive chat loop."""
    console = Console()
    
    # Load environment variables
    load_dotenv()
    
    # Check for API key
    if not os.getenv("OPENROUTER_API_KEY"):
        console.print("[bold red]❌ OPENROUTER_API_KEY missing[/bold red]")
        console.print("[yellow]Create a .env file with OPENROUTER_API_KEY=your_key_here[/yellow]")
        console.print("[dim]Get your key from: https://openrouter.ai/keys[/dim]")
        sys.exit(1)
    
    # Print welcome message
    print_welcome()
    
    # Initialize agent
    try:
        agent = MemoryAgent()
    except Exception as e:
        console.print(f"[bold red]❌ Error initializing agent: {e}[/bold red]")
        sys.exit(1)
    
    console.print("\n[bold green]✨ Agent ready! Start chatting...[/bold green]\n")
    
    # Main interaction loop
    while True:
        try:
            # Get user input
            user_input = console.input("[bold cyan]You:[/bold cyan] ").strip()
            
            # Check for empty input
            if not user_input:
                continue
            
            # Handle exit commands
            if user_input.lower() in ["exit", "quit", "bye"]:
                console.print("\n[bold yellow]👋 Goodbye! Your memories have been saved.[/bold yellow]")
                
                # Show session summary
                summary = agent.export_session_summary()
                console.print(Panel(
                    f"[green]Session Duration:[/green] {summary['session_duration']}\n"
                    f"[green]Messages Exchanged:[/green] {summary['messages_in_session']}\n"
                    f"[green]Total Memories:[/green] {summary['long_term_memory_stats']['total_memories']}",
                    title="📊 Session Summary",
                    border_style="green"
                ))
                
                # Gracefully close and ensure memory is saved
                agent.close()
                break
            
            # Handle help command
            if user_input.lower() in ["/help", "help", "?"]:
                print_help()
                continue
            
            # Handle memory inspection
            if user_input.lower() in ["/mem", "/memory"]:
                agent.inspect_memory()
                continue
            
            # Handle clear commands
            if user_input.lower().startswith("/clear"):
                parts = user_input.lower().split()
                if len(parts) == 1:
                    agent.clear_memory("short")
                elif "long" in parts:
                    confirm = console.input("[yellow]⚠️  Clear long-term memory? This cannot be undone. (yes/no):[/yellow] ")
                    if confirm.lower() == "yes":
                        agent.clear_memory("long")
                elif "all" in parts:
                    confirm = console.input("[yellow]⚠️  Clear ALL memory? This cannot be undone. (yes/no):[/yellow] ")
                    if confirm.lower() == "yes":
                        agent.clear_memory("all")
                continue
            
            # Process user input with agent
            console.print()  # Blank line for spacing
            
            with console.status("[bold green]🤔 Thinking...[/bold green]"):
                response = agent.run(user_input)
            
            # Display response
            console.print(Panel(
                response,
                title="🤖 Assistant",
                border_style="green",
                padding=(1, 2)
            ))
            console.print()  # Blank line for spacing
            
        except KeyboardInterrupt:
            console.print("\n\n[yellow]⚠️  Interrupted. Type 'exit' to quit properly.[/yellow]\n")
            continue
        except Exception as e:
            console.print(f"\n[bold red]❌ Error: {e}[/bold red]\n")
            console.print("[dim]If the error persists, check your API key and internet connection.[/dim]\n")


if __name__ == "__main__":
    main()

