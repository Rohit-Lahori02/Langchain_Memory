"""
AI Agent with Integrated Short-Term and Long-Term Memory
Demonstrates how conversation history interacts with persistent knowledge.
"""

import os
from typing import List, Dict, Any
from datetime import datetime

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from memory_store import LongTermMemoryStore


class MemoryAgent:
    """
    AI Agent with dual-memory system:
    - Short-term: Current conversation context (session memory)
    - Long-term: Persistent memory across sessions (vector store)
    """
    
    def __init__(self, model_name: str = "anthropic/claude-3-haiku", 
                 temperature: float = 0.7,
                 max_short_term_messages: int = 10):
        """
        Initialize the memory-enabled agent.
        
        Args:
            model_name: LLM model to use (via OpenRouter)
            temperature: Sampling temperature for responses
            max_short_term_messages: Maximum messages to keep in short-term memory
        """
        self.console = Console()
        
        # Initialize LLM via OpenRouter
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
            timeout=60,            # avoid hanging forever on router issues
            max_retries=3,         # handle transient router hiccups
            default_headers={
                "HTTP-Referer": "http://localhost:3000",
                "X-Title": "LangChain Memory Agent"
            }
        )
        
        # Short-term memory (current session)
        self.short_term_memory: List[Any] = []
        self.max_short_term_messages = max_short_term_messages
        
        # Long-term memory (persistent)
        self.long_term_memory = LongTermMemoryStore()
        self.long_term_memory.new_session()  # Increment session counter
        
        # Session metadata
        self.session_start = datetime.now()
        self.message_count = 0
        
        self.console.print("[bold green]🤖 Memory Agent Initialized[/bold green]")
        self._print_memory_stats()
    
    def _print_memory_stats(self):
        """Print current memory statistics."""
        stats = self.long_term_memory.get_stats()
        self.console.print(f"[dim]📊 Long-term memories: {stats['total_memories']} | "
                          f"Session messages: {len(self.short_term_memory)}[/dim]")
    
    def _build_context_with_retrieved_memories(self, user_input: str) -> str:
        """
        Retrieve relevant long-term memories and build context string.
        
        Args:
            user_input: Current user message
            
        Returns:
            Formatted context string with retrieved memories
        """
        # Retrieve relevant memories from long-term store
        relevant_memories = self.long_term_memory.retrieve_relevant_memories(
            user_input, k=3
        )
        
        if not relevant_memories:
            return ""
        
        # Format retrieved memories into context
        context_parts = ["=== RELEVANT PAST MEMORIES ==="]
        for i, memory in enumerate(relevant_memories, 1):
            timestamp = memory.get("timestamp", "Unknown time")
            user_msg = memory.get("user_message", "")
            assistant_msg = memory.get("assistant_response", "")
            
            context_parts.append(f"\nMemory {i} (from {timestamp}):")
            context_parts.append(f"User: {user_msg}")
            context_parts.append(f"Assistant: {assistant_msg}")
        
        context_parts.append("\n=== END OF RETRIEVED MEMORIES ===\n")
        
        # Show user what was retrieved
        self.console.print(f"[dim]🔍 Retrieved {len(relevant_memories)} relevant memories[/dim]")
        
        return "\n".join(context_parts)
    
    def _create_system_prompt(self, retrieved_context: str = "") -> SystemMessage:
        """Create the system prompt with optional retrieved memories."""
        base_instruction = """You are a helpful AI assistant with memory capabilities. 

You have access to:
1. SHORT-TERM MEMORY: The current conversation history
2. LONG-TERM MEMORY: Past conversations and interactions

When relevant memories are provided, use them to:
- Recall previous discussions and context
- Maintain consistency with past interactions
- Reference earlier conversations naturally
- Build upon previous knowledge about the user

Be conversational, helpful, and demonstrate that you remember past interactions when relevant."""

        if retrieved_context:
            full_prompt = f"{base_instruction}\n\n{retrieved_context}"
        else:
            full_prompt = base_instruction
        
        return SystemMessage(content=full_prompt)
    
    def _manage_short_term_memory(self):
        """Manage short-term memory size to prevent context overflow."""
        # Keep only the last N pairs of messages (2N messages total: N user + N assistant)
        if len(self.short_term_memory) > self.max_short_term_messages * 2:
            # Keep last N pairs of messages (user + assistant)
            self.short_term_memory = self.short_term_memory[-(self.max_short_term_messages * 2):]
    
    def run(self, user_input: str) -> str:
        """
        Process user input and generate response using dual-memory system.
        
        Args:
            user_input: The user's message
            
        Returns:
            The assistant's response
        """
        self.message_count += 1
        
        # Step 1: Retrieve relevant long-term memories
        retrieved_context = self._build_context_with_retrieved_memories(user_input)
        
        # Step 2: Build system prompt with retrieved context
        system_message = self._create_system_prompt(retrieved_context)
        
        # Step 3: Construct messages with short-term memory
        messages = [system_message] + self.short_term_memory + [HumanMessage(content=user_input)]
        
        # Step 4: Generate response
        try:
            response = self.llm.invoke(messages)
            assistant_message = response.content
        except Exception as e:
            assistant_message = f"Error generating response: {str(e)}"
            self.console.print(f"[bold red]❌ Error: {e}[/bold red]")
        
        # Step 5: Update short-term memory
        self.short_term_memory.append(HumanMessage(content=user_input))
        self.short_term_memory.append(AIMessage(content=assistant_message))
        self._manage_short_term_memory()
        
        # Step 6: Store in long-term memory
        self.long_term_memory.add_memory(
            user_message=user_input,
            assistant_response=assistant_message,
            additional_metadata={
                "session_start": self.session_start.isoformat(),
                "message_number": self.message_count
            }
        )
        
        return assistant_message
    
    def inspect_memory(self):
        """Display current memory state for debugging/inspection."""
        self.console.print("\n[bold cyan]🧠 MEMORY INSPECTION[/bold cyan]")
        
        # Short-term memory
        self.console.print(Panel(
            f"[yellow]Current Session Messages:[/yellow] {len(self.short_term_memory)}\n"
            f"[yellow]Session Duration:[/yellow] {datetime.now() - self.session_start}",
            title="📝 Short-Term Memory (Current Session)",
            border_style="cyan"
        ))
        
        # Long-term memory stats
        stats = self.long_term_memory.get_stats()
        self.console.print(Panel(
            f"[green]Total Stored Memories:[/green] {stats['total_memories']}\n"
            f"[green]Storage Location:[/green] {stats['persist_path']}",
            title="💾 Long-Term Memory (Persistent)",
            border_style="green"
        ))
        
        # Show recent memories
        memories = self.long_term_memory.get_all_memories()
        if memories:
            table = Table(title="Recent Memories (Last 5)", show_header=True, header_style="bold magenta")
            table.add_column("Time", style="dim")
            table.add_column("User", style="cyan")
            table.add_column("Assistant", style="green")
            
            for memory in memories[-5:]:
                timestamp = memory.get("timestamp", "")[:19]  # Truncate ISO timestamp
                user_msg = memory.get("user_message", "")
                assistant_msg = memory.get("assistant_response", "")
                
                # Only add "..." if string is actually longer than the limit
                if len(user_msg) > 50:
                    user_msg = user_msg[:50] + "..."
                if len(assistant_msg) > 50:
                    assistant_msg = assistant_msg[:50] + "..."
                
                table.add_row(timestamp, user_msg, assistant_msg)
            
            self.console.print(table)
        
        self.console.print()
    
    def clear_memory(self, memory_type: str = "all"):
        """
        Clear memory based on type.
        
        Args:
            memory_type: "short", "long", or "all"
        """
        if memory_type in ["short", "all"]:
            self.short_term_memory = []
            self.console.print("[yellow]🧹 Short-term memory cleared![/yellow]")
        
        if memory_type in ["long", "all"]:
            self.long_term_memory.clear_all_memories()
        
        if memory_type == "all":
            self.message_count = 0
            self.session_start = datetime.now()
        
        self._print_memory_stats()
    
    def close(self):
        """Gracefully close the agent and ensure all memory is persisted."""
        try:
            self.long_term_memory.save()
        except Exception:
            pass
    
    def export_session_summary(self) -> Dict[str, Any]:
        """Export summary of current session."""
        return {
            "session_start": self.session_start.isoformat(),
            "session_duration": str(datetime.now() - self.session_start),
            "messages_in_session": self.message_count,
            "short_term_memory_size": len(self.short_term_memory),
            "long_term_memory_stats": self.long_term_memory.get_stats()
        }

