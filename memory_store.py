"""
Long-Term Memory Store using FAISS Vector Database
Handles persistent storage and semantic retrieval of conversation history.
"""

import os
import pickle
from typing import List, Dict, Any
from datetime import datetime

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document


class LongTermMemoryStore:
    """
    Manages long-term memory storage using FAISS vector database.
    Stores conversation exchanges and retrieves relevant past context.
    """
    
    def __init__(self, embeddings_model: str = "all-MiniLM-L6-v2", 
                 persist_path: str = "./memory_data"):
        """
        Initialize the long-term memory store.
        
        Args:
            embeddings_model: HuggingFace embedding model to use (runs locally, free!)
            persist_path: Directory path to persist memory data
        """
        self.persist_path = persist_path
        self.vector_store_path = os.path.join(persist_path, "faiss_index")
        self.metadata_path = os.path.join(persist_path, "metadata.pkl")
        
        # Initialize LOCAL embeddings (no API calls, free!)
        # First run will download the model (~80MB), then cached locally
        print("🔧 Initializing local embeddings model (first run downloads ~80MB)...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embeddings_model,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Create persist directory if it doesn't exist
        os.makedirs(persist_path, exist_ok=True)
        
        # Load or initialize vector store
        self.vector_store = self._load_or_create_vector_store()
        self.metadata = self._load_metadata()
        
    def _load_or_create_vector_store(self) -> FAISS:
        """Load existing vector store or create a new one."""
        if os.path.exists(self.vector_store_path):
            try:
                print("📂 Loading existing long-term memory...")
                # allow_dangerous_deserialization=True required for FAISS pickle files
                # ONLY load indices you trust (pickle can execute arbitrary code)
                return FAISS.load_local(
                    self.vector_store_path, 
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
            except Exception as e:
                print(f"⚠️  Could not load existing memory: {e}")
                print("🆕 Creating new memory store...")
                return self._create_empty_vector_store()
        else:
            print("🆕 Creating new long-term memory store...")
            return self._create_empty_vector_store()
    
    def _create_empty_vector_store(self) -> FAISS:
        """Create an empty FAISS vector store with a dummy document."""
        dummy_doc = Document(
            page_content="This is the beginning of memory.",
            metadata={"type": "system", "timestamp": datetime.now().isoformat()}
        )
        return FAISS.from_documents([dummy_doc], self.embeddings)
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load metadata about stored memories."""
        if os.path.exists(self.metadata_path):
            try:
                with open(self.metadata_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"⚠️  Could not load metadata: {e}")
                return {"total_memories": 0, "session_count": 0}
        return {"total_memories": 0, "session_count": 0}
    
    def _save_metadata(self):
        """Save metadata to disk."""
        try:
            with open(self.metadata_path, 'wb') as f:
                pickle.dump(self.metadata, f)
        except Exception as e:
            print(f"⚠️  Could not save metadata: {e}")
    
    def add_memory(self, user_message: str, assistant_response: str, 
                   additional_metadata: Dict[str, Any] = None):
        """
        Add a conversation exchange to long-term memory.
        
        Args:
            user_message: The user's input
            assistant_response: The assistant's response
            additional_metadata: Optional additional metadata to store
        """
        # Don't store empty noise
        if not (user_message or "").strip() and not (assistant_response or "").strip():
            return
        
        timestamp = datetime.now().isoformat()
        
        # Create a combined text for better semantic search
        combined_text = f"User: {user_message}\nAssistant: {assistant_response}"
        
        metadata = {
            "user_message": user_message,
            "assistant_response": assistant_response,
            "timestamp": timestamp,
            "type": "conversation"
        }
        
        if additional_metadata:
            metadata.update(additional_metadata)
        
        # Create document
        doc = Document(
            page_content=combined_text,
            metadata=metadata
        )
        
        # Add to vector store
        self.vector_store.add_documents([doc])
        
        # Update metadata
        self.metadata["total_memories"] = self.metadata.get("total_memories", 0) + 1
        
        # Persist to disk
        self.save()
        
    def retrieve_relevant_memories(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieve the most relevant past memories based on semantic similarity.
        Uses MMR (Maximal Marginal Relevance) to avoid redundant results.
        
        Args:
            query: The current user input to find relevant memories for
            k: Number of memories to retrieve
            
        Returns:
            List of relevant memory dictionaries
        """
        try:
            # Use MMR retriever for better quality and diversity
            retriever = self.vector_store.as_retriever(
                search_type="mmr",
                search_kwargs={"k": max(5, k)}  # retrieve a bit more, return top k below
            )
            docs = retriever.get_relevant_documents(query)[:k]
            
            memories = []
            for doc in docs:
                # Skip system messages and dummy documents
                if doc.metadata.get("type") == "system":
                    continue
                    
                memories.append({
                    "user_message": doc.metadata.get("user_message", ""),
                    "assistant_response": doc.metadata.get("assistant_response", ""),
                    "timestamp": doc.metadata.get("timestamp", ""),
                    "content": doc.page_content
                })
            
            return memories
        except Exception as e:
            print(f"⚠️  Error retrieving memories: {e}")
            return []
    
    def get_all_memories(self) -> List[Dict[str, Any]]:
        """Return a sample of stored memories via similarity to a blank query.
        
        Note: this is NOT exhaustive; use an external JSONL log for a full listing.
        FAISS similarity search with empty query returns a sample, not all documents.
        """
        try:
            # Get sample of documents from the vector store
            k = max(1, self.metadata.get("total_memories", 100))
            all_docs = self.vector_store.similarity_search("", k=k)
            
            memories = []
            for doc in all_docs:
                if doc.metadata.get("type") == "system":
                    continue
                memories.append({
                    "user_message": doc.metadata.get("user_message", ""),
                    "assistant_response": doc.metadata.get("assistant_response", ""),
                    "timestamp": doc.metadata.get("timestamp", "")
                })
            
            # Best-effort sort by timestamp
            memories.sort(key=lambda m: m.get("timestamp", ""))
            
            return memories
        except Exception as e:
            print(f"⚠️  Error getting all memories: {e}")
            return []
    
    def clear_all_memories(self):
        """Clear all stored memories and reset the vector store."""
        print("🗑️  Clearing all long-term memories...")
        self.vector_store = self._create_empty_vector_store()
        self.metadata = {"total_memories": 0, "session_count": 0}
        self.save()
        print("✅ Long-term memory cleared!")
    
    def save(self):
        """Persist the vector store and metadata to disk."""
        try:
            self.vector_store.save_local(self.vector_store_path)
            self._save_metadata()
        except Exception as e:
            print(f"⚠️  Error saving memory: {e}")
    
    def new_session(self):
        """Increment session counter. Call once per agent initialization."""
        self.metadata["session_count"] = self.metadata.get("session_count", 0) + 1
        self._save_metadata()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the memory store."""
        return {
            "total_memories": self.metadata.get("total_memories", 0),
            "session_count": self.metadata.get("session_count", 0),
            "persist_path": self.persist_path
        }

