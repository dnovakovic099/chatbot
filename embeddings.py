"""
Embeddings and Vector Database Module.
Provides semantic search capabilities using ChromaDB and OpenAI embeddings.
"""

import json
import hashlib
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass

import chromadb
from chromadb.config import Settings
from openai import OpenAI

from config import settings


# Initialize OpenAI client for embeddings
openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)

# Initialize ChromaDB with persistent storage
chroma_client = chromadb.PersistentClient(
    path="./chroma_db",
    settings=Settings(anonymized_telemetry=False)
)


# ============ EMBEDDING GENERATION ============

def generate_embedding(text: str) -> List[float]:
    """
    Generate an embedding vector for the given text using OpenAI.
    
    Args:
        text: The text to embed
        
    Returns:
        List of floats representing the embedding vector
    """
    if not text or not text.strip():
        # Return zero vector for empty text
        return [0.0] * 1536  # text-embedding-3-small dimension
    
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=text[:8000]  # Limit to avoid token limits
    )
    return response.data[0].embedding


def generate_embeddings_batch(texts: List[str]) -> List[List[float]]:
    """
    Generate embeddings for multiple texts in a single API call.
    
    Args:
        texts: List of texts to embed
        
    Returns:
        List of embedding vectors
    """
    if not texts:
        return []
    
    # Filter and truncate
    processed_texts = [t[:8000] if t and t.strip() else " " for t in texts]
    
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=processed_texts
    )
    
    return [item.embedding for item in response.data]


# ============ COLLECTION MANAGEMENT ============

def get_or_create_collection(name: str) -> chromadb.Collection:
    """Get or create a ChromaDB collection."""
    return chroma_client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"}  # Use cosine similarity
    )


# Collection names
COLLECTION_PROPERTY_KNOWLEDGE = "property_knowledge"
COLLECTION_PAST_CONVERSATIONS = "past_conversations"
COLLECTION_STYLE_EXAMPLES = "style_examples"
COLLECTION_CORRECTIONS = "corrections"


# ============ PROPERTY KNOWLEDGE BASE ============

@dataclass
class PropertyDocument:
    """A document in a property's knowledge base."""
    property_id: str
    doc_type: str  # "house_rules", "faq", "local_tips", "appliance_guide", etc.
    title: str
    content: str
    metadata: Dict[str, Any] = None


def index_property_document(doc: PropertyDocument) -> str:
    """
    Index a property document in the vector database.
    
    Args:
        doc: The PropertyDocument to index
        
    Returns:
        The document ID
    """
    collection = get_or_create_collection(COLLECTION_PROPERTY_KNOWLEDGE)
    
    # Generate unique ID
    doc_id = hashlib.md5(
        f"{doc.property_id}:{doc.doc_type}:{doc.title}".encode()
    ).hexdigest()
    
    # Generate embedding
    embedding = generate_embedding(f"{doc.title}\n\n{doc.content}")
    
    # Prepare metadata
    metadata = {
        "property_id": doc.property_id,
        "doc_type": doc.doc_type,
        "title": doc.title,
        "indexed_at": datetime.utcnow().isoformat(),
        **(doc.metadata or {})
    }
    
    # Upsert into collection
    collection.upsert(
        ids=[doc_id],
        embeddings=[embedding],
        documents=[doc.content],
        metadatas=[metadata]
    )
    
    return doc_id


def search_property_knowledge(
    query: str,
    property_id: str,
    doc_types: Optional[List[str]] = None,
    top_k: int = 5
) -> List[Dict[str, Any]]:
    """
    Search the property knowledge base for relevant documents.
    
    Args:
        query: The search query
        property_id: The property to search within
        doc_types: Optional filter for document types
        top_k: Number of results to return
        
    Returns:
        List of matching documents with scores
    """
    collection = get_or_create_collection(COLLECTION_PROPERTY_KNOWLEDGE)
    
    # Generate query embedding
    query_embedding = generate_embedding(query)
    
    # Build filter
    where_filter = {"property_id": property_id}
    if doc_types:
        where_filter["doc_type"] = {"$in": doc_types}
    
    # Search
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where=where_filter,
        include=["documents", "metadatas", "distances"]
    )
    
    # Format results
    formatted = []
    if results["ids"] and results["ids"][0]:
        for i, doc_id in enumerate(results["ids"][0]):
            formatted.append({
                "id": doc_id,
                "content": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "score": 1 - results["distances"][0][i]  # Convert distance to similarity
            })
    
    return formatted


# ============ PAST CONVERSATIONS ============

@dataclass
class ConversationRecord:
    """A record of a past conversation for retrieval."""
    conversation_id: int
    property_id: str
    guest_message: str
    ai_response: str
    was_successful: bool  # True if auto-sent or approved without edit
    human_edited_response: Optional[str] = None
    intent: Optional[str] = None
    timestamp: Optional[datetime] = None


def index_conversation(record: ConversationRecord) -> str:
    """
    Index a conversation exchange for future retrieval.
    
    Args:
        record: The conversation record to index
        
    Returns:
        The record ID
    """
    collection = get_or_create_collection(COLLECTION_PAST_CONVERSATIONS)
    
    # Generate unique ID
    record_id = f"conv_{record.conversation_id}_{hashlib.md5(record.guest_message.encode()).hexdigest()[:8]}"
    
    # Embed the guest message (what we'll search by)
    embedding = generate_embedding(record.guest_message)
    
    # Store the best response (human-edited if available, otherwise AI)
    best_response = record.human_edited_response or record.ai_response
    
    # Prepare metadata
    metadata = {
        "conversation_id": record.conversation_id,
        "property_id": record.property_id,
        "was_successful": record.was_successful,
        "was_edited": record.human_edited_response is not None,
        "intent": record.intent or "unknown",
        "timestamp": (record.timestamp or datetime.utcnow()).isoformat()
    }
    
    # Store document as JSON with both Q and A
    document = json.dumps({
        "guest_message": record.guest_message,
        "response": best_response,
        "original_ai_response": record.ai_response if record.human_edited_response else None
    })
    
    collection.upsert(
        ids=[record_id],
        embeddings=[embedding],
        documents=[document],
        metadatas=[metadata]
    )
    
    return record_id


def search_similar_conversations(
    query: str,
    property_id: Optional[str] = None,
    intent: Optional[str] = None,
    successful_only: bool = True,
    top_k: int = 5
) -> List[Dict[str, Any]]:
    """
    Search for similar past conversations.
    
    Args:
        query: The guest message to find similar conversations for
        property_id: Optional property filter
        intent: Optional intent filter
        successful_only: Only return successful (approved) conversations
        top_k: Number of results
        
    Returns:
        List of similar conversations with their responses
    """
    collection = get_or_create_collection(COLLECTION_PAST_CONVERSATIONS)
    
    # Generate query embedding
    query_embedding = generate_embedding(query)
    
    # Build filter
    where_filter = {}
    if property_id:
        where_filter["property_id"] = property_id
    if intent:
        where_filter["intent"] = intent
    if successful_only:
        where_filter["was_successful"] = True
    
    # Search
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where=where_filter if where_filter else None,
        include=["documents", "metadatas", "distances"]
    )
    
    # Format results
    formatted = []
    if results["ids"] and results["ids"][0]:
        for i, record_id in enumerate(results["ids"][0]):
            doc_data = json.loads(results["documents"][0][i])
            formatted.append({
                "id": record_id,
                "guest_message": doc_data["guest_message"],
                "response": doc_data["response"],
                "original_ai_response": doc_data.get("original_ai_response"),
                "metadata": results["metadatas"][0][i],
                "score": 1 - results["distances"][0][i]
            })
    
    return formatted


# ============ STYLE EXAMPLES (SEMANTIC) ============

def index_style_examples(examples: List[Dict[str, Any]]) -> int:
    """
    Index style examples with embeddings for semantic retrieval.
    
    Args:
        examples: List of style example dictionaries
        
    Returns:
        Number of examples indexed
    """
    collection = get_or_create_collection(COLLECTION_STYLE_EXAMPLES)
    
    ids = []
    embeddings = []
    documents = []
    metadatas = []
    
    for i, example in enumerate(examples):
        guest_msg = example.get("guest_message", "")
        your_reply = example.get("your_reply", "")
        
        if not guest_msg or not your_reply:
            continue
        
        example_id = f"style_{i}_{hashlib.md5(guest_msg.encode()).hexdigest()[:8]}"
        
        ids.append(example_id)
        documents.append(json.dumps({
            "guest_message": guest_msg,
            "your_reply": your_reply
        }))
        metadatas.append({
            "category": example.get("category", "general"),
            "tags": ",".join(example.get("tags", []))
        })
    
    # Batch embed guest messages
    guest_messages = [json.loads(d)["guest_message"] for d in documents]
    embeddings = generate_embeddings_batch(guest_messages)
    
    # Upsert all
    if ids:
        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
    
    return len(ids)


def search_style_examples(
    query: str,
    category: Optional[str] = None,
    top_k: int = 3
) -> List[Dict[str, Any]]:
    """
    Search for similar style examples using semantic search.
    
    Args:
        query: The guest message
        category: Optional category filter
        top_k: Number of examples to return
        
    Returns:
        List of relevant style examples
    """
    collection = get_or_create_collection(COLLECTION_STYLE_EXAMPLES)
    
    query_embedding = generate_embedding(query)
    
    where_filter = {"category": category} if category else None
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where=where_filter,
        include=["documents", "metadatas", "distances"]
    )
    
    formatted = []
    if results["ids"] and results["ids"][0]:
        for i, example_id in enumerate(results["ids"][0]):
            doc_data = json.loads(results["documents"][0][i])
            formatted.append({
                "id": example_id,
                "guest_message": doc_data["guest_message"],
                "your_reply": doc_data["your_reply"],
                "category": results["metadatas"][0][i].get("category"),
                "tags": results["metadatas"][0][i].get("tags", "").split(","),
                "score": 1 - results["distances"][0][i]
            })
    
    return formatted


# ============ CORRECTIONS / FEEDBACK LOOP ============

@dataclass
class CorrectionRecord:
    """A record of a human correction to an AI response."""
    conversation_id: int
    property_id: str
    guest_message: str
    original_ai_response: str
    corrected_response: str
    correction_type: str  # "tone", "factual", "policy", "style", "other"
    corrected_by: Optional[str] = None
    timestamp: Optional[datetime] = None


def index_correction(record: CorrectionRecord) -> str:
    """
    Index a correction for learning.
    
    Args:
        record: The correction record
        
    Returns:
        The record ID
    """
    collection = get_or_create_collection(COLLECTION_CORRECTIONS)
    
    record_id = f"corr_{record.conversation_id}_{datetime.utcnow().timestamp()}"
    
    # Embed the guest message
    embedding = generate_embedding(record.guest_message)
    
    document = json.dumps({
        "guest_message": record.guest_message,
        "original_ai_response": record.original_ai_response,
        "corrected_response": record.corrected_response
    })
    
    metadata = {
        "conversation_id": record.conversation_id,
        "property_id": record.property_id,
        "correction_type": record.correction_type,
        "corrected_by": record.corrected_by or "unknown",
        "timestamp": (record.timestamp or datetime.utcnow()).isoformat()
    }
    
    collection.upsert(
        ids=[record_id],
        embeddings=[embedding],
        documents=[document],
        metadatas=[metadata]
    )
    
    return record_id


def search_corrections(
    query: str,
    property_id: Optional[str] = None,
    top_k: int = 3
) -> List[Dict[str, Any]]:
    """
    Search for relevant corrections to learn from.
    
    Args:
        query: The guest message
        property_id: Optional property filter
        top_k: Number of corrections to return
        
    Returns:
        List of relevant corrections
    """
    collection = get_or_create_collection(COLLECTION_CORRECTIONS)
    
    query_embedding = generate_embedding(query)
    
    where_filter = {"property_id": property_id} if property_id else None
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where=where_filter,
        include=["documents", "metadatas", "distances"]
    )
    
    formatted = []
    if results["ids"] and results["ids"][0]:
        for i, record_id in enumerate(results["ids"][0]):
            doc_data = json.loads(results["documents"][0][i])
            formatted.append({
                "id": record_id,
                "guest_message": doc_data["guest_message"],
                "original_ai_response": doc_data["original_ai_response"],
                "corrected_response": doc_data["corrected_response"],
                "metadata": results["metadatas"][0][i],
                "score": 1 - results["distances"][0][i]
            })
    
    return formatted


# ============ INITIALIZATION ============

def initialize_style_examples():
    """Load and index style examples from style_guide.json."""
    try:
        with open("style_guide.json", "r") as f:
            examples = json.load(f)
        count = index_style_examples(examples)
        print(f"[Embeddings] Indexed {count} style examples")
        return count
    except FileNotFoundError:
        print("[Embeddings] No style_guide.json found")
        return 0
    except Exception as e:
        print(f"[Embeddings] Error indexing style examples: {e}")
        return 0


def get_collection_stats() -> Dict[str, int]:
    """Get statistics about all collections."""
    stats = {}
    for name in [
        COLLECTION_PROPERTY_KNOWLEDGE,
        COLLECTION_PAST_CONVERSATIONS,
        COLLECTION_STYLE_EXAMPLES,
        COLLECTION_CORRECTIONS
    ]:
        try:
            collection = get_or_create_collection(name)
            stats[name] = collection.count()
        except Exception:
            stats[name] = 0
    return stats
