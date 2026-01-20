"""
Property Knowledge Base - RAG System for Property-Specific Information.

This module provides:
1. File upload and processing (PDF, DOCX, TXT, MD)
2. Knowledge extraction and structuring
3. Vector storage for semantic search
4. Learning from past conversations
"""

import os
import json
import hashlib
from datetime import datetime
from typing import List, Dict, Optional, Any
from pathlib import Path

from sqlalchemy import Column, Integer, String, Text, DateTime, Float, ForeignKey, Boolean, Enum as SQLEnum
from sqlalchemy.orm import relationship
import enum

from models import Base, SessionLocal, Message, Conversation, MessageDirection
from config import settings
from utils import log_event

# Directory for uploaded files
UPLOAD_DIR = Path("uploads/property_docs")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# ============ DATABASE MODELS ============

class KnowledgeType(enum.Enum):
    """Types of knowledge entries."""
    amenity = "amenity"
    house_rule = "house_rule"
    local_recommendation = "local_recommendation"
    appliance_guide = "appliance_guide"
    common_issue = "common_issue"
    faq = "faq"
    general = "general"


class PropertyKnowledge(Base):
    """
    Stores structured knowledge about a property.
    Each entry is a piece of information that can be retrieved via RAG.
    """
    __tablename__ = "property_knowledge"
    __table_args__ = {'extend_existing': True}
    
    id = Column(Integer, primary_key=True)
    listing_id = Column(String(50), index=True)  # Hostify listing ID
    listing_name = Column(String(255))
    
    # Knowledge content
    knowledge_type = Column(SQLEnum(KnowledgeType), default=KnowledgeType.general)
    title = Column(String(255))  # Brief title/topic
    content = Column(Text)  # Full knowledge content
    
    # For Q&A style entries
    question = Column(Text, nullable=True)  # Common question this answers
    answer = Column(Text, nullable=True)  # The answer
    
    # Metadata
    source = Column(String(50))  # "file_upload", "message_learning", "manual"
    source_file = Column(String(255), nullable=True)  # Original filename if from upload
    confidence = Column(Float, default=1.0)  # How confident we are in this knowledge
    
    # Usage tracking
    times_used = Column(Integer, default=0)
    last_used_at = Column(DateTime, nullable=True)
    was_helpful = Column(Boolean, nullable=True)  # Feedback tracking
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Vector embedding ID (for ChromaDB)
    embedding_id = Column(String(100), nullable=True)
    
    def __repr__(self):
        return f"<PropertyKnowledge {self.id} - {self.listing_name}: {self.title}>"


class UploadedFile(Base):
    """Tracks uploaded property documents."""
    __tablename__ = "uploaded_files"
    __table_args__ = {'extend_existing': True}
    
    id = Column(Integer, primary_key=True)
    listing_id = Column(String(50), index=True)
    listing_name = Column(String(255))
    
    filename = Column(String(255))
    original_filename = Column(String(255))
    file_type = Column(String(50))  # pdf, docx, txt, md
    file_size = Column(Integer)
    file_hash = Column(String(64))  # SHA256 hash for deduplication
    
    # Processing status
    processed = Column(Boolean, default=False)
    processing_error = Column(Text, nullable=True)
    knowledge_entries_created = Column(Integer, default=0)
    
    uploaded_at = Column(DateTime, default=datetime.utcnow)
    processed_at = Column(DateTime, nullable=True)
    
    def __repr__(self):
        return f"<UploadedFile {self.id} - {self.original_filename}>"


class LearningSession(Base):
    """Tracks message learning/audit sessions."""
    __tablename__ = "learning_sessions"
    __table_args__ = {'extend_existing': True}
    
    id = Column(Integer, primary_key=True)
    
    # What was analyzed
    listing_id = Column(String(50), nullable=True)  # Null = all listings
    conversations_analyzed = Column(Integer, default=0)
    messages_analyzed = Column(Integer, default=0)
    
    # Results
    knowledge_entries_created = Column(Integer, default=0)
    knowledge_entries_updated = Column(Integer, default=0)
    
    # Status
    status = Column(String(20), default="pending")  # pending, running, completed, failed
    error = Column(Text, nullable=True)
    
    started_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    
    def __repr__(self):
        return f"<LearningSession {self.id} - {self.status}>"


# ============ FILE PROCESSING ============

def extract_text_from_file(file_path: str, file_type: str) -> str:
    """
    Extract text content from various file formats.
    """
    file_path = Path(file_path)
    
    if file_type in ['txt', 'md']:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    elif file_type == 'pdf':
        try:
            import PyPDF2
            text = ""
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    text += page.extract_text() + "\n"
            return text
        except ImportError:
            # Try pdfplumber as fallback
            try:
                import pdfplumber
                text = ""
                with pdfplumber.open(file_path) as pdf:
                    for page in pdf.pages:
                        text += (page.extract_text() or "") + "\n"
                return text
            except ImportError:
                raise ImportError("Please install PyPDF2 or pdfplumber: pip install PyPDF2 pdfplumber")
    
    elif file_type == 'docx':
        try:
            from docx import Document
            doc = Document(file_path)
            return "\n".join([para.text for para in doc.paragraphs])
        except ImportError:
            raise ImportError("Please install python-docx: pip install python-docx")
    
    elif file_type in ['json']:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return json.dumps(data, indent=2)
    
    else:
        # Try to read as plain text
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()


async def process_uploaded_file(file_id: int) -> Dict[str, Any]:
    """
    Process an uploaded file and extract knowledge entries.
    Uses GPT-4o to structure the content.
    """
    from openai import OpenAI
    
    db = SessionLocal()
    client = OpenAI(api_key=settings.OPENAI_API_KEY)
    
    try:
        # Get the file record
        file_record = db.query(UploadedFile).filter(UploadedFile.id == file_id).first()
        if not file_record:
            return {"error": "File not found"}
        
        if file_record.processed:
            return {"error": "File already processed"}
        
        # Extract text
        file_path = UPLOAD_DIR / file_record.filename
        try:
            text_content = extract_text_from_file(str(file_path), file_record.file_type)
        except Exception as e:
            file_record.processing_error = str(e)
            db.commit()
            return {"error": f"Failed to extract text: {e}"}
        
        if not text_content.strip():
            file_record.processing_error = "No text content extracted"
            db.commit()
            return {"error": "No text content extracted from file"}
        
        # Use GPT-4o to structure the content
        system_prompt = """You are analyzing a property document to extract useful knowledge for a vacation rental AI assistant.

Extract all relevant information and structure it into knowledge entries.

For each piece of information, categorize it as one of:
- amenity: Pool, hot tub, grill, game room, etc.
- house_rule: Check-in/out times, quiet hours, smoking policy, pet rules
- local_recommendation: Restaurants, grocery stores, attractions, urgent care
- appliance_guide: How to use TV, thermostat, pool heater, grill, etc.
- common_issue: Known problems and their solutions
- faq: Frequently asked questions and answers
- general: Other useful information

OUTPUT FORMAT (JSON array):
[
    {
        "knowledge_type": "amenity",
        "title": "Pool",
        "content": "Heated pool available. Temperature set to 82°F. Pool hours 8AM-10PM.",
        "question": "What are the pool hours?",
        "answer": "The pool is open from 8AM to 10PM. It's heated to 82°F."
    },
    {
        "knowledge_type": "appliance_guide",
        "title": "Thermostat",
        "content": "Nest thermostat in hallway. Press to wake, swipe to adjust temperature.",
        "question": "How do I adjust the AC?",
        "answer": "The Nest thermostat is in the hallway. Press the display to wake it, then swipe up/down to adjust temperature."
    }
]

Extract as many relevant entries as possible. Be specific and detailed."""

        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Property: {file_record.listing_name}\n\nDocument content:\n{text_content[:15000]}"}  # Limit to ~15k chars
                ],
                temperature=0.3,
                max_tokens=4000
            )
            
            result = json.loads(response.choices[0].message.content)
            entries = result if isinstance(result, list) else result.get("entries", result.get("knowledge", []))
            
        except Exception as e:
            file_record.processing_error = f"AI processing failed: {e}"
            db.commit()
            return {"error": f"AI processing failed: {e}"}
        
        # Create knowledge entries
        entries_created = 0
        for entry in entries:
            try:
                knowledge_type = entry.get("knowledge_type", "general")
                if knowledge_type not in [e.value for e in KnowledgeType]:
                    knowledge_type = "general"
                
                knowledge = PropertyKnowledge(
                    listing_id=file_record.listing_id,
                    listing_name=file_record.listing_name,
                    knowledge_type=KnowledgeType(knowledge_type),
                    title=entry.get("title", "Untitled")[:255],
                    content=entry.get("content", ""),
                    question=entry.get("question"),
                    answer=entry.get("answer"),
                    source="file_upload",
                    source_file=file_record.original_filename,
                    confidence=0.9
                )
                db.add(knowledge)
                entries_created += 1
            except Exception as e:
                print(f"[Knowledge] Failed to create entry: {e}")
                continue
        
        # Update file record
        file_record.processed = True
        file_record.processed_at = datetime.utcnow()
        file_record.knowledge_entries_created = entries_created
        db.commit()
        
        log_event("file_processed", payload={
            "file_id": file_id,
            "entries_created": entries_created
        })
        
        return {
            "success": True,
            "entries_created": entries_created,
            "file_id": file_id
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e)}
    finally:
        db.close()


# ============ MESSAGE LEARNING ============

async def learn_from_messages(listing_id: Optional[str] = None, limit: int = 500) -> Dict[str, Any]:
    """
    Analyze past conversations to extract property knowledge.
    
    This looks at successful host responses to learn:
    - Common questions and good answers
    - Property-specific information mentioned by hosts
    - Patterns in how issues were resolved
    
    Args:
        listing_id: Optional - only learn from this property's conversations
        limit: Maximum conversations to analyze
        
    Returns:
        Summary of learning session
    """
    from openai import OpenAI
    
    db = SessionLocal()
    client = OpenAI(api_key=settings.OPENAI_API_KEY)
    
    # Create learning session record
    session = LearningSession(
        listing_id=listing_id,
        status="running"
    )
    db.add(session)
    db.commit()
    
    try:
        # Get conversations to analyze
        query = db.query(Conversation)
        if listing_id:
            query = query.filter(Conversation.listing_id == listing_id)
        conversations = query.order_by(Conversation.last_message_at.desc()).limit(limit).all()
        
        session.conversations_analyzed = len(conversations)
        
        all_qa_pairs = []
        messages_count = 0
        
        # Extract Q&A pairs from conversations
        for conv in conversations:
            messages = db.query(Message).filter(
                Message.conversation_id == conv.id
            ).order_by(Message.sent_at).all()
            
            messages_count += len(messages)
            
            # Find guest question -> host answer pairs
            for i, msg in enumerate(messages):
                if msg.direction == MessageDirection.inbound:  # Guest message
                    # Look for the next host response
                    for j in range(i + 1, min(i + 3, len(messages))):
                        if messages[j].direction == MessageDirection.outbound:
                            # Found a Q&A pair
                            all_qa_pairs.append({
                                "listing_id": conv.listing_id,
                                "listing_name": conv.listing_name,
                                "guest_question": msg.content,
                                "host_answer": messages[j].content
                            })
                            break
        
        session.messages_analyzed = messages_count
        
        if not all_qa_pairs:
            session.status = "completed"
            session.completed_at = datetime.utcnow()
            db.commit()
            return {
                "success": True,
                "message": "No Q&A pairs found to learn from",
                "session_id": session.id
            }
        
        # Group by listing
        by_listing = {}
        for qa in all_qa_pairs:
            lid = qa["listing_id"] or "unknown"
            if lid not in by_listing:
                by_listing[lid] = {
                    "listing_name": qa["listing_name"],
                    "pairs": []
                }
            by_listing[lid]["pairs"].append(qa)
        
        entries_created = 0
        entries_updated = 0
        
        # Process each listing's Q&A pairs
        for lid, data in by_listing.items():
            if len(data["pairs"]) < 2:
                continue  # Need at least a few pairs to learn patterns
            
            # Sample up to 50 pairs per listing
            pairs_sample = data["pairs"][:50]
            
            # Use GPT-4o to extract knowledge
            pairs_text = "\n\n".join([
                f"Guest: {p['guest_question']}\nHost: {p['host_answer']}"
                for p in pairs_sample
            ])
            
            system_prompt = """You are analyzing past vacation rental conversations to extract reusable knowledge.

Look for:
1. Property-specific information (amenities, features, access codes mentioned)
2. Common questions and good answers that can be reused
3. Troubleshooting patterns (issues and their solutions)
4. Local recommendations given by hosts

Extract knowledge that would be useful for an AI to answer similar questions in the future.

OUTPUT FORMAT (JSON array):
[
    {
        "knowledge_type": "faq",
        "title": "Brief topic title",
        "content": "Detailed knowledge content",
        "question": "Common question this answers",
        "answer": "Good answer template based on host responses",
        "confidence": 0.8
    }
]

Only extract high-confidence, reusable knowledge. If a response was specific to one guest's situation, don't include it."""

            try:
                response = client.chat.completions.create(
                    model="gpt-4o",
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Property: {data['listing_name']}\n\nConversations:\n{pairs_text[:12000]}"}
                    ],
                    temperature=0.3,
                    max_tokens=3000
                )
                
                result = json.loads(response.choices[0].message.content)
                entries = result if isinstance(result, list) else result.get("entries", result.get("knowledge", []))
                
            except Exception as e:
                print(f"[Learning] Failed to process listing {lid}: {e}")
                continue
            
            # Create/update knowledge entries
            for entry in entries:
                try:
                    knowledge_type = entry.get("knowledge_type", "faq")
                    if knowledge_type not in [e.value for e in KnowledgeType]:
                        knowledge_type = "faq"
                    
                    title = entry.get("title", "Untitled")[:255]
                    
                    # Check if similar entry exists
                    existing = db.query(PropertyKnowledge).filter(
                        PropertyKnowledge.listing_id == lid,
                        PropertyKnowledge.title == title
                    ).first()
                    
                    if existing:
                        # Update if new info is better
                        if entry.get("confidence", 0.7) > existing.confidence:
                            existing.content = entry.get("content", existing.content)
                            existing.answer = entry.get("answer", existing.answer)
                            existing.confidence = entry.get("confidence", 0.7)
                            existing.updated_at = datetime.utcnow()
                            entries_updated += 1
                    else:
                        # Create new entry
                        knowledge = PropertyKnowledge(
                            listing_id=lid,
                            listing_name=data["listing_name"],
                            knowledge_type=KnowledgeType(knowledge_type),
                            title=title,
                            content=entry.get("content", ""),
                            question=entry.get("question"),
                            answer=entry.get("answer"),
                            source="message_learning",
                            confidence=entry.get("confidence", 0.7)
                        )
                        db.add(knowledge)
                        entries_created += 1
                        
                except Exception as e:
                    print(f"[Learning] Failed to create entry: {e}")
                    continue
            
            db.commit()
        
        # Update session
        session.knowledge_entries_created = entries_created
        session.knowledge_entries_updated = entries_updated
        session.status = "completed"
        session.completed_at = datetime.utcnow()
        db.commit()
        
        log_event("learning_completed", payload={
            "session_id": session.id,
            "conversations_analyzed": session.conversations_analyzed,
            "entries_created": entries_created,
            "entries_updated": entries_updated
        })
        
        return {
            "success": True,
            "session_id": session.id,
            "conversations_analyzed": session.conversations_analyzed,
            "messages_analyzed": session.messages_analyzed,
            "entries_created": entries_created,
            "entries_updated": entries_updated
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        session.status = "failed"
        session.error = str(e)
        session.completed_at = datetime.utcnow()
        db.commit()
        return {"error": str(e), "session_id": session.id}
    finally:
        db.close()


# ============ KNOWLEDGE RETRIEVAL ============

def search_knowledge(
    query: str,
    listing_id: Optional[str] = None,
    knowledge_types: Optional[List[KnowledgeType]] = None,
    top_k: int = 5
) -> List[Dict[str, Any]]:
    """
    Search the knowledge base for relevant information.
    
    Uses keyword matching for now. Can be upgraded to vector search with ChromaDB.
    
    Args:
        query: The search query (usually the guest's question)
        listing_id: Optional - filter to specific property
        knowledge_types: Optional - filter to specific types
        top_k: Number of results to return
        
    Returns:
        List of relevant knowledge entries
    """
    db = SessionLocal()
    
    try:
        # Build base query
        q = db.query(PropertyKnowledge)
        
        if listing_id:
            q = q.filter(PropertyKnowledge.listing_id == listing_id)
        
        if knowledge_types:
            q = q.filter(PropertyKnowledge.knowledge_type.in_(knowledge_types))
        
        all_entries = q.all()
        
        if not all_entries:
            return []
        
        # Score entries by keyword relevance
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        scored = []
        for entry in all_entries:
            score = 0
            
            # Match against title
            if entry.title:
                title_lower = entry.title.lower()
                if any(word in title_lower for word in query_words):
                    score += 5
            
            # Match against content
            if entry.content:
                content_lower = entry.content.lower()
                word_matches = sum(1 for word in query_words if word in content_lower)
                score += word_matches * 2
            
            # Match against question
            if entry.question:
                question_lower = entry.question.lower()
                if any(word in question_lower for word in query_words):
                    score += 4
            
            # Boost by confidence
            score *= entry.confidence
            
            if score > 0:
                scored.append((score, entry))
        
        # Sort by score and return top_k
        scored.sort(key=lambda x: x[0], reverse=True)
        
        results = []
        for score, entry in scored[:top_k]:
            # Update usage tracking
            entry.times_used += 1
            entry.last_used_at = datetime.utcnow()
            
            results.append({
                "id": entry.id,
                "listing_id": entry.listing_id,
                "listing_name": entry.listing_name,
                "knowledge_type": entry.knowledge_type.value,
                "title": entry.title,
                "content": entry.content,
                "question": entry.question,
                "answer": entry.answer,
                "confidence": entry.confidence,
                "score": score
            })
        
        db.commit()
        return results
        
    finally:
        db.close()


def get_property_knowledge_summary(listing_id: str) -> Dict[str, Any]:
    """Get a summary of all knowledge for a property."""
    db = SessionLocal()
    
    try:
        entries = db.query(PropertyKnowledge).filter(
            PropertyKnowledge.listing_id == listing_id
        ).all()
        
        if not entries:
            return {
                "listing_id": listing_id,
                "total_entries": 0,
                "by_type": {},
                "entries": []
            }
        
        by_type = {}
        for entry in entries:
            type_name = entry.knowledge_type.value
            if type_name not in by_type:
                by_type[type_name] = 0
            by_type[type_name] += 1
        
        return {
            "listing_id": listing_id,
            "listing_name": entries[0].listing_name if entries else "",
            "total_entries": len(entries),
            "by_type": by_type,
            "entries": [
                {
                    "id": e.id,
                    "type": e.knowledge_type.value,
                    "title": e.title,
                    "content": e.content[:200] + "..." if len(e.content or "") > 200 else e.content,
                    "source": e.source,
                    "confidence": e.confidence,
                    "times_used": e.times_used
                }
                for e in entries
            ]
        }
    finally:
        db.close()
