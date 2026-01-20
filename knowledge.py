"""
Knowledge Base Management System.
Handles property-specific knowledge, guest profiles, and the feedback learning loop.
"""

import json
import os
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from pathlib import Path

from sqlalchemy.orm import Session
from sqlalchemy import func

from config import settings
from models import (
    SessionLocal, GuestProfile, PropertyKnowledge, 
    ConversationSummary, ConversationIntent, ResponseFeedback,
    Message, Conversation, MessageDirection, GuestIndex
)
from embeddings import (
    index_property_document, PropertyDocument,
    index_correction, CorrectionRecord,
    index_conversation, ConversationRecord,
    search_property_knowledge, get_collection_stats
)


# ============ PROPERTY KNOWLEDGE BASE ============

class KnowledgeBaseManager:
    """
    Manages property-specific knowledge bases.
    Handles creation, indexing, and retrieval of property information.
    """
    
    KNOWLEDGE_DIR = Path("./property_knowledge")
    
    DOC_TYPES = [
        "house_rules",
        "check_in",
        "check_out", 
        "faq",
        "local_tips",
        "appliance_guide",
        "known_issues",
        "parking",
        "amenities"
    ]
    
    def __init__(self):
        self.KNOWLEDGE_DIR.mkdir(exist_ok=True)
    
    def create_property_folder(self, property_id: str, property_name: str) -> Path:
        """Create a folder structure for a property's knowledge base."""
        property_dir = self.KNOWLEDGE_DIR / property_id
        property_dir.mkdir(exist_ok=True)
        
        # Create template files
        templates = {
            "house_rules.md": f"""# House Rules - {property_name}

## Quiet Hours
- Please keep noise to a minimum between 10 PM and 8 AM

## Smoking
- This is a non-smoking property
- Smoking is permitted on the outdoor patio only

## Pets
- [Add pet policy]

## Parties
- No parties or events without prior approval

## Additional Rules
- [Add any property-specific rules]
""",
            "check_in.md": f"""# Check-In Instructions - {property_name}

## Check-In Time
- Standard check-in: 4:00 PM

## Access
- Door Code: [Will be sent before arrival]
- [Add any specific access instructions]

## Parking
- [Add parking instructions]

## First Steps
1. Enter the door code on the keypad
2. Locate the welcome book on the kitchen counter
3. Connect to WiFi (details in welcome book)
4. Adjust thermostat as needed
""",
            "faq.md": f"""# Frequently Asked Questions - {property_name}

## WiFi
**Q: What's the WiFi password?**
A: Network: [network_name], Password: [password]

## Thermostat
**Q: How do I adjust the temperature?**
A: [Add thermostat instructions]

## TV/Entertainment
**Q: How do I use the TV?**
A: [Add TV instructions]

## Trash
**Q: Where do I put the trash?**
A: [Add trash/recycling instructions]
""",
            "local_tips.md": f"""# Local Recommendations - {property_name}

## Restaurants
- [Restaurant 1] - [Type of food] - [Distance]
- [Restaurant 2] - [Type of food] - [Distance]

## Coffee Shops
- [Coffee shop] - [Distance]

## Grocery Stores
- [Store name] - [Distance]

## Attractions
- [Attraction 1] - [Description]
- [Attraction 2] - [Description]

## Emergency Services
- Nearest Hospital: [Name and address]
- Police (non-emergency): [Number]
""",
            "known_issues.md": f"""# Known Issues & Workarounds - {property_name}

## [Issue Title]
**Description:** [What the issue is]
**Workaround:** [How to work around it]
**Status:** [Being fixed / Permanent workaround needed]

---
*Add new issues as they arise*
"""
        }
        
        for filename, content in templates.items():
            filepath = property_dir / filename
            if not filepath.exists():
                filepath.write_text(content)
        
        return property_dir
    
    def load_and_index_property(self, property_id: str, property_name: str = "") -> int:
        """
        Load all knowledge documents for a property and index them in the vector DB.
        
        Args:
            property_id: The property ID
            property_name: Optional property name
            
        Returns:
            Number of documents indexed
        """
        property_dir = self.KNOWLEDGE_DIR / property_id
        
        if not property_dir.exists():
            # Create folder with templates
            self.create_property_folder(property_id, property_name)
            return 0
        
        indexed_count = 0
        db = SessionLocal()
        
        try:
            for filepath in property_dir.glob("*.md"):
                doc_type = filepath.stem  # e.g., "house_rules" from "house_rules.md"
                content = filepath.read_text()
                
                # Skip empty or template files
                if not content.strip() or "[Add" in content[:200]:
                    continue
                
                # Extract title from first line if it's a header
                lines = content.strip().split("\n")
                title = lines[0].replace("#", "").strip() if lines[0].startswith("#") else doc_type
                
                # Index in vector DB
                doc = PropertyDocument(
                    property_id=property_id,
                    doc_type=doc_type,
                    title=title,
                    content=content
                )
                index_property_document(doc)
                
                # Also save to database
                existing = db.query(PropertyKnowledge).filter(
                    PropertyKnowledge.property_id == property_id,
                    PropertyKnowledge.doc_type == doc_type
                ).first()
                
                if existing:
                    existing.content = content
                    existing.title = title
                    existing.is_embedded = True
                    existing.embedded_at = datetime.utcnow()
                    existing.updated_at = datetime.utcnow()
                else:
                    knowledge = PropertyKnowledge(
                        property_id=property_id,
                        property_name=property_name,
                        doc_type=doc_type,
                        title=title,
                        content=content,
                        is_embedded=True,
                        embedded_at=datetime.utcnow()
                    )
                    db.add(knowledge)
                
                indexed_count += 1
            
            db.commit()
            
        finally:
            db.close()
        
        return indexed_count
    
    def add_knowledge(
        self,
        property_id: str,
        doc_type: str,
        title: str,
        content: str,
        created_by: str = None
    ) -> bool:
        """
        Add a new knowledge document for a property.
        
        Args:
            property_id: The property ID
            doc_type: Type of document (from DOC_TYPES)
            title: Document title
            content: Document content
            created_by: Who created this document
            
        Returns:
            True if successful
        """
        db = SessionLocal()
        
        try:
            # Save to database
            existing = db.query(PropertyKnowledge).filter(
                PropertyKnowledge.property_id == property_id,
                PropertyKnowledge.doc_type == doc_type,
                PropertyKnowledge.title == title
            ).first()
            
            if existing:
                existing.content = content
                existing.updated_at = datetime.utcnow()
            else:
                knowledge = PropertyKnowledge(
                    property_id=property_id,
                    doc_type=doc_type,
                    title=title,
                    content=content,
                    created_by=created_by
                )
                db.add(knowledge)
            
            db.commit()
            
            # Index in vector DB
            doc = PropertyDocument(
                property_id=property_id,
                doc_type=doc_type,
                title=title,
                content=content
            )
            index_property_document(doc)
            
            # Update embedding status
            if existing:
                existing.is_embedded = True
                existing.embedded_at = datetime.utcnow()
            db.commit()
            
            return True
            
        except Exception as e:
            db.rollback()
            print(f"[Knowledge] Error adding document: {e}")
            return False
        finally:
            db.close()
    
    def get_property_knowledge(self, property_id: str) -> List[Dict[str, Any]]:
        """Get all knowledge documents for a property."""
        db = SessionLocal()
        try:
            docs = db.query(PropertyKnowledge).filter(
                PropertyKnowledge.property_id == property_id,
                PropertyKnowledge.is_active == True
            ).all()
            
            return [
                {
                    "id": doc.id,
                    "doc_type": doc.doc_type,
                    "title": doc.title,
                    "content": doc.content,
                    "is_embedded": doc.is_embedded,
                    "updated_at": doc.updated_at.isoformat() if doc.updated_at else None
                }
                for doc in docs
            ]
        finally:
            db.close()


# Global instance
knowledge_manager = KnowledgeBaseManager()


# ============ GUEST PROFILE MANAGEMENT ============

class GuestProfileManager:
    """
    Manages guest profiles for personalization and learning.
    """
    
    def get_or_create_profile(self, guest_phone: str, db: Session) -> GuestProfile:
        """Get or create a guest profile."""
        profile = db.query(GuestProfile).filter(
            GuestProfile.guest_phone == guest_phone
        ).first()
        
        if not profile:
            profile = GuestProfile(guest_phone=guest_phone)
            db.add(profile)
            db.commit()
            db.refresh(profile)
        
        return profile
    
    def update_from_conversation(
        self,
        guest_phone: str,
        messages: List[Message],
        guest_context: Optional[GuestIndex] = None
    ):
        """
        Update a guest profile based on a conversation.
        Analyzes communication style, sentiment, and preferences.
        """
        db = SessionLocal()
        
        try:
            profile = self.get_or_create_profile(guest_phone, db)
            
            # Update basic info from guest context
            if guest_context:
                profile.guest_name = guest_context.guest_name
                profile.guest_email = guest_context.guest_email
            
            # Count inbound messages for style analysis
            inbound_messages = [m for m in messages if m.direction == MessageDirection.inbound]
            
            if inbound_messages:
                # Calculate average message length
                avg_length = sum(len(m.content) for m in inbound_messages) / len(inbound_messages)
                profile.avg_message_length = avg_length
                
                # Infer communication style
                if avg_length < 50:
                    profile.communication_style = "brief"
                elif avg_length < 150:
                    profile.communication_style = "moderate"
                else:
                    profile.communication_style = "detailed"
            
            # Update interaction timestamps
            profile.last_interaction_at = datetime.utcnow()
            if not profile.first_interaction_at:
                profile.first_interaction_at = datetime.utcnow()
            
            # Increment conversation count
            profile.total_conversations = (profile.total_conversations or 0) + 1
            
            db.commit()
            
        finally:
            db.close()
    
    def analyze_sentiment(self, guest_phone: str, message: str) -> str:
        """
        Analyze sentiment of a message and update profile.
        Returns: "positive", "neutral", or "negative"
        """
        from openai import OpenAI
        
        client = OpenAI(api_key=settings.OPENAI_API_KEY)
        
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "Analyze the sentiment of this guest message. Respond with exactly one word: positive, neutral, or negative"
                    },
                    {"role": "user", "content": message}
                ],
                temperature=0,
                max_tokens=10
            )
            
            sentiment = response.choices[0].message.content.strip().lower()
            if sentiment not in ["positive", "neutral", "negative"]:
                sentiment = "neutral"
            
            # Update profile
            db = SessionLocal()
            try:
                profile = self.get_or_create_profile(guest_phone, db)
                
                # Update sentiment history
                history = json.loads(profile.sentiment_history or "[]")
                history.append({
                    "sentiment": sentiment,
                    "timestamp": datetime.utcnow().isoformat()
                })
                # Keep last 20 entries
                profile.sentiment_history = json.dumps(history[-20:])
                
                # Update overall sentiment based on recent history
                recent = history[-5:]
                sentiments = [h["sentiment"] for h in recent]
                if sentiments.count("negative") >= 2:
                    profile.overall_sentiment = "negative"
                elif sentiments.count("positive") >= 3:
                    profile.overall_sentiment = "positive"
                else:
                    profile.overall_sentiment = "neutral"
                
                db.commit()
            finally:
                db.close()
            
            return sentiment
            
        except Exception:
            return "neutral"
    
    def mark_vip(self, guest_phone: str, is_vip: bool, notes: str = None):
        """Mark a guest as VIP (or remove VIP status)."""
        db = SessionLocal()
        try:
            profile = self.get_or_create_profile(guest_phone, db)
            profile.is_vip = is_vip
            if notes:
                profile.special_handling_notes = notes
            db.commit()
        finally:
            db.close()
    
    def add_issue(self, guest_phone: str, issue: str):
        """Record an issue for a guest."""
        db = SessionLocal()
        try:
            profile = self.get_or_create_profile(guest_phone, db)
            
            issues = json.loads(profile.past_issues or "[]")
            issues.append({
                "issue": issue,
                "timestamp": datetime.utcnow().isoformat()
            })
            profile.past_issues = json.dumps(issues[-10:])  # Keep last 10
            
            db.commit()
        finally:
            db.close()


# Global instance
guest_profile_manager = GuestProfileManager()


# ============ FEEDBACK LEARNING LOOP ============

class FeedbackLearner:
    """
    Learns from human corrections to improve future responses.
    """
    
    def record_feedback(
        self,
        conversation_id: int,
        message_id: Optional[int],
        original_response: str,
        original_confidence: float,
        was_approved: bool,
        was_edited: bool,
        corrected_response: str = None,
        correction_type: str = None,
        human_reviewer: str = None
    ) -> int:
        """
        Record feedback on an AI response.
        
        Returns:
            The feedback record ID
        """
        db = SessionLocal()
        
        try:
            feedback = ResponseFeedback(
                conversation_id=conversation_id,
                message_id=message_id,
                original_response=original_response,
                original_confidence=original_confidence,
                corrected_response=corrected_response,
                correction_type=correction_type,
                was_approved=was_approved,
                was_edited=was_edited,
                was_rejected=not was_approved and not was_edited,
                human_reviewer=human_reviewer
            )
            db.add(feedback)
            db.commit()
            db.refresh(feedback)
            
            return feedback.id
            
        finally:
            db.close()
    
    def index_correction(self, feedback_id: int):
        """
        Index a correction in the vector database for future learning.
        Only indexes edited responses (not just approvals).
        """
        db = SessionLocal()
        
        try:
            feedback = db.query(ResponseFeedback).filter(
                ResponseFeedback.id == feedback_id
            ).first()
            
            if not feedback or not feedback.was_edited:
                return
            
            # Get conversation context
            conversation = db.query(Conversation).filter(
                Conversation.id == feedback.conversation_id
            ).first()
            
            if not conversation:
                return
            
            # Get the guest message that triggered this response
            message = db.query(Message).filter(
                Message.conversation_id == feedback.conversation_id,
                Message.direction == MessageDirection.inbound,
                Message.sent_at <= feedback.created_at
            ).order_by(Message.sent_at.desc()).first()
            
            if not message:
                return
            
            # Get guest context for property_id
            guest_context = db.query(GuestIndex).filter(
                GuestIndex.guest_phone == conversation.guest_phone
            ).first()
            
            property_id = guest_context.listing_id if guest_context else "unknown"
            
            # Index the correction
            record = CorrectionRecord(
                conversation_id=feedback.conversation_id,
                property_id=property_id,
                guest_message=message.content,
                original_ai_response=feedback.original_response,
                corrected_response=feedback.corrected_response,
                correction_type=feedback.correction_type or "style",
                corrected_by=feedback.human_reviewer,
                timestamp=feedback.created_at
            )
            
            index_correction(record)
            
            # Mark as indexed
            feedback.is_indexed = True
            feedback.indexed_at = datetime.utcnow()
            db.commit()
            
            print(f"[Feedback] Indexed correction from conversation {feedback.conversation_id}")
            
        finally:
            db.close()
    
    def index_successful_conversation(
        self,
        conversation_id: int,
        guest_message: str,
        ai_response: str,
        was_edited: bool,
        edited_response: str = None,
        intent: str = None
    ):
        """
        Index a successful conversation exchange for future retrieval.
        """
        db = SessionLocal()
        
        try:
            # Get conversation for property context
            conversation = db.query(Conversation).filter(
                Conversation.id == conversation_id
            ).first()
            
            if not conversation:
                return
            
            guest_context = db.query(GuestIndex).filter(
                GuestIndex.guest_phone == conversation.guest_phone
            ).first()
            
            property_id = guest_context.listing_id if guest_context else "unknown"
            
            # Index the conversation
            record = ConversationRecord(
                conversation_id=conversation_id,
                property_id=property_id,
                guest_message=guest_message,
                ai_response=ai_response,
                was_successful=True,
                human_edited_response=edited_response if was_edited else None,
                intent=intent,
                timestamp=datetime.utcnow()
            )
            
            index_conversation(record)
            
        finally:
            db.close()
    
    def get_learning_stats(self) -> Dict[str, Any]:
        """Get statistics about the learning system."""
        db = SessionLocal()
        
        try:
            total_feedback = db.query(ResponseFeedback).count()
            approved = db.query(ResponseFeedback).filter(
                ResponseFeedback.was_approved == True
            ).count()
            edited = db.query(ResponseFeedback).filter(
                ResponseFeedback.was_edited == True
            ).count()
            rejected = db.query(ResponseFeedback).filter(
                ResponseFeedback.was_rejected == True
            ).count()
            indexed = db.query(ResponseFeedback).filter(
                ResponseFeedback.is_indexed == True
            ).count()
            
            # Get vector DB stats
            vector_stats = get_collection_stats()
            
            return {
                "total_feedback": total_feedback,
                "approved": approved,
                "edited": edited,
                "rejected": rejected,
                "indexed_corrections": indexed,
                "approval_rate": approved / total_feedback if total_feedback > 0 else 0,
                "edit_rate": edited / total_feedback if total_feedback > 0 else 0,
                "vector_db": vector_stats
            }
            
        finally:
            db.close()


# Global instance
feedback_learner = FeedbackLearner()


# ============ CONVERSATION SUMMARIZATION ============

class ConversationSummarizer:
    """
    Generates and manages conversation summaries for long threads.
    """
    
    SUMMARY_THRESHOLD = 15  # Messages before summarizing
    
    def should_summarize(self, conversation_id: int) -> bool:
        """Check if a conversation needs summarization."""
        db = SessionLocal()
        try:
            message_count = db.query(Message).filter(
                Message.conversation_id == conversation_id
            ).count()
            
            # Check if we have a recent summary
            latest_summary = db.query(ConversationSummary).filter(
                ConversationSummary.conversation_id == conversation_id
            ).order_by(ConversationSummary.created_at.desc()).first()
            
            if latest_summary:
                # Check how many new messages since last summary
                new_messages = db.query(Message).filter(
                    Message.conversation_id == conversation_id,
                    Message.sent_at > latest_summary.created_at
                ).count()
                return new_messages >= self.SUMMARY_THRESHOLD
            
            return message_count >= self.SUMMARY_THRESHOLD
            
        finally:
            db.close()
    
    def generate_summary(self, conversation_id: int) -> Optional[str]:
        """Generate a summary for a conversation."""
        from openai import OpenAI
        
        client = OpenAI(api_key=settings.OPENAI_API_KEY)
        db = SessionLocal()
        
        try:
            # Get messages to summarize
            messages = db.query(Message).filter(
                Message.conversation_id == conversation_id
            ).order_by(Message.sent_at.asc()).all()
            
            if len(messages) < self.SUMMARY_THRESHOLD:
                return None
            
            # Build message text
            message_text = ""
            for msg in messages:
                role = "Guest" if msg.direction == MessageDirection.inbound else "Host"
                time_str = msg.sent_at.strftime("%b %d %H:%M") if msg.sent_at else ""
                message_text += f"[{time_str}] {role}: {msg.content}\n"
            
            # Generate summary
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                response_format={"type": "json_object"},
                messages=[
                    {
                        "role": "system",
                        "content": """Summarize this property management conversation.

OUTPUT FORMAT (JSON):
{
    "summary": "2-4 sentence summary of the conversation",
    "key_topics": ["topic1", "topic2"],
    "resolved_issues": ["issue that was resolved"],
    "pending_issues": ["issue still open"],
    "sentiment": "positive/neutral/negative"
}"""
                    },
                    {"role": "user", "content": message_text}
                ],
                temperature=0.3,
                max_tokens=400
            )
            
            result = json.loads(response.choices[0].message.content)
            
            # Save summary
            summary = ConversationSummary(
                conversation_id=conversation_id,
                summary=result.get("summary", ""),
                key_topics=json.dumps(result.get("key_topics", [])),
                resolved_issues=json.dumps(result.get("resolved_issues", [])),
                pending_issues=json.dumps(result.get("pending_issues", [])),
                sentiment=result.get("sentiment", "neutral"),
                messages_summarized=len(messages),
                summary_start_at=messages[0].sent_at if messages else None,
                summary_end_at=messages[-1].sent_at if messages else None
            )
            db.add(summary)
            db.commit()
            
            return result.get("summary")
            
        except Exception as e:
            print(f"[Summarizer] Error generating summary: {e}")
            return None
        finally:
            db.close()
    
    def get_latest_summary(self, conversation_id: int) -> Optional[Dict[str, Any]]:
        """Get the latest summary for a conversation."""
        db = SessionLocal()
        try:
            summary = db.query(ConversationSummary).filter(
                ConversationSummary.conversation_id == conversation_id
            ).order_by(ConversationSummary.created_at.desc()).first()
            
            if summary:
                return {
                    "summary": summary.summary,
                    "key_topics": json.loads(summary.key_topics or "[]"),
                    "resolved_issues": json.loads(summary.resolved_issues or "[]"),
                    "pending_issues": json.loads(summary.pending_issues or "[]"),
                    "sentiment": summary.sentiment,
                    "messages_summarized": summary.messages_summarized,
                    "created_at": summary.created_at.isoformat()
                }
            return None
        finally:
            db.close()


# Global instance
conversation_summarizer = ConversationSummarizer()


# ============ INITIALIZATION ============

async def initialize_knowledge_system():
    """Initialize the knowledge system on startup."""
    print("[Knowledge] Initializing knowledge system...")
    
    # Index style examples
    from embeddings import initialize_style_examples
    initialize_style_examples()
    
    # Load any existing property knowledge
    db = SessionLocal()
    try:
        # Get all unique property IDs
        properties = db.query(GuestIndex.listing_id, GuestIndex.listing_name).distinct().all()
        
        for prop_id, prop_name in properties:
            if prop_id:
                count = knowledge_manager.load_and_index_property(prop_id, prop_name or "")
                if count > 0:
                    print(f"[Knowledge] Indexed {count} docs for property {prop_id}")
    finally:
        db.close()
    
    # Print stats
    stats = get_collection_stats()
    print(f"[Knowledge] Vector DB stats: {stats}")
