"""
Utility functions for AI Property Manager.
"""

import re
import json
from datetime import datetime
from typing import Optional, List

import pytz
from sqlalchemy.orm import Session

from config import settings
from models import (
    SessionLocal, Conversation, Message, SystemLog, 
    GuestIndex, MessageDirection, ConversationStatus
)


def normalize_phone(phone: str) -> str:
    """
    Normalize phone number to E.164 format.
    
    Examples:
        "(555) 123-4567" -> "+15551234567"
        "5551234567" -> "+15551234567"
        "+1-555-123-4567" -> "+15551234567"
    """
    if not phone:
        return ""
    
    # Remove all non-digit characters
    digits = re.sub(r'\D', '', phone)
    
    # Add country code for US numbers
    if len(digits) == 10:
        digits = "1" + digits
    
    return "+" + digits


def is_business_hours() -> bool:
    """Check if current time is within business hours."""
    tz = pytz.timezone(settings.TIMEZONE)
    now = datetime.now(tz)
    return settings.BUSINESS_HOURS_START <= now.hour < settings.BUSINESS_HOURS_END


def log_event(
    event_type: str, 
    conversation_id: Optional[int] = None, 
    guest_phone: Optional[str] = None, 
    payload: Optional[dict] = None,
    db: Optional[Session] = None
):
    """
    Log an event to the SystemLog table.
    
    Event types:
        - message_received, ai_response_generated, auto_sent, draft_created
        - escalated_l1, escalated_l2, escalation_timeout
        - human_approved, human_edited, human_snoozed, human_resolved
        - api_error, cache_miss, force_sync_triggered
        - external_reply_detected, unknown_guest, rate_limit_hit
        - webhook_received, webhook_verification_failed
    """
    close_db = False
    if db is None:
        db = SessionLocal()
        close_db = True
    
    try:
        log = SystemLog(
            event_type=event_type,
            conversation_id=conversation_id,
            guest_phone=guest_phone,
            payload=json.dumps(payload) if payload else None
        )
        db.add(log)
        db.commit()
    finally:
        if close_db:
            db.close()


def get_or_create_conversation(guest_phone: str, db: Session) -> Conversation:
    """
    Get existing conversation for a guest or create a new one.
    Only considers active conversations (not resolved).
    """
    conversation = db.query(Conversation).filter(
        Conversation.guest_phone == guest_phone,
        Conversation.status != ConversationStatus.resolved
    ).first()
    
    if not conversation:
        conversation = Conversation(guest_phone=guest_phone)
        db.add(conversation)
        db.commit()
        db.refresh(conversation)
    
    return conversation


def save_message(
    conversation_id: int,
    content: str,
    direction: str,
    source: str,
    db: Session,
    external_id: Optional[str] = None,
    ai_confidence: Optional[float] = None,
    ai_reasoning: Optional[str] = None,
    was_auto_sent: bool = False,
    was_human_edited: bool = False
) -> Message:
    """Save a message to the database."""
    message = Message(
        conversation_id=conversation_id,
        content=content,
        direction=MessageDirection[direction],
        source=source,
        external_id=external_id,
        ai_confidence=ai_confidence,
        ai_reasoning=ai_reasoning,
        was_auto_sent=was_auto_sent,
        was_human_edited=was_human_edited
    )
    db.add(message)
    db.commit()
    db.refresh(message)
    return message


def message_already_processed(msg_id: str, db: Session) -> bool:
    """Check if a message with this external ID has already been processed."""
    if not msg_id:
        return False
    existing = db.query(Message).filter(Message.external_id == msg_id).first()
    return existing is not None


def get_conversation(conversation_id: int, db: Session) -> Optional[Conversation]:
    """Get a conversation by ID."""
    return db.query(Conversation).filter(Conversation.id == conversation_id).first()


def get_recent_messages(conversation_id: int, db: Session, hours: int = 24) -> List[Message]:
    """Get recent messages for a conversation."""
    cutoff = datetime.utcnow() - __import__('datetime').timedelta(hours=hours)
    return db.query(Message).filter(
        Message.conversation_id == conversation_id,
        Message.sent_at >= cutoff
    ).order_by(Message.sent_at.asc()).all()


def get_last_inbound_message(conversation_id: int, db: Session) -> Optional[str]:
    """Get the content of the last inbound message."""
    message = db.query(Message).filter(
        Message.conversation_id == conversation_id,
        Message.direction == MessageDirection.inbound
    ).order_by(Message.sent_at.desc()).first()
    return message.content if message else None


def get_last_message(conversation_id: int, db: Session) -> Optional[str]:
    """Get the content of the last message (any direction)."""
    message = db.query(Message).filter(
        Message.conversation_id == conversation_id
    ).order_by(Message.sent_at.desc()).first()
    return message.content if message else None


def get_guest_from_cache(guest_phone: str, db: Session) -> Optional[GuestIndex]:
    """Look up guest information from the local cache."""
    return db.query(GuestIndex).filter(
        GuestIndex.guest_phone == guest_phone
    ).order_by(GuestIndex.synced_at.desc()).first()


def check_escalation_keywords(message: str) -> Optional[str]:
    """
    Check if message contains any escalation keywords.
    Returns the matched keyword or None.
    """
    message_lower = message.lower()
    for keyword in settings.escalation_keywords_list:
        if keyword in message_lower:
            return keyword
    return None


def check_rate_limit(conversation: Conversation, db: Session) -> bool:
    """
    Check if we've exceeded the outbound rate limit.
    Returns True if sending is allowed, False if rate limited.
    """
    now = datetime.utcnow()
    
    # Reset counter if hour has passed
    if (conversation.outbound_hour_started is None or 
        (now - conversation.outbound_hour_started).total_seconds() > 3600):
        conversation.outbound_count_this_hour = 0
        conversation.outbound_hour_started = now
        db.commit()
        return True
    
    if conversation.outbound_count_this_hour >= settings.MAX_OUTBOUND_PER_HOUR:
        return False
    
    return True


# Draft cache (simple in-memory, could use Redis in production)
_draft_cache: dict = {}


def cache_draft(conversation_id: int, draft: str):
    """Cache a draft reply for a conversation."""
    _draft_cache[conversation_id] = draft


def get_cached_draft(conversation_id: int) -> Optional[str]:
    """Get a cached draft for a conversation."""
    return _draft_cache.get(conversation_id)


def clear_cached_draft(conversation_id: int):
    """Clear a cached draft for a conversation."""
    _draft_cache.pop(conversation_id, None)
