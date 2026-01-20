"""
Admin API endpoints for the dashboard.
Provides read/write access to conversations, logs, and testing utilities.
"""

from datetime import datetime, timedelta
from typing import Optional, List
from pydantic import BaseModel

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from sqlalchemy import desc, func

from models import (
    get_db, Conversation, Message, GuestIndex, SystemLog,
    ConversationStatus, MessageDirection
)
from utils import normalize_phone, log_event
from config import settings

router = APIRouter(prefix="/api/admin", tags=["admin"])


# ============ RESPONSE MODELS ============

class ConversationSummary(BaseModel):
    id: int
    guest_phone: str
    guest_name: Optional[str]
    listing_name: Optional[str]
    status: str
    last_message_at: Optional[datetime]
    message_count: int
    last_message_preview: Optional[str]
    check_in_date: Optional[datetime] = None
    check_out_date: Optional[datetime] = None
    booking_source: Optional[str] = None

    class Config:
        from_attributes = True


class MessageDetail(BaseModel):
    id: int
    direction: str
    source: str
    content: str
    attachment_url: Optional[str] = None  # Image/file attachments
    sent_at: datetime
    ai_confidence: Optional[float]
    was_auto_sent: bool
    was_human_edited: bool
    # AI suggestion (for inbound messages)
    ai_suggested_reply: Optional[str] = None
    ai_suggestion_confidence: Optional[float] = None
    ai_suggestion_reasoning: Optional[str] = None

    class Config:
        from_attributes = True


class SuggestedReply(BaseModel):
    text: str
    confidence: float
    reasoning: Optional[str] = None
    needs_human_review: bool = False

class ConversationDetail(BaseModel):
    id: int
    guest_phone: str
    guest_name: Optional[str]
    listing_name: Optional[str]
    hostify_reservation_id: Optional[str]
    status: str
    created_at: datetime
    last_message_at: Optional[datetime]
    slack_thread_ts: Optional[str]
    check_in_date: Optional[datetime] = None
    check_out_date: Optional[datetime] = None
    booking_source: Optional[str] = None
    messages: List[MessageDetail]
    needs_response: bool = False  # True if last message is from guest
    suggested_reply: Optional[SuggestedReply] = None

    class Config:
        from_attributes = True


class GuestIndexItem(BaseModel):
    id: int
    guest_phone: Optional[str]
    guest_name: str
    reservation_id: str
    listing_name: str
    check_in_date: Optional[datetime]
    check_out_date: Optional[datetime]
    synced_at: datetime
    source: Optional[str] = None
    is_phone_verified: bool = False

    class Config:
        from_attributes = True


class LinkPhoneRequest(BaseModel):
    reservation_id: str
    phone: str


class LogEntry(BaseModel):
    id: int
    timestamp: datetime
    event_type: str
    conversation_id: Optional[int]
    guest_phone: Optional[str]
    payload: Optional[str]

    class Config:
        from_attributes = True


class SimulateMessageRequest(BaseModel):
    phone: str
    message: str
    source: str = "test"


class DashboardStats(BaseModel):
    total_conversations: int
    active_conversations: int
    escalated_conversations: int
    resolved_today: int
    messages_today: int
    auto_sent_today: int
    guests_in_cache: int


# ============ ENDPOINTS ============

@router.get("/stats", response_model=DashboardStats)
async def get_dashboard_stats(db: Session = Depends(get_db)):
    """Get dashboard statistics."""
    today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    
    total_conversations = db.query(Conversation).count()
    
    active_conversations = db.query(Conversation).filter(
        Conversation.status == ConversationStatus.active
    ).count()
    
    escalated_conversations = db.query(Conversation).filter(
        Conversation.status.in_([
            ConversationStatus.escalated_l1,
            ConversationStatus.escalated_l2
        ])
    ).count()
    
    resolved_today = db.query(Conversation).filter(
        Conversation.status == ConversationStatus.resolved,
        Conversation.last_message_at >= today
    ).count()
    
    messages_today = db.query(Message).filter(
        Message.sent_at >= today
    ).count()
    
    auto_sent_today = db.query(Message).filter(
        Message.sent_at >= today,
        Message.was_auto_sent == True,
        Message.direction == MessageDirection.outbound
    ).count()
    
    guests_in_cache = db.query(GuestIndex).count()
    
    return DashboardStats(
        total_conversations=total_conversations,
        active_conversations=active_conversations,
        escalated_conversations=escalated_conversations,
        resolved_today=resolved_today,
        messages_today=messages_today,
        auto_sent_today=auto_sent_today,
        guests_in_cache=guests_in_cache
    )


@router.get("/conversations", response_model=List[ConversationSummary])
async def list_conversations(
    status: Optional[str] = None,
    limit: int = 50,
    db: Session = Depends(get_db)
):
    """List all conversations with optional status filter."""
    query = db.query(Conversation)
    
    if status:
        try:
            status_enum = ConversationStatus[status]
            query = query.filter(Conversation.status == status_enum)
        except KeyError:
            pass
    
    conversations = query.order_by(desc(Conversation.last_message_at)).limit(limit).all()
    
    result = []
    for conv in conversations:
        # Get message count and last message
        message_count = db.query(Message).filter(
            Message.conversation_id == conv.id
        ).count()
        
        last_msg = db.query(Message).filter(
            Message.conversation_id == conv.id
        ).order_by(desc(Message.sent_at)).first()
        
        result.append(ConversationSummary(
            id=conv.id,
            guest_phone=conv.guest_phone,
            guest_name=conv.guest_name,
            listing_name=conv.listing_name,
            status=conv.status.value if conv.status else "unknown",
            last_message_at=conv.last_message_at,
            message_count=message_count,
            last_message_preview=last_msg.content[:100] if last_msg else None,
            check_in_date=conv.check_in_date,
            check_out_date=conv.check_out_date,
            booking_source=conv.booking_source
        ))
    
    return result


@router.get("/conversations/{conversation_id}", response_model=ConversationDetail)
async def get_conversation(conversation_id: int, db: Session = Depends(get_db)):
    """Get detailed conversation with all messages and AI suggestion if needed."""
    conv = db.query(Conversation).filter(Conversation.id == conversation_id).first()
    
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    messages = db.query(Message).filter(
        Message.conversation_id == conversation_id
    ).order_by(Message.sent_at).all()
    
    # Check if last message is from guest (needs response)
    needs_response = False
    
    if messages:
        last_msg = messages[-1]
        if last_msg.direction and last_msg.direction.value == "inbound":
            needs_response = True
    
    return ConversationDetail(
        id=conv.id,
        guest_phone=conv.guest_phone,
        guest_name=conv.guest_name,
        listing_name=conv.listing_name,
        hostify_reservation_id=conv.hostify_reservation_id,
        status=conv.status.value if conv.status else "unknown",
        created_at=conv.created_at,
        last_message_at=conv.last_message_at,
        slack_thread_ts=conv.slack_thread_ts,
        check_in_date=conv.check_in_date,
        check_out_date=conv.check_out_date,
        booking_source=conv.booking_source,
        needs_response=needs_response,
        suggested_reply=None,  # Generated on-demand via /generate-suggestion endpoint
        messages=[
            MessageDetail(
                id=m.id,
                direction=m.direction.value if m.direction else "unknown",
                source=m.source,
                content=m.content,
                attachment_url=m.attachment_url,  # Include image attachments
                sent_at=m.sent_at,
                ai_confidence=m.ai_confidence,
                was_auto_sent=m.was_auto_sent,
                was_human_edited=m.was_human_edited,
                # Include AI suggestion for inbound messages
                ai_suggested_reply=m.ai_suggested_reply,
                ai_suggestion_confidence=m.ai_suggestion_confidence,
                ai_suggestion_reasoning=m.ai_suggestion_reasoning
            )
            for m in messages
        ]
    )


@router.post("/conversations/{conversation_id}/resolve")
async def resolve_conversation(conversation_id: int, db: Session = Depends(get_db)):
    """Mark a conversation as resolved."""
    conv = db.query(Conversation).filter(Conversation.id == conversation_id).first()
    
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    conv.status = ConversationStatus.resolved
    conv.last_human_action_at = datetime.utcnow()
    db.commit()
    
    log_event("human_resolved", conversation_id=conversation_id, payload={"source": "dashboard"})
    
    return {"status": "resolved"}


@router.post("/conversations/{conversation_id}/generate-suggestion")
async def generate_suggestion(
    conversation_id: int, 
    for_learning: bool = False,
    db: Session = Depends(get_db)
):
    """
    Generate an AI suggestion for a conversation on-demand.
    
    Args:
        conversation_id: The conversation to generate for
        for_learning: If True, generates suggestion for last guest message even if host has replied
                     This allows comparing AI suggestions to actual host responses
    """
    from brain import generate_ai_response, get_style_examples
    from models import GuestIndex
    
    conv = db.query(Conversation).filter(Conversation.id == conversation_id).first()
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    all_messages = db.query(Message).filter(
        Message.conversation_id == conversation_id
    ).order_by(Message.sent_at).all()
    
    if not all_messages:
        return {"error": "No messages in conversation"}
    
    try:
        # Build guest context directly from conversation data (not the cache!)
        guest_context = GuestIndex(
            guest_phone=conv.guest_phone or "unknown",
            guest_name=conv.guest_name,
            listing_id=conv.listing_id,  # Include listing_id for RAG lookup
            listing_name=conv.listing_name,
            check_in_date=conv.check_in_date,
            check_out_date=conv.check_out_date,
            source=conv.booking_source or "unknown"
        )
        
        # For learning mode: find the last guest message and generate what AI would have said
        # This allows comparing AI suggestion to actual host response
        actual_host_reply = None
        messages_for_ai = all_messages
        
        if for_learning or (all_messages[-1].direction and all_messages[-1].direction.value == "outbound"):
            # Find the last guest message
            last_guest_idx = None
            for i in range(len(all_messages) - 1, -1, -1):
                if all_messages[i].direction and all_messages[i].direction.value == "inbound":
                    last_guest_idx = i
                    break
            
            if last_guest_idx is not None:
                # Only include messages up to and including the last guest message
                messages_for_ai = all_messages[:last_guest_idx + 1]
                
                # Capture the actual host reply (if any) for comparison
                if last_guest_idx < len(all_messages) - 1:
                    next_msg = all_messages[last_guest_idx + 1]
                    if next_msg.direction and next_msg.direction.value == "outbound":
                        actual_host_reply = next_msg.content
        
        if not messages_for_ai:
            return {"error": "No guest messages found"}
        
        # Get style examples based on last guest message
        last_guest_msg = messages_for_ai[-1].content if messages_for_ai else ""
        style_examples = get_style_examples(last_guest_msg, n=3)
        
        # Generate suggestion
        ai_response = await generate_ai_response(
            messages=messages_for_ai,
            guest_context=guest_context,
            style_examples=style_examples,
            conversation_id=conversation_id,
            use_advanced=False  # Use simple mode for quick suggestions
        )
        
        result = {
            "suggested_reply": {
                "text": ai_response.reply_text,
                "confidence": ai_response.confidence_score,
                "reasoning": ai_response.reasoning,
                "needs_human_review": ai_response.requires_human
            }
        }
        
        # Include actual host reply for learning/comparison
        if actual_host_reply:
            result["actual_host_reply"] = actual_host_reply
            result["for_learning"] = True
        
        return result
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "error": str(e),
            "suggested_reply": None
        }


@router.post("/conversations/{conversation_id}/generate-and-save-suggestion")
async def generate_and_save_suggestion(conversation_id: int, db: Session = Depends(get_db)):
    """
    Generate an AI suggestion and save it to the last guest message.
    
    This is used when the user clicks "Generate AI Suggestion" in the UI.
    The suggestion is saved to the database so it persists and can be
    compared to the actual host response later.
    """
    from brain import generate_ai_response, get_style_examples
    from models import GuestIndex
    
    conv = db.query(Conversation).filter(Conversation.id == conversation_id).first()
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    # Get all messages
    all_messages = db.query(Message).filter(
        Message.conversation_id == conversation_id
    ).order_by(Message.sent_at).all()
    
    if not all_messages:
        return {"error": "No messages in conversation"}
    
    # Find the last guest message
    last_guest_msg = None
    for msg in reversed(all_messages):
        if msg.direction and msg.direction.value == "inbound":
            last_guest_msg = msg
            break
    
    if not last_guest_msg:
        return {"error": "No guest messages found"}
    
    try:
        # Build guest context
        guest_context = GuestIndex(
            guest_phone=conv.guest_phone or "unknown",
            guest_name=conv.guest_name,
            listing_id=conv.listing_id,  # Include listing_id for RAG lookup
            listing_name=conv.listing_name,
            check_in_date=conv.check_in_date,
            check_out_date=conv.check_out_date,
            source=conv.booking_source or "unknown"
        )
        
        # Get messages up to and including the last guest message
        msg_idx = all_messages.index(last_guest_msg)
        messages_for_ai = all_messages[:msg_idx + 1]
        
        # Get style examples
        style_examples = get_style_examples(last_guest_msg.content, n=3)
        
        # Generate suggestion
        ai_response = await generate_ai_response(
            messages=messages_for_ai,
            guest_context=guest_context,
            style_examples=style_examples,
            conversation_id=conversation_id,
            use_advanced=False
        )
        
        # Save to the message
        last_guest_msg.ai_suggested_reply = ai_response.reply_text
        last_guest_msg.ai_suggestion_confidence = ai_response.confidence_score
        last_guest_msg.ai_suggestion_reasoning = ai_response.reasoning
        last_guest_msg.ai_suggestion_generated_at = datetime.utcnow()
        db.commit()
        
        log_event("ai_suggestion_saved", payload={
            "message_id": last_guest_msg.id,
            "conversation_id": conversation_id,
            "confidence": ai_response.confidence_score
        })
        
        return {
            "success": True,
            "message_id": last_guest_msg.id,
            "suggested_reply": {
                "text": ai_response.reply_text,
                "confidence": ai_response.confidence_score,
                "reasoning": ai_response.reasoning,
                "needs_human_review": ai_response.requires_human
            }
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e), "success": False}


@router.post("/messages/{message_id}/generate-suggestion")
async def generate_suggestion_for_message(message_id: int, db: Session = Depends(get_db)):
    """
    Generate an AI suggestion for a specific guest message.
    
    This allows generating suggestions for any historical guest message,
    enabling comparison with what the host actually replied.
    """
    from brain import generate_ai_response, get_style_examples
    from models import GuestIndex
    
    # Get the target message
    target_msg = db.query(Message).filter(Message.id == message_id).first()
    if not target_msg:
        raise HTTPException(status_code=404, detail="Message not found")
    
    if target_msg.direction != MessageDirection.inbound:
        return {"error": "Can only generate suggestions for guest (inbound) messages"}
    
    # Check if already has suggestion
    if target_msg.ai_suggested_reply:
        return {
            "success": True,
            "message_id": message_id,
            "already_exists": True,
            "suggested_reply": {
                "text": target_msg.ai_suggested_reply,
                "confidence": target_msg.ai_suggestion_confidence,
                "reasoning": target_msg.ai_suggestion_reasoning
            }
        }
    
    # Get the conversation
    conv = db.query(Conversation).filter(Conversation.id == target_msg.conversation_id).first()
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    # Get all messages up to and including the target message
    messages_for_ai = db.query(Message).filter(
        Message.conversation_id == conv.id,
        Message.sent_at <= target_msg.sent_at
    ).order_by(Message.sent_at).all()
    
    try:
        # Build guest context
        guest_context = GuestIndex(
            guest_phone=conv.guest_phone or "unknown",
            guest_name=conv.guest_name,
            listing_id=conv.listing_id,  # Include listing_id for RAG lookup
            listing_name=conv.listing_name,
            check_in_date=conv.check_in_date,
            check_out_date=conv.check_out_date,
            source=conv.booking_source or "unknown"
        )
        
        # Get style examples
        style_examples = get_style_examples(target_msg.content, n=3)
        
        # Generate suggestion
        ai_response = await generate_ai_response(
            messages=messages_for_ai,
            guest_context=guest_context,
            style_examples=style_examples,
            conversation_id=conv.id,
            use_advanced=False
        )
        
        # Save to the message
        target_msg.ai_suggested_reply = ai_response.reply_text
        target_msg.ai_suggestion_confidence = ai_response.confidence_score
        target_msg.ai_suggestion_reasoning = ai_response.reasoning
        target_msg.ai_suggestion_generated_at = datetime.utcnow()
        db.commit()
        
        log_event("ai_suggestion_saved", payload={
            "message_id": message_id,
            "conversation_id": conv.id,
            "confidence": ai_response.confidence_score
        })
        
        return {
            "success": True,
            "message_id": message_id,
            "suggested_reply": {
                "text": ai_response.reply_text,
                "confidence": ai_response.confidence_score,
                "reasoning": ai_response.reasoning,
                "needs_human_review": ai_response.requires_human
            }
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e), "success": False}


@router.get("/guests", response_model=List[GuestIndexItem])
async def list_guests(
    limit: int = 100, 
    has_phone: Optional[bool] = None,
    search: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """
    List cached guests.
    
    Args:
        limit: Max results to return
        has_phone: Filter by phone availability (true/false/null for all)
        search: Search by name, property, or reservation ID
    """
    query = db.query(GuestIndex)
    
    if has_phone is True:
        query = query.filter(GuestIndex.guest_phone.isnot(None))
    elif has_phone is False:
        query = query.filter(GuestIndex.guest_phone.is_(None))
    
    if search:
        search_term = f"%{search}%"
        query = query.filter(
            (GuestIndex.guest_name.ilike(search_term)) |
            (GuestIndex.listing_name.ilike(search_term)) |
            (GuestIndex.reservation_id.ilike(search_term))
        )
    
    guests = query.order_by(desc(GuestIndex.check_in_date)).limit(limit).all()
    
    return [
        GuestIndexItem(
            id=g.id,
            guest_phone=g.guest_phone,
            guest_name=g.guest_name,
            reservation_id=g.reservation_id,
            listing_name=g.listing_name,
            check_in_date=g.check_in_date,
            check_out_date=g.check_out_date,
            synced_at=g.synced_at,
            source=getattr(g, 'source', None),
            is_phone_verified=getattr(g, 'is_phone_verified', False)
        )
        for g in guests
    ]


@router.post("/guests/link-phone")
async def link_phone_to_guest(request: LinkPhoneRequest, db: Session = Depends(get_db)):
    """
    Manually link a phone number to a guest reservation.
    Used when a guest messages from an unknown number and identifies their reservation.
    """
    guest = db.query(GuestIndex).filter(
        GuestIndex.reservation_id == request.reservation_id
    ).first()
    
    if not guest:
        raise HTTPException(status_code=404, detail="Reservation not found")
    
    normalized = normalize_phone(request.phone)
    if not normalized:
        raise HTTPException(status_code=400, detail="Invalid phone number")
    
    guest.guest_phone = normalized
    guest.is_phone_verified = False  # Manual link, not from API
    db.commit()
    
    log_event("guest_phone_linked", payload={
        "reservation_id": request.reservation_id,
        "phone": normalized,
        "guest_name": guest.guest_name
    })
    
    return {
        "status": "linked",
        "reservation_id": request.reservation_id,
        "phone": normalized,
        "guest_name": guest.guest_name
    }


@router.get("/guests/unmatched")
async def list_unmatched_guests(limit: int = 50, db: Session = Depends(get_db)):
    """
    List guests without phone numbers (need manual matching).
    Shows upcoming/current reservations first.
    """
    from datetime import datetime
    
    now = datetime.utcnow()
    
    guests = db.query(GuestIndex).filter(
        GuestIndex.guest_phone.is_(None),
        GuestIndex.check_out_date >= now  # Only current/future reservations
    ).order_by(GuestIndex.check_in_date).limit(limit).all()
    
    return [
        {
            "id": g.id,
            "guest_name": g.guest_name,
            "reservation_id": g.reservation_id,
            "listing_name": g.listing_name,
            "check_in_date": g.check_in_date.isoformat() if g.check_in_date else None,
            "check_out_date": g.check_out_date.isoformat() if g.check_out_date else None,
            "source": getattr(g, 'source', None)
        }
        for g in guests
    ]


@router.get("/logs", response_model=List[LogEntry])
async def list_logs(
    event_type: Optional[str] = None,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """List system logs."""
    query = db.query(SystemLog)
    
    if event_type:
        query = query.filter(SystemLog.event_type == event_type)
    
    logs = query.order_by(desc(SystemLog.timestamp)).limit(limit).all()
    
    return [
        LogEntry(
            id=log.id,
            timestamp=log.timestamp,
            event_type=log.event_type,
            conversation_id=log.conversation_id,
            guest_phone=log.guest_phone,
            payload=log.payload
        )
        for log in logs
    ]


@router.post("/simulate")
async def simulate_message(request: SimulateMessageRequest, db: Session = Depends(get_db)):
    """
    Simulate an incoming message for testing.
    This bypasses the webhook and directly queues a message for processing.
    """
    from main import queue_message_for_processing
    
    phone = normalize_phone(request.phone)
    
    if not phone:
        raise HTTPException(status_code=400, detail="Invalid phone number")
    
    # Queue the message for processing
    await queue_message_for_processing(
        guest_phone=phone,
        content=request.message,
        source=request.source,
        msg_id=f"test_{datetime.utcnow().timestamp()}"
    )
    
    log_event("test_message_simulated", guest_phone=phone, payload={
        "message": request.message,
        "source": request.source
    })
    
    return {
        "status": "queued",
        "phone": phone,
        "message": request.message,
        "note": f"Message will be processed in {settings.MESSAGE_BURST_DELAY_SECONDS} seconds"
    }


@router.post("/sync")
async def trigger_sync():
    """Manually trigger a reservation sync."""
    from cache import sync_reservations
    
    await sync_reservations()
    
    return {"status": "sync_complete"}


@router.post("/sync/messages")
async def trigger_message_sync(limit: int = 200):
    """
    Sync messages from Hostify inbox threads.
    Creates Conversation and Message records.
    
    Args:
        limit: Max number of inbox threads to sync (default 200)
    """
    from cache import sync_messages
    
    await sync_messages(limit_threads=limit)
    
    return {"status": "message_sync_complete", "limit": limit}


@router.post("/sync/messages/recent")
async def sync_recent_messages(days: int = 3, db: Session = Depends(get_db)):
    """
    Fetch and save messages from the last N days as if they came in via webhook.
    This backfills recent message history.
    
    Args:
        days: Number of days to look back (default 3)
    """
    from cache import hostify_client, _parse_datetime
    from models import Conversation, Message, ConversationStatus
    from utils import normalize_phone
    import asyncio
    
    cutoff = datetime.utcnow() - timedelta(days=days)
    
    print(f"[Sync Recent] Fetching messages from last {days} days (since {cutoff})...")
    
    # Fetch all threads
    all_threads = []
    page = 1
    per_page = 100
    
    while True:
        threads = await hostify_client.get_inbox_threads(limit=100, page=page)
        if not threads:
            break
        all_threads.extend(threads)
        print(f"[Sync Recent] Fetched page {page}: {len(threads)} threads")
        page += 1
        await asyncio.sleep(0.1)
    
    print(f"[Sync Recent] Total threads: {len(all_threads)}")
    
    # Filter to threads with recent activity
    recent_threads = []
    for t in all_threads:
        last_msg = t.get("last_message")
        if last_msg:
            last_msg_time = _parse_datetime(last_msg)
            if last_msg_time and last_msg_time >= cutoff:
                recent_threads.append(t)
    
    print(f"[Sync Recent] Threads with activity in last {days} days: {len(recent_threads)}")
    
    conversations_created = 0
    messages_synced = 0
    
    for i, thread in enumerate(recent_threads):
        try:
            inbox_id = thread.get("id")
            reservation_id = str(thread.get("reservation_id", ""))
            guest_name = thread.get("guest_name", "Guest")
            guest_phone = str(thread.get("guest_phone", "")) if thread.get("guest_phone") else None
            listing_name = thread.get("listing_title") or thread.get("listing", "")
            listing_id = str(thread.get("listing_id", ""))
            
            checkin_str = thread.get("checkin")
            checkout_str = thread.get("checkout")
            check_in_date = _parse_datetime(checkin_str) if checkin_str else None
            check_out_date = _parse_datetime(checkout_str) if checkout_str else None
            booking_source = thread.get("integration_type_name")
            
            # Find or create conversation
            conversation = None
            if reservation_id:
                conversation = db.query(Conversation).filter(
                    Conversation.hostify_reservation_id == reservation_id
                ).first()
            
            if not conversation and guest_phone:
                normalized_phone = normalize_phone(guest_phone)
                if normalized_phone:
                    conversation = db.query(Conversation).filter(
                        Conversation.guest_phone == normalized_phone
                    ).first()
            
            if not conversation:
                phone_to_use = normalize_phone(guest_phone) if guest_phone else f"inbox_{inbox_id}"
                conversation = Conversation(
                    guest_phone=phone_to_use,
                    guest_name=guest_name,
                    listing_id=listing_id,
                    listing_name=listing_name,
                    hostify_reservation_id=reservation_id,
                    check_in_date=check_in_date,
                    check_out_date=check_out_date,
                    booking_source=booking_source,
                    status=ConversationStatus.active,
                    created_at=datetime.utcnow()
                )
                db.add(conversation)
                db.flush()
                conversations_created += 1
            else:
                # Update existing
                if listing_name and not conversation.listing_name:
                    conversation.listing_name = listing_name
                if listing_id and not conversation.listing_id:
                    conversation.listing_id = listing_id
                if not conversation.check_in_date and check_in_date:
                    conversation.check_in_date = check_in_date
                if not conversation.check_out_date and check_out_date:
                    conversation.check_out_date = check_out_date
            
            # Fetch messages
            messages = await hostify_client.get_inbox_messages(inbox_id, limit=100)
            
            for msg_data in messages:
                msg_id = str(msg_data.get("id", ""))
                
                # Skip if exists
                if db.query(Message).filter(Message.external_id == msg_id).first():
                    continue
                
                # Determine direction
                sender_field = msg_data.get("from", "")
                if sender_field == "guest":
                    direction = "inbound"
                else:
                    direction = "outbound"
                
                msg_time = msg_data.get("created") or msg_data.get("sent_at")
                sent_at = _parse_datetime(msg_time) if msg_time else datetime.utcnow()
                
                content = msg_data.get("message") or msg_data.get("body") or ""
                attachment_url = msg_data.get("attachment_url") or msg_data.get("image")
                
                if not content and not attachment_url:
                    continue
                
                message = Message(
                    conversation_id=conversation.id,
                    direction=direction,
                    source="hostify",
                    content=content,
                    attachment_url=attachment_url,
                    external_id=msg_id,
                    sent_at=sent_at,
                    was_auto_sent=False,
                    was_human_edited=False
                )
                db.add(message)
                messages_synced += 1
            
            db.commit()
            
            if messages:
                print(f"[Sync Recent] [{i+1}/{len(recent_threads)}] {guest_name}: {len(messages)} messages")
            
            await asyncio.sleep(0.1)
            
        except Exception as e:
            print(f"[Sync Recent] Error on thread {thread.get('id')}: {e}")
            db.rollback()
            continue
    
    print(f"[Sync Recent] Done! Conversations: {conversations_created}, Messages: {messages_synced}")
    
    return {
        "status": "complete",
        "days": days,
        "threads_processed": len(recent_threads),
        "conversations_created": conversations_created,
        "messages_synced": messages_synced
    }


@router.post("/sync/test")
async def test_sync_one_listing(db: Session = Depends(get_db)):
    """
    Test sync with just ONE listing to verify everything works.
    """
    from cache import hostify_client, _extract_guest_data_all, _upsert_guest_index
    
    # Fetch just one listing
    listings = await hostify_client.get_listings()
    if not listings:
        return {"error": "No listings found"}
    
    listing = listings[0]
    print(f"[Test] Testing with listing: {listing.get('name')}")
    
    # Fetch reservations for this listing
    from datetime import datetime, timedelta
    now = datetime.utcnow()
    start_date = (now - timedelta(days=30)).strftime("%Y-%m-%d")
    end_date = (now + timedelta(days=30)).strftime("%Y-%m-%d")
    
    reservations = await hostify_client.get_reservations_for_listing(
        listing.get("id"),
        start_date,
        end_date
    )
    
    print(f"[Test] Found {len(reservations)} reservations")
    
    saved = 0
    errors = []
    
    for res in reservations[:5]:  # Just first 5
        try:
            guest_data = _extract_guest_data_all(res, listing)
            if guest_data:
                _upsert_guest_index(db, guest_data)
                saved += 1
                print(f"[Test] Saved: {guest_data.get('guest_name')} - {guest_data.get('reservation_id')}")
            else:
                errors.append(f"No data extracted for {res.get('id')}")
        except Exception as e:
            errors.append(f"Error: {str(e)}")
    
    return {
        "listing": listing.get("name"),
        "reservations_found": len(reservations),
        "saved": saved,
        "errors": errors
    }


@router.get("/config")
async def get_config():
    """Get current configuration (non-sensitive)."""
    return {
        "dry_run_mode": settings.DRY_RUN_MODE,
        "confidence_auto_send": settings.CONFIDENCE_AUTO_SEND,
        "confidence_soft_escalate": settings.CONFIDENCE_SOFT_ESCALATE,
        "message_burst_delay_seconds": settings.MESSAGE_BURST_DELAY_SECONDS,
        "escalation_l1_timeout_mins": settings.ESCALATION_L1_TIMEOUT_MINS,
        "escalation_l2_timeout_mins": settings.ESCALATION_L2_TIMEOUT_MINS,
        "max_outbound_per_hour": settings.MAX_OUTBOUND_PER_HOUR,
        "business_hours": f"{settings.BUSINESS_HOURS_START}:00 - {settings.BUSINESS_HOURS_END}:00",
        "timezone": settings.TIMEZONE,
        "escalation_keywords": settings.escalation_keywords_list
    }


@router.post("/test-guest")
async def add_test_guest(db: Session = Depends(get_db)):
    """Add a test guest to the cache for testing purposes."""
    from models import GuestIndex
    
    test_guest = GuestIndex(
        guest_phone="+15551234567",
        guest_email="test@example.com",
        guest_name="Test Guest",
        reservation_id="TEST-001",
        listing_id="LISTING-001",
        listing_name="Cozy Downtown Apartment",
        listing_address="123 Main Street, Austin, TX 78701",
        check_in_date=datetime.utcnow(),
        check_out_date=datetime.utcnow() + timedelta(days=3),
        door_code="1234#",
        wifi_name="CozyStay_5G",
        wifi_password="Welcome2024",
        special_instructions="Please remove shoes at the door. Extra towels in the hallway closet.",
        synced_at=datetime.utcnow()
    )
    
    # Check if already exists
    existing = db.query(GuestIndex).filter(
        GuestIndex.reservation_id == "TEST-001"
    ).first()
    
    if existing:
        return {"status": "exists", "guest": test_guest.guest_name, "phone": test_guest.guest_phone}
    
    db.add(test_guest)
    db.commit()
    
    return {"status": "created", "guest": test_guest.guest_name, "phone": test_guest.guest_phone}


# ============ RAG & KNOWLEDGE BASE ENDPOINTS ============

class AddKnowledgeRequest(BaseModel):
    property_id: str
    doc_type: str
    title: str
    content: str


@router.get("/knowledge/stats")
async def get_knowledge_stats():
    """Get statistics about the RAG knowledge system."""
    from knowledge import feedback_learner
    from embeddings import get_collection_stats
    
    try:
        learning_stats = feedback_learner.get_learning_stats()
        return {
            "status": "enabled",
            **learning_stats
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }


@router.get("/knowledge/properties")
async def list_properties_with_knowledge(db: Session = Depends(get_db)):
    """List all properties and their knowledge base status."""
    from models import PropertyKnowledge
    
    # Get unique property IDs
    properties = db.query(GuestIndex.listing_id, GuestIndex.listing_name).distinct().all()
    
    result = []
    for prop_id, prop_name in properties:
        if not prop_id:
            continue
            
        # Count knowledge docs for this property
        doc_count = db.query(PropertyKnowledge).filter(
            PropertyKnowledge.property_id == prop_id,
            PropertyKnowledge.is_active == True
        ).count()
        
        embedded_count = db.query(PropertyKnowledge).filter(
            PropertyKnowledge.property_id == prop_id,
            PropertyKnowledge.is_embedded == True
        ).count()
        
        result.append({
            "property_id": prop_id,
            "property_name": prop_name,
            "knowledge_docs": doc_count,
            "embedded_docs": embedded_count
        })
    
    return result


@router.get("/knowledge/property/{property_id}")
async def get_property_knowledge(property_id: str, db: Session = Depends(get_db)):
    """Get all knowledge documents for a property."""
    from knowledge import knowledge_manager
    
    docs = knowledge_manager.get_property_knowledge(property_id)
    return {
        "property_id": property_id,
        "documents": docs,
        "doc_types_available": knowledge_manager.DOC_TYPES
    }


@router.post("/knowledge/property/{property_id}/init")
async def initialize_property_knowledge(property_id: str, db: Session = Depends(get_db)):
    """Initialize knowledge base folder for a property with templates."""
    from knowledge import knowledge_manager
    
    # Get property name
    guest = db.query(GuestIndex).filter(GuestIndex.listing_id == property_id).first()
    property_name = guest.listing_name if guest else property_id
    
    path = knowledge_manager.create_property_folder(property_id, property_name)
    
    return {
        "status": "created",
        "property_id": property_id,
        "path": str(path),
        "message": f"Template files created. Edit the .md files in {path} and call /index to load them."
    }


@router.post("/knowledge/property/{property_id}/index")
async def index_property_knowledge(property_id: str, db: Session = Depends(get_db)):
    """Load and index all knowledge documents for a property."""
    from knowledge import knowledge_manager
    
    # Get property name
    guest = db.query(GuestIndex).filter(GuestIndex.listing_id == property_id).first()
    property_name = guest.listing_name if guest else ""
    
    count = knowledge_manager.load_and_index_property(property_id, property_name)
    
    return {
        "status": "indexed",
        "property_id": property_id,
        "documents_indexed": count
    }


@router.post("/knowledge/add")
async def add_knowledge_document(request: AddKnowledgeRequest, db: Session = Depends(get_db)):
    """Add a knowledge document to a property's knowledge base."""
    from knowledge import knowledge_manager
    
    if request.doc_type not in knowledge_manager.DOC_TYPES:
        return {
            "status": "error",
            "error": f"Invalid doc_type. Must be one of: {knowledge_manager.DOC_TYPES}"
        }
    
    success = knowledge_manager.add_knowledge(
        property_id=request.property_id,
        doc_type=request.doc_type,
        title=request.title,
        content=request.content
    )
    
    return {
        "status": "added" if success else "error",
        "property_id": request.property_id,
        "doc_type": request.doc_type,
        "title": request.title
    }


@router.post("/knowledge/reindex-all")
async def reindex_all_knowledge(db: Session = Depends(get_db)):
    """Reindex all property knowledge and style examples."""
    from knowledge import knowledge_manager, initialize_knowledge_system
    from embeddings import initialize_style_examples
    
    # Reindex style examples
    style_count = initialize_style_examples()
    
    # Reindex all properties
    properties = db.query(GuestIndex.listing_id, GuestIndex.listing_name).distinct().all()
    
    total_docs = 0
    for prop_id, prop_name in properties:
        if prop_id:
            count = knowledge_manager.load_and_index_property(prop_id, prop_name or "")
            total_docs += count
    
    return {
        "status": "reindexed",
        "style_examples": style_count,
        "property_documents": total_docs,
        "properties_processed": len(properties)
    }


# ============ GUEST PROFILES ============

@router.get("/profiles")
async def list_guest_profiles(
    limit: int = 50,
    vip_only: bool = False,
    db: Session = Depends(get_db)
):
    """List guest profiles."""
    from models import GuestProfile
    
    query = db.query(GuestProfile)
    
    if vip_only:
        query = query.filter(GuestProfile.is_vip == True)
    
    profiles = query.order_by(desc(GuestProfile.last_interaction_at)).limit(limit).all()
    
    return [
        {
            "id": p.id,
            "guest_phone": p.guest_phone,
            "guest_name": p.guest_name,
            "total_stays": p.total_stays,
            "total_conversations": p.total_conversations,
            "communication_style": p.communication_style,
            "overall_sentiment": p.overall_sentiment,
            "is_vip": p.is_vip,
            "last_interaction_at": p.last_interaction_at.isoformat() if p.last_interaction_at else None
        }
        for p in profiles
    ]


@router.get("/profiles/{phone}")
async def get_guest_profile(phone: str, db: Session = Depends(get_db)):
    """Get detailed guest profile."""
    from models import GuestProfile
    import json
    
    normalized = normalize_phone(phone)
    profile = db.query(GuestProfile).filter(GuestProfile.guest_phone == normalized).first()
    
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")
    
    return {
        "id": profile.id,
        "guest_phone": profile.guest_phone,
        "guest_name": profile.guest_name,
        "guest_email": profile.guest_email,
        "total_stays": profile.total_stays,
        "total_conversations": profile.total_conversations,
        "first_interaction_at": profile.first_interaction_at.isoformat() if profile.first_interaction_at else None,
        "last_interaction_at": profile.last_interaction_at.isoformat() if profile.last_interaction_at else None,
        "preferences": json.loads(profile.preferences) if profile.preferences else {},
        "communication_style": profile.communication_style,
        "avg_message_length": profile.avg_message_length,
        "overall_sentiment": profile.overall_sentiment,
        "sentiment_history": json.loads(profile.sentiment_history) if profile.sentiment_history else [],
        "past_issues": json.loads(profile.past_issues) if profile.past_issues else [],
        "is_vip": profile.is_vip,
        "special_handling_notes": profile.special_handling_notes,
        "internal_notes": profile.internal_notes
    }


@router.post("/profiles/{phone}/vip")
async def set_guest_vip(phone: str, is_vip: bool = True, notes: str = None, db: Session = Depends(get_db)):
    """Mark a guest as VIP."""
    from knowledge import guest_profile_manager
    
    normalized = normalize_phone(phone)
    guest_profile_manager.mark_vip(normalized, is_vip, notes)
    
    return {
        "status": "updated",
        "phone": normalized,
        "is_vip": is_vip
    }


# ============ CONVERSATION SUMMARIES ============

@router.get("/conversations/{conversation_id}/summary")
async def get_conversation_summary(conversation_id: int, db: Session = Depends(get_db)):
    """Get AI-generated summary of a conversation."""
    from knowledge import conversation_summarizer
    
    summary = conversation_summarizer.get_latest_summary(conversation_id)
    
    if not summary:
        # Generate one on demand
        generated = conversation_summarizer.generate_summary(conversation_id)
        if generated:
            summary = conversation_summarizer.get_latest_summary(conversation_id)
    
    if not summary:
        return {"status": "no_summary", "reason": "Conversation too short or no messages"}
    
    return {
        "status": "success",
        "conversation_id": conversation_id,
        **summary
    }


# ============ FEEDBACK & LEARNING ============

@router.get("/feedback")
async def list_feedback(
    limit: int = 50,
    edited_only: bool = False,
    db: Session = Depends(get_db)
):
    """List response feedback records."""
    from models import ResponseFeedback
    
    query = db.query(ResponseFeedback)
    
    if edited_only:
        query = query.filter(ResponseFeedback.was_edited == True)
    
    records = query.order_by(desc(ResponseFeedback.created_at)).limit(limit).all()
    
    return [
        {
            "id": r.id,
            "conversation_id": r.conversation_id,
            "original_response": r.original_response[:200] + "..." if len(r.original_response) > 200 else r.original_response,
            "corrected_response": r.corrected_response[:200] + "..." if r.corrected_response and len(r.corrected_response) > 200 else r.corrected_response,
            "correction_type": r.correction_type,
            "was_approved": r.was_approved,
            "was_edited": r.was_edited,
            "is_indexed": r.is_indexed,
            "human_reviewer": r.human_reviewer,
            "created_at": r.created_at.isoformat() if r.created_at else None
        }
        for r in records
    ]


@router.post("/feedback/index-pending")
async def index_pending_feedback(db: Session = Depends(get_db)):
    """Index any unindexed corrections for learning."""
    from models import ResponseFeedback
    from knowledge import feedback_learner
    
    pending = db.query(ResponseFeedback).filter(
        ResponseFeedback.was_edited == True,
        ResponseFeedback.is_indexed == False
    ).all()
    
    indexed = 0
    for feedback in pending:
        try:
            feedback_learner.index_correction(feedback.id)
            indexed += 1
        except Exception as e:
            print(f"[Feedback] Error indexing {feedback.id}: {e}")
    
    return {
        "status": "complete",
        "pending_found": len(pending),
        "indexed": indexed
    }


# ============ RAG SEARCH TEST ============

@router.post("/rag/test-search")
async def test_rag_search(
    query: str,
    property_id: Optional[str] = None
):
    """Test RAG search capabilities."""
    from embeddings import (
        search_property_knowledge,
        search_similar_conversations,
        search_style_examples,
        search_corrections
    )
    
    results = {
        "query": query,
        "property_id": property_id
    }
    
    # Search property knowledge
    if property_id:
        results["property_knowledge"] = search_property_knowledge(query, property_id, top_k=3)
    
    # Search similar conversations
    results["similar_conversations"] = search_similar_conversations(
        query, property_id=property_id, top_k=3
    )
    
    # Search style examples
    results["style_examples"] = search_style_examples(query, top_k=3)
    
    # Search corrections
    results["relevant_corrections"] = search_corrections(query, property_id=property_id, top_k=2)
    
    return results


@router.post("/rag/test-intent")
async def test_intent_classification(message: str, context: str = ""):
    """Test intent classification."""
    from agents import classify_intent
    
    intent = classify_intent(message, context)
    
    return {
        "message": message,
        "primary_intent": intent.primary_intent,
        "secondary_intents": intent.secondary_intents,
        "confidence": intent.confidence,
        "requires_action": intent.requires_action,
        "is_urgent": intent.is_urgent,
        "entities": intent.entities
    }


# ============ HOSTIFY WEBHOOK MANAGEMENT ============

class WebhookRegistration(BaseModel):
    url: str
    notification_type: str = "message_new"
    auth: Optional[str] = None


@router.get("/webhooks/hostify")
async def list_hostify_webhooks():
    """List all registered Hostify webhooks (v2)."""
    import httpx
    
    api_key = settings.HOSTIFY_API_KEY
    if not api_key:
        raise HTTPException(status_code=400, detail="HOSTIFY_API_KEY not configured")
    
    async with httpx.AsyncClient() as client:
        response = await client.get(
            "https://api-rms.hostify.com/webhooks_v2",
            headers={"x-api-key": api_key}
        )
        
        if response.status_code != 200:
            return {"error": f"Hostify API error: {response.status_code}", "body": response.text}
        
        return response.json()


@router.post("/webhooks/hostify/register")
async def register_hostify_webhook(webhook: WebhookRegistration):
    """
    Register a webhook with Hostify (v2 API using Amazon SNS).
    
    After registration, Hostify will send a SubscriptionConfirmation request
    to your URL which must be confirmed.
    
    notification_type options:
    - message_new
    - new_reservation
    - update_reservation
    - move_reservation
    - create_listing
    - update_listing
    """
    import httpx
    
    api_key = settings.HOSTIFY_API_KEY
    if not api_key:
        raise HTTPException(status_code=400, detail="HOSTIFY_API_KEY not configured")
    
    payload = {
        "url": webhook.url,
        "notification_type": webhook.notification_type
    }
    if webhook.auth:
        payload["auth"] = webhook.auth
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://api-rms.hostify.com/webhooks_v2",
            headers={"x-api-key": api_key},
            json=payload
        )
        
        log_event("webhook_registration_attempt", payload={
            "url": webhook.url,
            "type": webhook.notification_type,
            "status": response.status_code
        })
        
        if response.status_code not in [200, 201]:
            return {
                "error": f"Hostify API error: {response.status_code}",
                "body": response.text
            }
        
        return {
            "status": "registered",
            "message": "Webhook registered. Waiting for SNS subscription confirmation.",
            "response": response.json() if response.text else {}
        }


@router.delete("/webhooks/hostify/{webhook_id}")
async def delete_hostify_webhook(webhook_id: int):
    """Delete a registered Hostify webhook."""
    import httpx
    
    api_key = settings.HOSTIFY_API_KEY
    if not api_key:
        raise HTTPException(status_code=400, detail="HOSTIFY_API_KEY not configured")
    
    async with httpx.AsyncClient() as client:
        response = await client.delete(
            f"https://api-rms.hostify.com/webhooks_v2/{webhook_id}",
            headers={"x-api-key": api_key}
        )
        
        if response.status_code not in [200, 204]:
            return {"error": f"Hostify API error: {response.status_code}", "body": response.text}
        
        return {"status": "deleted", "webhook_id": webhook_id}


# ============ PROPERTY KNOWLEDGE BASE ============

from fastapi import UploadFile, File, Form, BackgroundTasks
import shutil
import hashlib
from pathlib import Path


class KnowledgeEntryCreate(BaseModel):
    listing_id: str
    listing_name: str
    knowledge_type: str  # amenity, house_rule, local_recommendation, appliance_guide, common_issue, faq, general
    title: str
    content: str
    question: Optional[str] = None
    answer: Optional[str] = None


class KnowledgeEntryResponse(BaseModel):
    id: int
    listing_id: str
    listing_name: str
    knowledge_type: str
    title: str
    content: str
    question: Optional[str]
    answer: Optional[str]
    source: str
    confidence: float
    times_used: int
    created_at: datetime

    class Config:
        from_attributes = True


@router.get("/knowledge/listings")
async def get_listings_with_knowledge(db: Session = Depends(get_db)):
    """Get all listings that have knowledge entries or uploaded files."""
    from knowledge_base import PropertyKnowledge, UploadedFile
    
    # Get unique listings from knowledge
    knowledge_listings = db.query(
        PropertyKnowledge.listing_id,
        PropertyKnowledge.listing_name,
        func.count(PropertyKnowledge.id).label('knowledge_count')
    ).group_by(
        PropertyKnowledge.listing_id,
        PropertyKnowledge.listing_name
    ).all()
    
    # Get unique listings from uploaded files
    file_listings = db.query(
        UploadedFile.listing_id,
        UploadedFile.listing_name,
        func.count(UploadedFile.id).label('file_count')
    ).group_by(
        UploadedFile.listing_id,
        UploadedFile.listing_name
    ).all()
    
    # Combine results
    listings = {}
    for lid, lname, count in knowledge_listings:
        if lid not in listings:
            listings[lid] = {"listing_id": lid, "listing_name": lname, "knowledge_count": 0, "file_count": 0}
        listings[lid]["knowledge_count"] = count
    
    for lid, lname, count in file_listings:
        if lid not in listings:
            listings[lid] = {"listing_id": lid, "listing_name": lname, "knowledge_count": 0, "file_count": 0}
        listings[lid]["file_count"] = count
    
    # Also include all properties from conversations that don't have knowledge yet
    conv_listings = db.query(
        Conversation.listing_id,
        Conversation.listing_name
    ).filter(
        Conversation.listing_id.isnot(None)
    ).distinct().all()
    
    for lid, lname in conv_listings:
        if lid and lid not in listings:
            listings[lid] = {"listing_id": lid, "listing_name": lname, "knowledge_count": 0, "file_count": 0}
    
    return {"listings": list(listings.values())}


@router.get("/knowledge/{listing_id}")
async def get_property_knowledge(listing_id: str, db: Session = Depends(get_db)):
    """Get all knowledge entries for a specific property."""
    from knowledge_base import PropertyKnowledge, UploadedFile, get_property_knowledge_summary
    
    summary = get_property_knowledge_summary(listing_id)
    
    # Also get uploaded files
    files = db.query(UploadedFile).filter(
        UploadedFile.listing_id == listing_id
    ).order_by(UploadedFile.uploaded_at.desc()).all()
    
    summary["files"] = [
        {
            "id": f.id,
            "filename": f.original_filename,
            "file_type": f.file_type,
            "file_size": f.file_size,
            "processed": f.processed,
            "entries_created": f.knowledge_entries_created,
            "uploaded_at": f.uploaded_at.isoformat() if f.uploaded_at else None
        }
        for f in files
    ]
    
    return summary


@router.post("/knowledge/{listing_id}/upload")
async def upload_property_file(
    listing_id: str,
    listing_name: str = Form(...),
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None,
    db: Session = Depends(get_db)
):
    """
    Upload a property document for knowledge extraction.
    
    Supported formats: PDF, DOCX, TXT, MD, JSON
    """
    from knowledge_base import UploadedFile, UPLOAD_DIR, process_uploaded_file
    
    # Validate file type
    filename = file.filename or "unknown"
    file_ext = filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''
    
    allowed_types = ['pdf', 'docx', 'txt', 'md', 'json', 'doc']
    if file_ext not in allowed_types:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file type: {file_ext}. Allowed: {', '.join(allowed_types)}"
        )
    
    # Read file content
    content = await file.read()
    file_size = len(content)
    
    if file_size > 10 * 1024 * 1024:  # 10MB limit
        raise HTTPException(status_code=400, detail="File too large. Maximum size is 10MB.")
    
    # Calculate hash for deduplication
    file_hash = hashlib.sha256(content).hexdigest()
    
    # Check for duplicate
    existing = db.query(UploadedFile).filter(
        UploadedFile.listing_id == listing_id,
        UploadedFile.file_hash == file_hash
    ).first()
    
    if existing:
        return {
            "status": "duplicate",
            "message": "This file has already been uploaded for this property",
            "file_id": existing.id
        }
    
    # Save file
    stored_filename = f"{listing_id}_{file_hash[:8]}_{filename}"
    file_path = UPLOAD_DIR / stored_filename
    
    with open(file_path, 'wb') as f:
        f.write(content)
    
    # Create database record
    file_record = UploadedFile(
        listing_id=listing_id,
        listing_name=listing_name,
        filename=stored_filename,
        original_filename=filename,
        file_type=file_ext,
        file_size=file_size,
        file_hash=file_hash
    )
    db.add(file_record)
    db.commit()
    db.refresh(file_record)
    
    log_event("file_uploaded", payload={
        "listing_id": listing_id,
        "filename": filename,
        "file_size": file_size
    })
    
    # Process file in background
    if background_tasks:
        background_tasks.add_task(process_uploaded_file, file_record.id)
    
    return {
        "status": "uploaded",
        "file_id": file_record.id,
        "filename": filename,
        "message": "File uploaded. Processing will begin shortly."
    }


@router.post("/knowledge/{listing_id}/process/{file_id}")
async def process_file(listing_id: str, file_id: int, db: Session = Depends(get_db)):
    """Manually trigger processing of an uploaded file."""
    from knowledge_base import UploadedFile, process_uploaded_file
    
    file_record = db.query(UploadedFile).filter(
        UploadedFile.id == file_id,
        UploadedFile.listing_id == listing_id
    ).first()
    
    if not file_record:
        raise HTTPException(status_code=404, detail="File not found")
    
    result = await process_uploaded_file(file_id)
    return result


@router.post("/knowledge/learn")
async def trigger_learning(
    listing_id: Optional[str] = None,
    limit: int = 500,
    background_tasks: BackgroundTasks = None,
    db: Session = Depends(get_db)
):
    """
    Trigger AI learning from past conversations.
    
    This analyzes past guest-host conversations to extract:
    - Common questions and good answers
    - Property-specific information
    - Troubleshooting patterns
    
    Args:
        listing_id: Optional - only learn from this property's conversations
        limit: Maximum conversations to analyze (default 500)
    """
    from knowledge_base import learn_from_messages, LearningSession
    
    # Check if learning is already running
    running = db.query(LearningSession).filter(
        LearningSession.status == "running"
    ).first()
    
    if running:
        return {
            "status": "already_running",
            "message": "A learning session is already in progress",
            "session_id": running.id
        }
    
    # Run learning (this can take a while, so run in background if available)
    if background_tasks:
        background_tasks.add_task(learn_from_messages, listing_id, limit)
        return {
            "status": "started",
            "message": "Learning session started in background"
        }
    else:
        result = await learn_from_messages(listing_id, limit)
        return result


@router.get("/knowledge/learning-sessions")
async def get_learning_sessions(limit: int = 10, db: Session = Depends(get_db)):
    """Get recent learning session history."""
    from knowledge_base import LearningSession
    
    sessions = db.query(LearningSession).order_by(
        LearningSession.started_at.desc()
    ).limit(limit).all()
    
    return {
        "sessions": [
            {
                "id": s.id,
                "listing_id": s.listing_id,
                "status": s.status,
                "conversations_analyzed": s.conversations_analyzed,
                "messages_analyzed": s.messages_analyzed,
                "entries_created": s.knowledge_entries_created,
                "entries_updated": s.knowledge_entries_updated,
                "error": s.error,
                "started_at": s.started_at.isoformat() if s.started_at else None,
                "completed_at": s.completed_at.isoformat() if s.completed_at else None
            }
            for s in sessions
        ]
    }


@router.post("/knowledge/entry")
async def create_knowledge_entry(entry: KnowledgeEntryCreate, db: Session = Depends(get_db)):
    """Manually create a knowledge entry."""
    from knowledge_base import PropertyKnowledge, KnowledgeType
    
    try:
        knowledge_type = KnowledgeType(entry.knowledge_type)
    except ValueError:
        knowledge_type = KnowledgeType.general
    
    knowledge = PropertyKnowledge(
        listing_id=entry.listing_id,
        listing_name=entry.listing_name,
        knowledge_type=knowledge_type,
        title=entry.title,
        content=entry.content,
        question=entry.question,
        answer=entry.answer,
        source="manual",
        confidence=1.0
    )
    db.add(knowledge)
    db.commit()
    db.refresh(knowledge)
    
    return {
        "status": "created",
        "id": knowledge.id
    }


@router.delete("/knowledge/entry/{entry_id}")
async def delete_knowledge_entry(entry_id: int, db: Session = Depends(get_db)):
    """Delete a knowledge entry."""
    from knowledge_base import PropertyKnowledge
    
    entry = db.query(PropertyKnowledge).filter(PropertyKnowledge.id == entry_id).first()
    if not entry:
        raise HTTPException(status_code=404, detail="Entry not found")
    
    db.delete(entry)
    db.commit()
    
    return {"status": "deleted", "id": entry_id}


@router.get("/knowledge/search")
async def search_knowledge_base(
    query: str,
    listing_id: Optional[str] = None,
    knowledge_type: Optional[str] = None,
    limit: int = 10,
    db: Session = Depends(get_db)
):
    """Search the knowledge base."""
    from knowledge_base import search_knowledge, KnowledgeType
    
    types = None
    if knowledge_type:
        try:
            types = [KnowledgeType(knowledge_type)]
        except ValueError:
            pass
    
    results = search_knowledge(
        query=query,
        listing_id=listing_id,
        knowledge_types=types,
        top_k=limit
    )
    
    return {"results": results, "query": query}


# ============ GUEST HEALTH MONITORING ============

class GuestHealthSettingCreate(BaseModel):
    listing_id: str
    listing_name: str
    is_enabled: bool = True


@router.get("/guest-health/settings")
async def get_guest_health_settings(db: Session = Depends(get_db)):
    """Get all guest health monitoring settings."""
    from models import GuestHealthSettings
    
    settings_list = db.query(GuestHealthSettings).all()
    
    return {
        "settings": [
            {
                "id": s.id,
                "listing_id": s.listing_id,
                "listing_name": s.listing_name,
                "is_enabled": s.is_enabled,
                "created_at": s.created_at.isoformat() if s.created_at else None,
                "updated_at": s.updated_at.isoformat() if s.updated_at else None
            }
            for s in settings_list
        ]
    }


@router.get("/guest-health/available-properties")
async def get_available_properties_for_monitoring(db: Session = Depends(get_db)):
    """Get all properties available for guest health monitoring."""
    from models import GuestHealthSettings, Conversation
    
    # Get unique listings from CONVERSATIONS (not GuestIndex)
    # This is important because Hostify uses different listing_ids in 
    # reservations API vs inbox API - we need the inbox listing_ids
    # since that's where the messages are
    listings = db.query(
        Conversation.listing_id,
        Conversation.listing_name
    ).filter(
        Conversation.listing_id.isnot(None),
        Conversation.listing_id != ""
    ).distinct().all()
    
    # Also include listings from conversations
    conv_listings = db.query(
        Conversation.listing_id,
        Conversation.listing_name
    ).filter(
        Conversation.listing_id.isnot(None),
        Conversation.listing_id != ""
    ).distinct().all()
    
    # Combine and dedupe
    all_listings = {}
    for lid, lname in listings + conv_listings:
        if lid and lid not in all_listings:
            all_listings[lid] = lname or lid
    
    # Check which are already monitored
    monitored = db.query(GuestHealthSettings.listing_id).all()
    monitored_ids = {m[0] for m in monitored}
    
    return {
        "properties": [
            {
                "listing_id": lid,
                "listing_name": lname,
                "is_monitored": lid in monitored_ids
            }
            for lid, lname in all_listings.items()
        ]
    }


@router.post("/guest-health/settings")
async def add_guest_health_setting(
    setting: GuestHealthSettingCreate,
    db: Session = Depends(get_db)
):
    """Add a property to guest health monitoring."""
    from models import GuestHealthSettings
    
    # Check if already exists
    existing = db.query(GuestHealthSettings).filter(
        GuestHealthSettings.listing_id == setting.listing_id
    ).first()
    
    if existing:
        # Update existing
        existing.listing_name = setting.listing_name
        existing.is_enabled = setting.is_enabled
        db.commit()
        return {
            "status": "updated",
            "id": existing.id,
            "listing_id": existing.listing_id
        }
    
    # Create new
    new_setting = GuestHealthSettings(
        listing_id=setting.listing_id,
        listing_name=setting.listing_name,
        is_enabled=setting.is_enabled
    )
    db.add(new_setting)
    db.commit()
    db.refresh(new_setting)
    
    return {
        "status": "created",
        "id": new_setting.id,
        "listing_id": new_setting.listing_id
    }


@router.delete("/guest-health/settings/{listing_id}")
async def remove_guest_health_setting(listing_id: str, db: Session = Depends(get_db)):
    """Remove a property from guest health monitoring."""
    from models import GuestHealthSettings
    
    setting = db.query(GuestHealthSettings).filter(
        GuestHealthSettings.listing_id == listing_id
    ).first()
    
    if not setting:
        raise HTTPException(status_code=404, detail="Setting not found")
    
    db.delete(setting)
    db.commit()
    
    return {"status": "deleted", "listing_id": listing_id}


@router.post("/guest-health/settings/bulk")
async def bulk_update_guest_health_settings(
    listing_ids: List[str],
    db: Session = Depends(get_db)
):
    """
    Bulk update guest health monitoring settings.
    Enables monitoring for the provided listing_ids and disables all others.
    """
    from models import GuestHealthSettings
    
    # Get all current settings
    current_settings = db.query(GuestHealthSettings).all()
    current_ids = {s.listing_id for s in current_settings}
    
    # Get listing names for new properties (from Conversations, not GuestIndex)
    # Hostify uses different listing_ids in reservations vs inbox API
    listing_names = {}
    listings = db.query(Conversation.listing_id, Conversation.listing_name).filter(
        Conversation.listing_id.in_(listing_ids)
    ).distinct().all()
    for lid, lname in listings:
        listing_names[lid] = lname
    
    # Add new settings
    added = 0
    for lid in listing_ids:
        if lid not in current_ids:
            new_setting = GuestHealthSettings(
                listing_id=lid,
                listing_name=listing_names.get(lid, lid),
                is_enabled=True
            )
            db.add(new_setting)
            added += 1
    
    # Remove settings not in the list
    removed = 0
    for setting in current_settings:
        if setting.listing_id not in listing_ids:
            db.delete(setting)
            removed += 1
    
    db.commit()
    
    return {
        "status": "updated",
        "added": added,
        "removed": removed,
        "total_monitored": len(listing_ids)
    }


@router.get("/guest-health/guests")
async def get_guest_health_data(db: Session = Depends(get_db)):
    """Get guest health data for all checked-in guests at monitored properties."""
    from guest_health import get_guest_health_summary
    
    guests = get_guest_health_summary(db)
    
    return {
        "guests": guests,
        "total": len(guests),
        "needs_attention": len([g for g in guests if g["needs_attention"]]),
        "at_risk": len([g for g in guests if g["risk_level"] in ["high", "critical"]])
    }


@router.post("/guest-health/refresh")
async def refresh_guest_health(
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Refresh guest health analysis for all monitored properties.
    Fetches latest messages and runs AI analysis.
    """
    from guest_health import refresh_all_guest_health
    
    # Run in background for large portfolios
    result = await refresh_all_guest_health(db)
    
    return result


@router.get("/guest-health/guests/{reservation_id}")
async def get_guest_health_detail(reservation_id: str, db: Session = Depends(get_db)):
    """Get detailed guest health analysis for a specific reservation."""
    from models import GuestHealthAnalysis
    import json
    
    analysis = db.query(GuestHealthAnalysis).filter(
        GuestHealthAnalysis.reservation_id == reservation_id
    ).first()
    
    if not analysis:
        raise HTTPException(status_code=404, detail="Guest analysis not found")
    
    # Get messages for this guest
    from guest_health import get_guest_messages
    messages = get_guest_messages(db, reservation_id, analysis.guest_phone)
    
    return {
        "analysis": {
            "id": analysis.id,
            "reservation_id": analysis.reservation_id,
            "guest_name": analysis.guest_name,
            "guest_phone": analysis.guest_phone,
            "guest_email": analysis.guest_email,
            "listing_id": analysis.listing_id,
            "listing_name": analysis.listing_name,
            "check_in_date": analysis.check_in_date.isoformat() if analysis.check_in_date else None,
            "check_out_date": analysis.check_out_date.isoformat() if analysis.check_out_date else None,
            "nights_stayed": analysis.nights_stayed,
            "nights_remaining": analysis.nights_remaining,
            "reservation_total": analysis.reservation_total,
            "booking_source": analysis.booking_source,
            "sentiment": analysis.sentiment.value if analysis.sentiment else "neutral",
            "sentiment_score": analysis.sentiment_score,
            "sentiment_reasoning": analysis.sentiment_reasoning,
            "complaints": json.loads(analysis.complaints) if analysis.complaints else [],
            "unresolved_issues": json.loads(analysis.unresolved_issues) if analysis.unresolved_issues else [],
            "resolved_issues": json.loads(analysis.resolved_issues) if analysis.resolved_issues else [],
            "total_messages": analysis.total_messages,
            "guest_messages": analysis.guest_messages,
            "avg_response_time_mins": analysis.avg_response_time_mins,
            "last_message_at": analysis.last_message_at.isoformat() if analysis.last_message_at else None,
            "last_message_from": analysis.last_message_from,
            "risk_level": analysis.risk_level,
            "needs_attention": analysis.needs_attention,
            "attention_reason": analysis.attention_reason,
            "recommended_actions": json.loads(analysis.recommended_actions) if analysis.recommended_actions else [],
            "last_analyzed_at": analysis.last_analyzed_at.isoformat() if analysis.last_analyzed_at else None,
            "conversation_id": analysis.conversation_id
        },
        "messages": messages
    }


@router.post("/guest-health/guests/{reservation_id}/refresh")
async def refresh_single_guest_health(reservation_id: str, db: Session = Depends(get_db)):
    """Refresh guest health analysis for a single guest."""
    from guest_health import analyze_and_save_guest
    
    # Get the guest from GuestIndex
    guest = db.query(GuestIndex).filter(
        GuestIndex.reservation_id == reservation_id
    ).first()
    
    if not guest:
        raise HTTPException(status_code=404, detail="Guest not found")
    
    result = await analyze_and_save_guest(db, guest)
    
    if result:
        return {
            "status": "analyzed",
            "reservation_id": reservation_id,
            "sentiment": result.sentiment.value if result.sentiment else "neutral",
            "risk_level": result.risk_level
        }
    else:
        return {
            "status": "error",
            "message": "Failed to analyze guest"
        }


@router.get("/guest-health/stats")
async def get_guest_health_stats(db: Session = Depends(get_db)):
    """Get summary statistics for guest health monitoring."""
    from models import GuestHealthSettings, GuestHealthAnalysis, SentimentLevel
    
    now = datetime.utcnow()
    today = now.replace(hour=0, minute=0, second=0, microsecond=0)
    
    # Count monitored properties
    monitored_properties = db.query(GuestHealthSettings).filter(
        GuestHealthSettings.is_enabled == True
    ).count()
    
    # Get current analyses
    analyses = db.query(GuestHealthAnalysis).filter(
        GuestHealthAnalysis.check_out_date >= today
    ).all()
    
    total_guests = len(analyses)
    
    # Count by sentiment
    sentiment_counts = {
        "very_unhappy": 0,
        "unhappy": 0,
        "neutral": 0,
        "happy": 0,
        "very_happy": 0
    }
    
    for a in analyses:
        sentiment_key = a.sentiment.value if a.sentiment else "neutral"
        sentiment_counts[sentiment_key] = sentiment_counts.get(sentiment_key, 0) + 1
    
    # Count by risk level
    risk_counts = {"low": 0, "medium": 0, "high": 0, "critical": 0}
    for a in analyses:
        risk_counts[a.risk_level] = risk_counts.get(a.risk_level, 0) + 1
    
    # Count needing attention
    needs_attention = len([a for a in analyses if a.needs_attention])
    
    # Count total unresolved issues
    import json
    total_unresolved = 0
    for a in analyses:
        if a.unresolved_issues:
            try:
                issues = json.loads(a.unresolved_issues)
                total_unresolved += len(issues)
            except:
                pass
    
    return {
        "monitored_properties": monitored_properties,
        "total_checked_in_guests": total_guests,
        "needs_attention": needs_attention,
        "sentiment_breakdown": sentiment_counts,
        "risk_breakdown": risk_counts,
        "total_unresolved_issues": total_unresolved,
        "at_risk_count": risk_counts.get("high", 0) + risk_counts.get("critical", 0)
    }


# ============ INQUIRY ANALYSIS ============

@router.get("/inquiries")
async def get_inquiries(
    listing_id: Optional[str] = None,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """Get analyzed inquiries that didn't convert to bookings."""
    from guest_health import get_inquiry_analyses
    
    inquiries = get_inquiry_analyses(db, listing_id=listing_id, limit=limit)
    
    return {
        "inquiries": inquiries,
        "total": len(inquiries)
    }


@router.get("/inquiries/summary")
async def get_inquiry_summary_endpoint(db: Session = Depends(get_db)):
    """Get summary statistics for inquiry analysis."""
    from guest_health import get_inquiry_summary
    
    return get_inquiry_summary(db)


@router.post("/inquiries/refresh")
async def refresh_inquiries(
    days_back: int = 30,
    limit: int = 50,
    db: Session = Depends(get_db)
):
    """
    Analyze recent inquiries that didn't convert to bookings.
    
    Args:
        days_back: How many days of inquiries to analyze
        limit: Max number of inquiries to analyze
    """
    from guest_health import refresh_inquiry_analysis
    
    result = await refresh_inquiry_analysis(db, days_back=days_back, limit=limit)
    
    return result


@router.get("/inquiries/{thread_id}")
async def get_inquiry_detail(thread_id: int, db: Session = Depends(get_db)):
    """Get detailed inquiry analysis for a specific thread."""
    from models import InquiryAnalysis, HostifyMessage
    import json
    
    analysis = db.query(InquiryAnalysis).filter(
        InquiryAnalysis.thread_id == thread_id
    ).first()
    
    if not analysis:
        raise HTTPException(status_code=404, detail="Inquiry analysis not found")
    
    # Get messages for this inquiry
    messages = db.query(HostifyMessage).filter(
        HostifyMessage.inbox_id == thread_id
    ).order_by(HostifyMessage.sent_at).all()
    
    message_list = []
    for m in messages:
        direction = m.direction if m.direction else "unknown"
        if m.sender_type == "guest":
            direction = "inbound"
        elif m.sender_type in ["host", "automatic"]:
            direction = "outbound"
        
        message_list.append({
            "id": m.id,
            "direction": direction,
            "content": m.content,
            "sender_name": m.sender_name,
            "sent_at": m.sent_at.isoformat() if m.sent_at else None
        })
    
    return {
        "analysis": {
            "id": analysis.id,
            "thread_id": analysis.thread_id,
            "guest_name": analysis.guest_name,
            "guest_email": analysis.guest_email,
            "listing_id": analysis.listing_id,
            "listing_name": analysis.listing_name,
            "inquiry_date": analysis.inquiry_date.isoformat() if analysis.inquiry_date else None,
            "requested_checkin": analysis.requested_checkin.isoformat() if analysis.requested_checkin else None,
            "requested_checkout": analysis.requested_checkout.isoformat() if analysis.requested_checkout else None,
            "first_response_minutes": analysis.first_response_minutes,
            "total_messages": analysis.total_messages,
            "team_messages": analysis.team_messages,
            "guest_messages": analysis.guest_messages,
            "conversation_duration_hours": analysis.conversation_duration_hours,
            "outcome": analysis.outcome,
            "outcome_reasoning": analysis.outcome_reasoning,
            "guest_requirements": json.loads(analysis.guest_requirements) if analysis.guest_requirements else [],
            "guest_questions": json.loads(analysis.guest_questions) if analysis.guest_questions else [],
            "questions_answered": analysis.questions_answered,
            "unanswered_questions": json.loads(analysis.unanswered_questions) if analysis.unanswered_questions else [],
            "team_mistakes": json.loads(analysis.team_mistakes) if analysis.team_mistakes else [],
            "team_strengths": json.loads(analysis.team_strengths) if analysis.team_strengths else [],
            "response_quality_score": analysis.response_quality_score,
            "conversion_likelihood": analysis.conversion_likelihood,
            "lost_revenue_estimate": analysis.lost_revenue_estimate,
            "recommendations": json.loads(analysis.recommendations) if analysis.recommendations else [],
            "training_opportunities": json.loads(analysis.training_opportunities) if analysis.training_opportunities else [],
            "analyzed_at": analysis.analyzed_at.isoformat() if analysis.analyzed_at else None
        },
        "messages": message_list
    }
