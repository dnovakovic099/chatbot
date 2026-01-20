"""
Admin API endpoints for the dashboard.
Provides read/write access to conversations, logs, and testing utilities.
"""

from datetime import datetime, timedelta
from typing import Optional, List
from pydantic import BaseModel

from fastapi import APIRouter, Depends, HTTPException
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
    sent_at: datetime
    ai_confidence: Optional[float]
    was_auto_sent: bool
    was_human_edited: bool

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
                sent_at=m.sent_at,
                ai_confidence=m.ai_confidence,
                was_auto_sent=m.was_auto_sent,
                was_human_edited=m.was_human_edited
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
