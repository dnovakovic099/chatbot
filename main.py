"""
AI Property Manager v3.1 - Main FastAPI Application

This is the main entry point for the application. It handles:
- Webhook endpoints for Hostify, OpenPhone, and Slack
- Message processing queue with burst handling
- Startup initialization and scheduled jobs
"""

import json
import hmac
import hashlib
from datetime import datetime, timedelta
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, HTTPException, BackgroundTasks, Depends
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore

from config import settings
from models import (
    init_db, get_db, SessionLocal,
    Conversation, ConversationStatus, Message
)
from utils import (
    normalize_phone, log_event, get_or_create_conversation,
    save_message, message_already_processed, get_conversation,
    get_recent_messages, get_guest_from_cache, check_escalation_keywords,
    check_rate_limit, cache_draft, get_cached_draft
)
from cache import sync_reservations, force_sync_reservations
from brain import generate_ai_response, get_style_examples
from escalation import (
    set_scheduler, trigger_escalation_l1, handle_new_message_while_escalated,
    post_soft_escalation_to_slack, post_to_slack, cancel_escalation_timers,
    handle_approve_draft, handle_snooze, handle_mark_resolved, 
    handle_edit_submit, open_edit_modal, escalation_l1_timeout
)
from dispatch import send_reply
from admin import router as admin_router


# ============ SCHEDULER SETUP ============

jobstores = {
    'default': SQLAlchemyJobStore(url='sqlite:///jobs.db')
}
scheduler = AsyncIOScheduler(jobstores=jobstores)

# Track pending burst timers
pending_burst_jobs = {}  # {conversation_id: job_id}


# ============ APP LIFECYCLE ============

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle manager."""
    # Startup
    print("🚀 Starting AI Property Manager v4.0 (RAG + Agents)...")
    
    # Initialize database
    init_db()
    print("✅ Database initialized")
    
    # Set scheduler reference in escalation module
    set_scheduler(scheduler)
    
    # Start scheduler
    scheduler.start()
    print("✅ Scheduler started")
    
    # Initialize knowledge system (vector DB, embeddings)
    try:
        from knowledge import initialize_knowledge_system
        await initialize_knowledge_system()
        print("✅ Knowledge system initialized (RAG enabled)")
    except Exception as e:
        print(f"⚠️ Knowledge system initialization failed: {e}")
        print("   Falling back to simple mode")
    
    # Schedule recurring jobs
    scheduler.add_job(
        sync_reservations,
        'interval',
        minutes=30,
        id='sync_reservations',
        replace_existing=True
    )
    print("✅ Scheduled reservation sync (every 30 mins)")
    
    # Schedule message sync every 2 minutes (catches messages if webhooks fail)
    from cache import sync_messages
    scheduler.add_job(
        sync_messages,
        'interval',
        minutes=2,
        id='sync_messages',
        replace_existing=True,
        kwargs={'limit_threads': 20}  # Only check recent threads
    )
    print("✅ Scheduled message sync (every 2 mins - webhook backup)")
    
    # Schedule initial sync to run in 5 seconds (non-blocking startup)
    scheduler.add_job(
        sync_reservations,
        'date',
        run_date=datetime.utcnow() + timedelta(seconds=5),
        id='initial_sync',
        replace_existing=True
    )
    print("✅ Initial sync scheduled (in 5 seconds)")
    
    # Recovery: check for orphaned escalations
    await recover_orphaned_escalations()
    print("✅ Orphaned escalation recovery complete")
    
    print("🎉 AI Property Manager v4.0 is ready!")
    print("   ├── RAG: Enabled (ChromaDB + OpenAI Embeddings)")
    print("   ├── Intent Classification: Enabled")
    print("   ├── Guest Profiles: Enabled")
    print("   └── Feedback Learning: Enabled")
    
    yield
    
    # Shutdown
    print("👋 Shutting down...")
    scheduler.shutdown()


app = FastAPI(
    title="AI Property Manager v3.1",
    description="Automated guest communication for short-term rentals",
    version="3.1.0",
    lifespan=lifespan
)

# Include admin API router
app.include_router(admin_router)

# Mount static files for dashboard
app.mount("/static", StaticFiles(directory="static"), name="static")


# ============ WEBHOOK SIGNATURE VERIFICATION ============

def verify_slack_signature(request: Request, body: bytes) -> bool:
    """Verify Slack request signature."""
    if settings.SKIP_WEBHOOK_VERIFICATION:
        return True
    
    timestamp = request.headers.get("X-Slack-Request-Timestamp", "")
    signature = request.headers.get("X-Slack-Signature", "")
    
    if not timestamp or not signature:
        return False
    
    sig_basestring = f"v0:{timestamp}:{body.decode()}"
    my_signature = "v0=" + hmac.new(
        settings.SLACK_SIGNING_SECRET.encode(),
        sig_basestring.encode(),
        hashlib.sha256
    ).hexdigest()
    
    return hmac.compare_digest(my_signature, signature)


def verify_openphone_signature(request: Request, body: bytes) -> bool:
    """Verify OpenPhone request signature."""
    if settings.SKIP_WEBHOOK_VERIFICATION:
        return True
    
    # OpenPhone signature verification
    signature = request.headers.get("X-OpenPhone-Signature", "")
    if not signature or not settings.OPENPHONE_WEBHOOK_SECRET:
        return True  # Skip if not configured
    
    expected = hmac.new(
        settings.OPENPHONE_WEBHOOK_SECRET.encode(),
        body,
        hashlib.sha256
    ).hexdigest()
    
    return hmac.compare_digest(expected, signature)


# ============ WEBHOOKS ============

@app.post("/webhook/hostify")
async def webhook_hostify(
    request: Request, 
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Handle incoming webhooks from Hostify."""
    body = await request.body()
    payload = await request.json()
    
    log_event("webhook_received", payload={"source": "hostify", "data": payload})
    
    # Extract message details (adjust based on actual Hostify webhook format)
    msg_id = payload.get("message_id") or payload.get("id")
    guest_phone = normalize_phone(
        payload.get("guest_phone") or 
        payload.get("guest", {}).get("phone", "")
    )
    content = payload.get("content") or payload.get("message") or payload.get("body", "")
    direction = payload.get("direction") or payload.get("type", "")
    
    # Ignore outbound messages (prevent self-reply loop)
    if direction in ["outbound", "sent"]:
        return {"status": "ignored", "reason": "outbound message"}
    
    if not guest_phone or not content:
        return {"status": "ignored", "reason": "missing phone or content"}
    
    # Deduplication
    if message_already_processed(msg_id, db):
        return {"status": "ignored", "reason": "duplicate"}
    
    # Queue for processing
    background_tasks.add_task(
        queue_message_for_processing, 
        guest_phone, content, "hostify", msg_id
    )
    
    return {"status": "queued"}


@app.post("/webhook/hostify-sns")
async def webhook_hostify_sns(
    request: Request, 
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Handle Amazon SNS webhooks from Hostify (message_new notifications).
    
    This handles:
    1. SubscriptionConfirmation - confirms the SNS subscription
    2. Notification - processes new message events
    """
    import httpx
    
    body = await request.body()
    
    # Log raw request for debugging
    print(f"[Webhook] Raw body: {body.decode()[:500]}")
    
    # SNS sends JSON but with different content types
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        log_event("sns_webhook_error", payload={"error": "Invalid JSON", "body": body.decode()[:500]})
        return {"status": "error", "reason": "Invalid JSON"}
    
    sns_type = payload.get("Type")
    action = payload.get("action")  # Direct Hostify format uses "action"
    
    # Log for debugging
    log_event("webhook_received", payload={
        "sns_type": sns_type, 
        "action": action,
        "is_incoming": payload.get("is_incoming"),
        "thread_id": payload.get("thread_id"),
        "message_preview": str(payload.get("message", ""))[:100]
    })
    print(f"[Webhook] SNS Type: {sns_type}, Action: {action}, is_incoming: {payload.get('is_incoming')}")
    
    # ==== Handle SNS Subscription Confirmation ====
    if sns_type == "SubscriptionConfirmation":
        subscribe_url = payload.get("SubscribeURL")
        if subscribe_url:
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.get(subscribe_url)
                    if response.status_code == 200:
                        log_event("sns_subscription_confirmed", payload={"topic": payload.get("TopicArn")})
                        return {"status": "confirmed"}
                    else:
                        log_event("sns_subscription_failed", payload={"status": response.status_code})
                        return {"status": "error", "reason": f"Confirmation failed: {response.status_code}"}
            except Exception as e:
                log_event("sns_subscription_error", payload={"error": str(e)})
                return {"status": "error", "reason": str(e)}
        return {"status": "error", "reason": "No SubscribeURL"}
    
    # ==== Handle SNS Notification (wrapped format) ====
    if sns_type == "Notification":
        try:
            message_data = json.loads(payload.get("Message", "{}"))
        except json.JSONDecodeError:
            message_data = {"raw": payload.get("Message", "")}
        
        log_event("sns_message_received", payload=message_data)
        inbox_id = message_data.get("inbox_id") or message_data.get("thread_id")
        is_incoming = message_data.get("is_incoming", 0)
        
        if not is_incoming:
            return {"status": "ignored", "reason": "outbound message"}
        
        if inbox_id:
            from cache import sync_single_inbox_thread
            background_tasks.add_task(sync_single_inbox_thread, str(inbox_id))
            return {"status": "queued", "inbox_id": inbox_id}
        
        return {"status": "received", "data": message_data}
    
    # ==== Handle Direct Hostify Format (action: "message_new") ====
    if action == "message_new":
        thread_id = payload.get("thread_id")
        is_incoming = payload.get("is_incoming", 0)
        message_content = payload.get("message") or ""  # Handle null message
        
        log_event("hostify_message_new", payload={
            "thread_id": thread_id,
            "is_incoming": is_incoming,
            "message_preview": message_content[:100] if message_content else "(empty)"
        })
        
        # Only process inbound messages (from guest)
        # is_incoming: 1 = guest message, 0 = host message
        if not is_incoming:
            print(f"[Webhook] Ignoring outbound message (is_incoming=0)")
            return {"status": "ignored", "reason": "outbound message (host sent)"}
        
        if thread_id:
            print(f"[Webhook] ✅ New guest message! Syncing thread {thread_id}")
            from cache import sync_single_inbox_thread
            background_tasks.add_task(sync_single_inbox_thread, str(thread_id))
            return {"status": "queued", "thread_id": thread_id, "message": "Syncing and generating AI suggestion"}
        
        return {"status": "error", "reason": "No thread_id in payload"}
    
    # Handle unsubscribe confirmation
    if sns_type == "UnsubscribeConfirmation":
        log_event("sns_unsubscribed", payload={"topic": payload.get("TopicArn")})
        return {"status": "unsubscribed"}
    
    return {"status": "unknown_format", "sns_type": sns_type, "action": action}


@app.post("/webhook/openphone")
async def webhook_openphone(
    request: Request, 
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Handle incoming webhooks from OpenPhone."""
    body = await request.body()
    
    if not verify_openphone_signature(request, body):
        log_event("webhook_verification_failed", payload={"source": "openphone"})
        raise HTTPException(status_code=401, detail="Invalid signature")
    
    payload = await request.json()
    log_event("webhook_received", payload={"source": "openphone", "data": payload})
    
    # OpenPhone event types: incomingMessageReceived, outgoingMessageSent, etc.
    event_type = payload.get("type")
    
    if event_type != "incomingMessageReceived":
        return {"status": "ignored", "reason": f"event type: {event_type}"}
    
    data = payload.get("data", {})
    msg_id = data.get("id")
    guest_phone = normalize_phone(data.get("from", ""))
    content = data.get("body", "")
    
    if not guest_phone or not content:
        return {"status": "ignored", "reason": "missing phone or content"}
    
    # Deduplication
    if message_already_processed(msg_id, db):
        return {"status": "ignored", "reason": "duplicate"}
    
    background_tasks.add_task(
        queue_message_for_processing, 
        guest_phone, content, "openphone", msg_id
    )
    
    return {"status": "queued"}


@app.post("/webhook/slack/interactivity")
async def webhook_slack_interactivity(request: Request):
    """Handle Slack interactive components (buttons, modals)."""
    body = await request.body()
    
    if not verify_slack_signature(request, body):
        log_event("webhook_verification_failed", payload={"source": "slack"})
        raise HTTPException(status_code=401, detail="Invalid signature")
    
    # Slack sends form-encoded payload
    form_data = await request.form()
    payload = json.loads(form_data.get("payload", "{}"))
    
    payload_type = payload.get("type")
    
    if payload_type == "block_actions":
        # Button clicks
        action = payload.get("actions", [{}])[0]
        action_id = action.get("action_id")
        conversation_id = int(action.get("value", 0))
        user_id = payload.get("user", {}).get("id")
        
        db = SessionLocal()
        try:
            if action_id == "approve_draft":
                await handle_approve_draft(conversation_id, user_id, db)
            elif action_id == "edit_and_send":
                trigger_id = payload.get("trigger_id")
                open_edit_modal(trigger_id, conversation_id)
            elif action_id == "snooze":
                await handle_snooze(conversation_id, user_id, db)
            elif action_id == "mark_resolved":
                await handle_mark_resolved(conversation_id, user_id, db)
        finally:
            db.close()
        
        return JSONResponse(content={"ok": True})
    
    elif payload_type == "view_submission":
        # Modal submission
        callback_id = payload.get("view", {}).get("callback_id", "")
        if callback_id.startswith("edit_modal_"):
            conversation_id = int(payload.get("view", {}).get("private_metadata", 0))
            user_id = payload.get("user", {}).get("id")
            
            # Extract edited text
            values = payload.get("view", {}).get("state", {}).get("values", {})
            reply_block = values.get("reply_block", {})
            edited_text = reply_block.get("reply_input", {}).get("value", "")
            
            if edited_text and conversation_id:
                db = SessionLocal()
                try:
                    await handle_edit_submit(conversation_id, user_id, edited_text, db)
                finally:
                    db.close()
        
        return JSONResponse(content={"response_action": "clear"})
    
    return JSONResponse(content={"ok": True})


# ============ BURST HANDLER & PROCESSING QUEUE ============

async def queue_message_for_processing(
    guest_phone: str, 
    content: str, 
    source: str, 
    msg_id: str
):
    """
    Implements the "burst handler" - waits 15 seconds for additional messages
    before processing, so multi-text guests get one combined response.
    """
    db = SessionLocal()
    try:
        # Get or create conversation
        conversation = get_or_create_conversation(guest_phone, db)
        
        # Save the message
        save_message(
            conversation_id=conversation.id, 
            content=content, 
            direction="inbound", 
            source=source, 
            external_id=msg_id,
            db=db
        )
        
        # Update conversation
        conversation.last_message_at = datetime.utcnow()
        db.commit()
        
        # Cancel existing burst timer if any
        if conversation.id in pending_burst_jobs:
            try:
                scheduler.remove_job(pending_burst_jobs[conversation.id])
            except Exception:
                pass
        
        # Schedule processing in 15 seconds (resets with each new message)
        job = scheduler.add_job(
            process_conversation,
            'date',
            run_date=datetime.utcnow() + timedelta(seconds=settings.MESSAGE_BURST_DELAY_SECONDS),
            args=[conversation.id],
            id=f"burst_{conversation.id}_{datetime.utcnow().timestamp()}"
        )
        pending_burst_jobs[conversation.id] = job.id
        
        log_event("message_received", conversation_id=conversation.id, payload={
            "phone": guest_phone,
            "content": content,
            "source": source
        })
        
    finally:
        db.close()


async def process_conversation(conversation_id: int):
    """Main AI processing logic."""
    db = SessionLocal()
    try:
        conversation = get_conversation(conversation_id, db)
        if not conversation:
            return
        
        # Remove from pending bursts
        pending_burst_jobs.pop(conversation_id, None)
        
        # If conversation is already escalated, handle differently
        if conversation.status in [ConversationStatus.escalated_l1, ConversationStatus.escalated_l2]:
            await handle_new_message_while_escalated(conversation, db)
            return
        
        # ============ CONTEXT ASSEMBLY ============
        
        # 1. Get recent messages (last 24h)
        recent_messages = get_recent_messages(conversation_id, db, hours=24)
        
        # 2. Get guest context from cache
        guest_context = get_guest_from_cache(conversation.guest_phone, db)
        
        if not guest_context:
            # Force sync and retry
            log_event("cache_miss", conversation_id=conversation_id)
            await force_sync_reservations()
            guest_context = get_guest_from_cache(conversation.guest_phone, db)
        
        if not guest_context:
            # Unknown guest flow
            await handle_unknown_guest(conversation, db)
            return
        
        # Update conversation with guest info
        conversation.guest_name = guest_context.guest_name
        conversation.hostify_reservation_id = guest_context.reservation_id
        conversation.listing_name = guest_context.listing_name
        db.commit()
        
        # 3. Get style examples (few-shot)
        last_guest_message = recent_messages[-1].content if recent_messages else ""
        style_examples = get_style_examples(last_guest_message, n=3)
        
        # ============ KEYWORD CHECK ============
        
        matched_keyword = check_escalation_keywords(last_guest_message)
        if matched_keyword:
            log_event("escalated_l1", conversation_id=conversation_id, payload={
                "reason": "keyword_trigger",
                "keyword": matched_keyword
            })
            await trigger_escalation_l1(
                conversation, 
                reason=f"Keyword detected: '{matched_keyword}'", 
                draft="",
                db=db
            )
            return
        
        # ============ AI GENERATION ============
        
        ai_response = await generate_ai_response(
            messages=recent_messages,
            guest_context=guest_context,
            style_examples=style_examples,
            conversation_id=conversation_id,
            use_advanced=True  # Use full RAG + agents pipeline
        )
        
        log_event("ai_response_generated", conversation_id=conversation_id, payload={
            "confidence": ai_response.confidence_score,
            "reasoning": ai_response.reasoning,
            "requires_human": ai_response.requires_human,
            "intent": getattr(ai_response, 'intent', None),
            "handler_used": getattr(ai_response, 'handler_used', None),
            "context_sources": getattr(ai_response, 'context_sources', [])
        })
        
        # ============ DECISION ENGINE ============
        
        # Check rate limit
        if not check_rate_limit(conversation, db):
            log_event("rate_limit_hit", conversation_id=conversation_id)
            await trigger_escalation_l1(
                conversation, 
                reason="Rate limit exceeded - possible bug", 
                draft=ai_response.reply_text,
                db=db
            )
            return
        
        if ai_response.requires_human:
            await trigger_escalation_l1(
                conversation, 
                reason=ai_response.escalation_reason or "AI requested human review", 
                draft=ai_response.reply_text,
                db=db
            )
        
        elif ai_response.confidence_score >= settings.CONFIDENCE_AUTO_SEND:
            # High confidence - auto send
            await send_reply(
                conversation, 
                ai_response.reply_text, 
                auto_sent=True, 
                confidence=ai_response.confidence_score,
                db=db
            )
            log_event("auto_sent", conversation_id=conversation_id)
        
        elif ai_response.confidence_score >= settings.CONFIDENCE_SOFT_ESCALATE:
            # Medium confidence - send but flag for review
            await send_reply(
                conversation, 
                ai_response.reply_text, 
                auto_sent=True, 
                confidence=ai_response.confidence_score,
                db=db
            )
            await post_soft_escalation_to_slack(conversation, ai_response, db)
            log_event("auto_sent", conversation_id=conversation_id, payload={"soft_escalated": True})
        
        else:
            # Low confidence - escalate
            await trigger_escalation_l1(
                conversation, 
                reason=ai_response.reasoning, 
                draft=ai_response.reply_text,
                db=db
            )
            
    finally:
        db.close()


async def handle_unknown_guest(conversation: Conversation, db: Session):
    """
    Handle messages from unknown guests.
    Uses AI to generate a friendly response asking for their reservation details.
    """
    log_event("unknown_guest", conversation_id=conversation.id, payload={
        "phone": conversation.guest_phone
    })
    
    # Get recent messages for context
    recent_messages = get_recent_messages(conversation.id, db, hours=24)
    
    # Get list of property names from cache (for context)
    from sqlalchemy import distinct
    property_names = [
        r[0] for r in db.query(distinct(GuestIndex.listing_name))
        .filter(GuestIndex.listing_name.isnot(None))
        .filter(GuestIndex.listing_name != "")
        .limit(10)
        .all()
    ]
    
    # Generate AI response asking for identification
    from brain import generate_unknown_guest_response
    ai_response = await generate_unknown_guest_response(
        messages=recent_messages,
        available_properties=property_names
    )
    
    await send_reply(
        conversation, 
        ai_response.reply_text, 
        auto_sent=True, 
        confidence=ai_response.confidence_score,
        db=db
    )
    
    # Also post to Slack for awareness
    from utils import get_last_message
    last_msg = get_last_message(conversation.id, db) or "(no message)"
    
    await post_to_slack(
        channel=settings.SLACK_CHANNEL_L1,
        text=f"🔍 *Unknown Guest*\n*Phone:* {conversation.guest_phone}\n*Message:* {last_msg}\n\n_Sent auto-response asking for reservation details._",
        conversation_id=conversation.id
    )


async def recover_orphaned_escalations():
    """On startup, re-create timers for any stuck escalations."""
    db = SessionLocal()
    try:
        stuck_conversations = db.query(Conversation).filter(
            Conversation.status.in_([
                ConversationStatus.escalated_l1, 
                ConversationStatus.escalated_l2
            ])
        ).all()
        
        for conv in stuck_conversations:
            # Check if timer already exists
            job_id = f"escalation_l1_{conv.id}" if conv.status == ConversationStatus.escalated_l1 else f"escalation_l2_{conv.id}"
            
            if not scheduler.get_job(job_id):
                # Re-create timer (start fresh)
                timeout_mins = (
                    settings.ESCALATION_L1_TIMEOUT_MINS 
                    if conv.status == ConversationStatus.escalated_l1 
                    else settings.ESCALATION_L2_TIMEOUT_MINS
                )
                
                from escalation import escalation_l2_timeout
                handler = (
                    escalation_l1_timeout 
                    if conv.status == ConversationStatus.escalated_l1 
                    else escalation_l2_timeout
                )
                
                scheduler.add_job(
                    handler,
                    'date',
                    run_date=datetime.utcnow() + timedelta(minutes=timeout_mins),
                    args=[conv.id],
                    id=job_id
                )
                
                log_event("escalation_timer_recovered", conversation_id=conv.id)
                print(f"  ↳ Recovered escalation timer for conversation {conv.id}")
                
    finally:
        db.close()


# ============ HEALTH CHECK ============

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "version": "3.1.0",
        "dry_run_mode": settings.DRY_RUN_MODE
    }


@app.get("/")
async def root():
    """Serve the dashboard."""
    return FileResponse("static/index.html")


@app.get("/api")
async def api_info():
    """API info endpoint."""
    return {
        "name": "AI Property Manager",
        "version": "3.1.0",
        "docs": "/docs"
    }


# ============ RUN SERVER ============

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
