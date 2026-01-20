"""
Slack escalation engine.
Handles L1/L2 escalations, timeouts, and human action handlers.
"""

from datetime import datetime, timedelta
from typing import Optional

from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from sqlalchemy.orm import Session

from config import settings
from models import Conversation, ConversationStatus, SessionLocal
from utils import (
    log_event, is_business_hours, cache_draft, get_cached_draft,
    get_last_inbound_message, clear_cached_draft
)


# Initialize Slack client
slack_client = WebClient(token=settings.SLACK_BOT_TOKEN)

# Reference to scheduler (set from main.py)
_scheduler = None


def set_scheduler(scheduler):
    """Set the scheduler reference from main.py."""
    global _scheduler
    _scheduler = scheduler


def get_scheduler():
    """Get the scheduler instance."""
    return _scheduler


async def trigger_escalation_l1(
    conversation: Conversation, 
    reason: str, 
    draft: str,
    db: Session
):
    """
    Trigger Level 1 escalation to Slack.
    Posts to the L1 channel with action buttons for human review.
    """
    conversation.status = ConversationStatus.escalated_l1
    db.commit()
    
    # Determine timeout based on business hours
    timeout_mins = settings.ESCALATION_L1_TIMEOUT_MINS
    if not is_business_hours():
        timeout_mins = settings.ESCALATION_L1_TIMEOUT_AFTER_HOURS_MINS
    
    # Get last guest message
    last_message = get_last_inbound_message(conversation.id, db) or "(no message)"
    
    # Build Slack blocks
    blocks = [
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"⚠️ *Escalation Required*\n*Guest:* {conversation.guest_name or 'Unknown'}\n*Property:* {conversation.listing_name or 'Unknown'}\n*Phone:* {conversation.guest_phone}"
            }
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*Reason:* {reason}"
            }
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*Guest said:*\n> {last_message}"
            }
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*AI Draft:*\n```{draft}```" if draft else "*AI Draft:* _(none generated)_"
            }
        },
        {
            "type": "actions",
            "elements": [
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "✅ Approve Draft"},
                    "style": "primary",
                    "action_id": "approve_draft",
                    "value": str(conversation.id)
                },
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "✏️ Edit & Send"},
                    "action_id": "edit_and_send",
                    "value": str(conversation.id)
                },
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "💤 Snooze"},
                    "action_id": "snooze",
                    "value": str(conversation.id)
                },
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "✔️ Mark Resolved"},
                    "style": "danger",
                    "action_id": "mark_resolved",
                    "value": str(conversation.id),
                    "confirm": {
                        "title": {"type": "plain_text", "text": "Confirm Resolution"},
                        "text": {"type": "plain_text", "text": "Are you sure this is resolved? The guest will not receive a reply."},
                        "confirm": {"type": "plain_text", "text": "Yes, Resolved"},
                        "deny": {"type": "plain_text", "text": "Cancel"}
                    }
                }
            ]
        }
    ]
    
    try:
        response = slack_client.chat_postMessage(
            channel=settings.SLACK_CHANNEL_L1,
            text=f"⚠️ Escalation for {conversation.guest_name or conversation.guest_phone}",
            blocks=blocks
        )
        
        # Save thread info
        conversation.slack_thread_ts = response["ts"]
        conversation.slack_channel_id = response["channel"]
        db.commit()
        
        # Cache the draft for "Approve" button
        if draft:
            cache_draft(conversation.id, draft)
        
        # Schedule L1 timeout
        scheduler = get_scheduler()
        if scheduler:
            scheduler.add_job(
                escalation_l1_timeout,
                'date',
                run_date=datetime.utcnow() + timedelta(minutes=timeout_mins),
                args=[conversation.id],
                id=f"escalation_l1_{conversation.id}",
                replace_existing=True
            )
        
        log_event("escalated_l1", conversation_id=conversation.id, payload={"reason": reason})
        
    except SlackApiError as e:
        log_event("api_error", conversation_id=conversation.id, payload={
            "service": "slack",
            "error": str(e)
        })


async def handle_new_message_while_escalated(conversation: Conversation, db: Session):
    """Handle new messages that arrive while already escalated."""
    
    if not conversation.slack_thread_ts:
        # No thread to reply to, trigger fresh escalation
        await trigger_escalation_l1(conversation, reason="New message (no existing thread)", draft="", db=db)
        return
    
    # Post update to existing Slack thread
    new_message = get_last_inbound_message(conversation.id, db)
    
    try:
        slack_client.chat_postMessage(
            channel=conversation.slack_channel_id,
            thread_ts=conversation.slack_thread_ts,
            text=f"📨 *Guest sent another message:*\n> {new_message}"
        )
        
        # Reset escalation timer
        scheduler = get_scheduler()
        if scheduler:
            try:
                scheduler.remove_job(f"escalation_l1_{conversation.id}")
            except Exception:
                pass
            
            timeout_mins = settings.ESCALATION_L1_TIMEOUT_MINS
            if not is_business_hours():
                timeout_mins = settings.ESCALATION_L1_TIMEOUT_AFTER_HOURS_MINS
            
            scheduler.add_job(
                escalation_l1_timeout,
                'date',
                run_date=datetime.utcnow() + timedelta(minutes=timeout_mins),
                args=[conversation.id],
                id=f"escalation_l1_{conversation.id}",
                replace_existing=True
            )
        
    except SlackApiError as e:
        log_event("api_error", payload={"service": "slack", "error": str(e)})


async def escalation_l1_timeout(conversation_id: int):
    """Called when L1 escalation times out without human action."""
    db = SessionLocal()
    try:
        conversation = db.query(Conversation).filter(
            Conversation.id == conversation_id
        ).first()
        
        if not conversation or conversation.status != ConversationStatus.escalated_l1:
            return  # Already resolved
        
        log_event("escalation_timeout", conversation_id=conversation_id, payload={"level": "L1"})
        
        # Escalate to L2
        conversation.status = ConversationStatus.escalated_l2
        db.commit()
        
        # Post urgent message to L2 channel
        try:
            slack_client.chat_postMessage(
                channel=settings.SLACK_CHANNEL_L2,
                text=f"🚨 *URGENT: Guest waiting {settings.ESCALATION_L1_TIMEOUT_MINS}+ mins*\n"
                     f"*Guest:* {conversation.guest_name or conversation.guest_phone}\n"
                     f"*Property:* {conversation.listing_name or 'Unknown'}\n"
                     f"<!channel>"
            )
            
            # Also reply in original thread
            if conversation.slack_thread_ts:
                slack_client.chat_postMessage(
                    channel=conversation.slack_channel_id,
                    thread_ts=conversation.slack_thread_ts,
                    text=f"⏰ *No response after {settings.ESCALATION_L1_TIMEOUT_MINS} minutes. Escalated to L2.*"
                )
        except SlackApiError as e:
            log_event("api_error", payload={"service": "slack", "error": str(e)})
        
        # Schedule L2 timeout
        scheduler = get_scheduler()
        if scheduler:
            scheduler.add_job(
                escalation_l2_timeout,
                'date',
                run_date=datetime.utcnow() + timedelta(minutes=settings.ESCALATION_L2_TIMEOUT_MINS),
                args=[conversation_id],
                id=f"escalation_l2_{conversation_id}",
                replace_existing=True
            )
        
        log_event("escalated_l2", conversation_id=conversation_id)
        
    finally:
        db.close()


async def escalation_l2_timeout(conversation_id: int):
    """Called when L2 escalation times out - critical alert."""
    db = SessionLocal()
    try:
        conversation = db.query(Conversation).filter(
            Conversation.id == conversation_id
        ).first()
        
        if not conversation or conversation.status != ConversationStatus.escalated_l2:
            return
        
        log_event("escalation_timeout", conversation_id=conversation_id, payload={"level": "L2"})
        
        # Post critical alert
        try:
            total_wait = settings.ESCALATION_L1_TIMEOUT_MINS + settings.ESCALATION_L2_TIMEOUT_MINS
            slack_client.chat_postMessage(
                channel=settings.SLACK_CHANNEL_ERRORS,
                text=f"🔴 *CRITICAL: Guest waiting {total_wait}+ mins with NO response*\n"
                     f"*Guest:* {conversation.guest_name or conversation.guest_phone}\n"
                     f"*Property:* {conversation.listing_name or 'Unknown'}\n"
                     f"*Phone:* {conversation.guest_phone}\n"
                     f"<!channel> - IMMEDIATE ACTION REQUIRED"
            )
        except SlackApiError as e:
            log_event("api_error", payload={"service": "slack", "error": str(e)})
            
    finally:
        db.close()


async def post_soft_escalation_to_slack(conversation: Conversation, ai_response, db: Session):
    """Post a soft escalation notification for review (message already sent)."""
    try:
        slack_client.chat_postMessage(
            channel=settings.SLACK_CHANNEL_L1,
            text=f"👀 *Auto-sent (Review Recommended)*\n"
                 f"*Guest:* {conversation.guest_name or 'Unknown'}\n"
                 f"*Property:* {conversation.listing_name or 'Unknown'}\n"
                 f"*Confidence:* {ai_response.confidence_score:.0%}\n"
                 f"*Response sent:* {ai_response.reply_text}\n"
                 f"*Reasoning:* {ai_response.reasoning}"
        )
    except SlackApiError as e:
        log_event("api_error", payload={"service": "slack", "error": str(e)})


async def post_to_slack(channel: str, text: str, conversation_id: Optional[int] = None):
    """Generic helper to post a message to Slack."""
    try:
        slack_client.chat_postMessage(channel=channel, text=text)
    except SlackApiError as e:
        log_event("api_error", conversation_id=conversation_id, payload={
            "service": "slack",
            "error": str(e)
        })


def cancel_escalation_timers(conversation_id: int):
    """Cancel all escalation timers for a conversation."""
    scheduler = get_scheduler()
    if not scheduler:
        return
        
    for job_id in [f"escalation_l1_{conversation_id}", f"escalation_l2_{conversation_id}"]:
        try:
            scheduler.remove_job(job_id)
        except Exception:
            pass


# ============ Slack Action Handlers ============

async def handle_approve_draft(conversation_id: int, user_id: str, db: Session):
    """Handle 'Approve Draft' button click."""
    from dispatch import send_reply
    from knowledge import feedback_learner
    from utils import get_last_inbound_message
    
    conversation = db.query(Conversation).filter(
        Conversation.id == conversation_id
    ).first()
    
    if not conversation:
        return
    
    draft = get_cached_draft(conversation_id)
    
    if not draft:
        # No draft cached, post error
        try:
            slack_client.chat_postMessage(
                channel=conversation.slack_channel_id,
                thread_ts=conversation.slack_thread_ts,
                text="❌ Error: No draft found. Please use 'Edit & Send' instead."
            )
        except SlackApiError:
            pass
        return
    
    await send_reply(
        conversation, 
        draft, 
        auto_sent=False, 
        confidence=None, 
        human_approved_by=user_id,
        db=db
    )
    
    conversation.status = ConversationStatus.resolved
    conversation.last_human_action_at = datetime.utcnow()
    db.commit()
    
    # Record feedback for learning
    try:
        feedback_id = feedback_learner.record_feedback(
            conversation_id=conversation_id,
            message_id=None,
            original_response=draft,
            original_confidence=0.0,  # Was escalated, so low confidence
            was_approved=True,
            was_edited=False,
            human_reviewer=user_id
        )
        
        # Index successful conversation for future retrieval
        guest_message = get_last_inbound_message(conversation_id, db)
        if guest_message:
            feedback_learner.index_successful_conversation(
                conversation_id=conversation_id,
                guest_message=guest_message,
                ai_response=draft,
                was_edited=False
            )
    except Exception as e:
        print(f"[Feedback] Error recording approval: {e}")
    
    # Cancel escalation timers
    cancel_escalation_timers(conversation_id)
    clear_cached_draft(conversation_id)
    
    # Update Slack thread
    try:
        slack_client.chat_postMessage(
            channel=conversation.slack_channel_id,
            thread_ts=conversation.slack_thread_ts,
            text=f"✅ Draft approved and sent by <@{user_id}>"
        )
    except SlackApiError:
        pass
    
    log_event("human_approved", conversation_id=conversation_id, payload={"user": user_id})


async def handle_snooze(conversation_id: int, user_id: str, db: Session):
    """Handle 'Snooze' button - pauses escalation but doesn't resolve."""
    conversation = db.query(Conversation).filter(
        Conversation.id == conversation_id
    ).first()
    
    if not conversation:
        return
    
    conversation.status = ConversationStatus.snoozed
    conversation.last_human_action_at = datetime.utcnow()
    db.commit()
    
    cancel_escalation_timers(conversation_id)
    
    try:
        slack_client.chat_postMessage(
            channel=conversation.slack_channel_id,
            thread_ts=conversation.slack_thread_ts,
            text=f"💤 Snoozed by <@{user_id}>. Will re-escalate if guest sends another message."
        )
    except SlackApiError:
        pass
    
    log_event("human_snoozed", conversation_id=conversation_id, payload={"user": user_id})


async def handle_mark_resolved(conversation_id: int, user_id: str, db: Session):
    """Handle 'Mark Resolved' button."""
    conversation = db.query(Conversation).filter(
        Conversation.id == conversation_id
    ).first()
    
    if not conversation:
        return
    
    conversation.status = ConversationStatus.resolved
    conversation.last_human_action_at = datetime.utcnow()
    db.commit()
    
    cancel_escalation_timers(conversation_id)
    clear_cached_draft(conversation_id)
    
    try:
        slack_client.chat_postMessage(
            channel=conversation.slack_channel_id,
            thread_ts=conversation.slack_thread_ts,
            text=f"✔️ Marked resolved by <@{user_id}>"
        )
    except SlackApiError:
        pass
    
    log_event("human_resolved", conversation_id=conversation_id, payload={"user": user_id})


async def handle_edit_submit(conversation_id: int, user_id: str, edited_text: str, db: Session):
    """Handle submission of edited reply from modal."""
    from dispatch import send_reply
    from knowledge import feedback_learner
    from utils import get_last_inbound_message
    
    conversation = db.query(Conversation).filter(
        Conversation.id == conversation_id
    ).first()
    
    if not conversation:
        return
    
    # Get original draft for comparison
    original_draft = get_cached_draft(conversation_id) or ""
    
    await send_reply(
        conversation,
        edited_text,
        auto_sent=False,
        confidence=None,
        human_approved_by=user_id,
        human_edited=True,
        db=db
    )
    
    conversation.status = ConversationStatus.resolved
    conversation.last_human_action_at = datetime.utcnow()
    db.commit()
    
    # Record feedback for learning - THIS IS CRITICAL FOR IMPROVING THE AI
    try:
        feedback_id = feedback_learner.record_feedback(
            conversation_id=conversation_id,
            message_id=None,
            original_response=original_draft,
            original_confidence=0.0,
            was_approved=False,
            was_edited=True,
            corrected_response=edited_text,
            correction_type="style",  # Could be enhanced with UI to select type
            human_reviewer=user_id
        )
        
        # Index the correction for future learning
        feedback_learner.index_correction(feedback_id)
        
        # Index successful conversation for future retrieval
        guest_message = get_last_inbound_message(conversation_id, db)
        if guest_message:
            feedback_learner.index_successful_conversation(
                conversation_id=conversation_id,
                guest_message=guest_message,
                ai_response=original_draft,
                was_edited=True,
                edited_response=edited_text
            )
    except Exception as e:
        print(f"[Feedback] Error recording edit: {e}")
    
    cancel_escalation_timers(conversation_id)
    clear_cached_draft(conversation_id)
    
    try:
        slack_client.chat_postMessage(
            channel=conversation.slack_channel_id,
            thread_ts=conversation.slack_thread_ts,
            text=f"✏️ Edited and sent by <@{user_id}>"
        )
    except SlackApiError:
        pass
    
    log_event("human_edited", conversation_id=conversation_id, payload={"user": user_id})


def open_edit_modal(trigger_id: str, conversation_id: int) -> dict:
    """Open Slack modal for editing reply."""
    draft = get_cached_draft(conversation_id) or ""
    
    try:
        slack_client.views_open(
            trigger_id=trigger_id,
            view={
                "type": "modal",
                "callback_id": f"edit_modal_{conversation_id}",
                "title": {"type": "plain_text", "text": "Edit & Send"},
                "submit": {"type": "plain_text", "text": "Send"},
                "close": {"type": "plain_text", "text": "Cancel"},
                "blocks": [
                    {
                        "type": "input",
                        "block_id": "reply_block",
                        "element": {
                            "type": "plain_text_input",
                            "action_id": "reply_input",
                            "multiline": True,
                            "initial_value": draft,
                            "placeholder": {"type": "plain_text", "text": "Type your reply..."}
                        },
                        "label": {"type": "plain_text", "text": "Your Reply"}
                    }
                ],
                "private_metadata": str(conversation_id)
            }
        )
        return {"ok": True}
    except SlackApiError as e:
        return {"ok": False, "error": str(e)}
