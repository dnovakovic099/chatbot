"""
Dispatch module - Send replies via Hostify or OpenPhone.
"""

from datetime import datetime
from typing import Optional

import httpx
from sqlalchemy.orm import Session

from config import settings
from models import Conversation, ConversationStatus
from utils import log_event, save_message
from escalation import post_to_slack


class OpenPhoneClient:
    """Client for interacting with OpenPhone API."""
    
    def __init__(self):
        self.api_key = settings.OPENPHONE_API_KEY
        self.from_number = settings.OPENPHONE_NUMBER
        self.base_url = "https://api.openphone.com/v1"
        self.headers = {
            "Authorization": self.api_key,
            "Content-Type": "application/json"
        }
    
    async def send_message(self, to_number: str, message: str) -> bool:
        """
        Send an SMS message via OpenPhone.
        
        Args:
            to_number: Recipient phone number in E.164 format
            message: Message text to send
            
        Returns:
            True if successful, raises exception otherwise
        """
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.base_url}/messages",
                    headers=self.headers,
                    json={
                        "from": self.from_number,
                        "to": [to_number],
                        "content": message
                    },
                    timeout=30.0
                )
                response.raise_for_status()
                return True
            except httpx.HTTPError as e:
                log_event("api_error", payload={
                    "service": "openphone",
                    "endpoint": "messages",
                    "error": str(e)
                })
                raise


# Global client instance
openphone_client = OpenPhoneClient()


async def send_via_hostify(reservation_id: str, message: str):
    """Send a message via Hostify API."""
    from cache import hostify_client
    await hostify_client.send_message(reservation_id, message)


async def send_via_openphone(phone_number: str, message: str):
    """Send a message via OpenPhone API."""
    await openphone_client.send_message(phone_number, message)


async def send_reply(
    conversation: Conversation,
    text: str,
    auto_sent: bool,
    confidence: Optional[float],
    db: Session,
    human_approved_by: Optional[str] = None,
    human_edited: bool = False
):
    """
    Send reply via appropriate channel (Hostify or OpenPhone).
    
    Args:
        conversation: The conversation to reply to
        text: The message text to send
        auto_sent: Whether this was automatically sent by AI
        confidence: AI confidence score (if applicable)
        db: Database session
        human_approved_by: Slack user ID if approved by human
        human_edited: Whether the message was edited by human
    """
    
    # Check dry run mode
    if settings.DRY_RUN_MODE:
        channel = "hostify" if conversation.hostify_reservation_id else "openphone"
        log_event("dry_run_send", conversation_id=conversation.id, payload={
            "text": text,
            "would_send_via": channel
        })
        print(f"[DRY RUN] Would send to {conversation.guest_phone} via {channel}: {text}")
        
        # Still save the message locally
        save_message(
            conversation_id=conversation.id,
            content=text,
            direction="outbound",
            source="system",
            db=db,
            ai_confidence=confidence,
            was_auto_sent=auto_sent,
            was_human_edited=human_edited
        )
        return
    
    # Increment rate limit counter
    conversation.outbound_count_this_hour += 1
    db.commit()
    
    try:
        if conversation.hostify_reservation_id:
            # Send via Hostify (preferred for guests with reservations)
            await send_via_hostify(conversation.hostify_reservation_id, text)
            source = "hostify"
        else:
            # Send via OpenPhone (fallback for unknown guests)
            await send_via_openphone(conversation.guest_phone, text)
            source = "openphone"
        
        # Log the message
        save_message(
            conversation_id=conversation.id,
            content=text,
            direction="outbound",
            source=source,
            db=db,
            ai_confidence=confidence,
            was_auto_sent=auto_sent,
            was_human_edited=human_edited
        )
        
        # Update conversation status
        if conversation.status in [
            ConversationStatus.escalated_l1, 
            ConversationStatus.escalated_l2, 
            ConversationStatus.snoozed
        ]:
            conversation.status = ConversationStatus.active
        
        conversation.last_message_at = datetime.utcnow()
        db.commit()
        
        log_event(
            "message_sent",
            conversation_id=conversation.id,
            payload={
                "via": source,
                "auto_sent": auto_sent,
                "confidence": confidence,
                "human_approved": human_approved_by is not None,
                "human_edited": human_edited
            }
        )
        
    except Exception as e:
        log_event("api_error", conversation_id=conversation.id, payload={
            "service": "send_reply",
            "error": str(e)
        })
        # Escalate on send failure
        await post_to_slack(
            channel=settings.SLACK_CHANNEL_ERRORS,
            text=f"🔴 *SEND FAILED*\nGuest: {conversation.guest_phone}\nError: {str(e)}",
            conversation_id=conversation.id
        )
        raise
