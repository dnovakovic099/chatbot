"""
Guest cache synchronization logic.
Syncs reservation data from Hostify to local database for fast phone lookups.
"""

import httpx
import asyncio
from datetime import datetime, timedelta
from typing import List, Optional

from sqlalchemy.orm import Session

from config import settings
from models import SessionLocal, GuestIndex, Conversation, ConversationStatus, Message, HostifyMessage
from utils import normalize_phone, log_event


class HostifyClient:
    """Client for interacting with Hostify API."""
    
    def __init__(self):
        self.base_url = settings.HOSTIFY_BASE_URL
        self.api_key = settings.HOSTIFY_API_KEY
        self.headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json"
        }
    
    async def get_listings(self) -> List[dict]:
        """Fetch all listings from Hostify with pagination."""
        all_listings = []
        page = 1
        per_page = 100
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            while True:
                try:
                    print(f"[Hostify] Fetching listings page {page}...")
                    response = await client.get(
                        f"{self.base_url}/listings",
                        headers=self.headers,
                        params={
                            "service_pms": 1,
                            "page": page,
                            "per_page": per_page
                        }
                    )
                    response.raise_for_status()
                    data = response.json()
                    
                    if not data.get("success"):
                        print(f"[Hostify] Listings API returned success=false")
                        break
                    
                    listings = data.get("listings", [])
                    all_listings.extend(listings)
                    print(f"[Hostify] Got {len(listings)} listings (total: {len(all_listings)})")
                    
                    if len(listings) < per_page:
                        break
                    page += 1
                    
                    # Small delay to avoid rate limiting
                    await asyncio.sleep(0.1)
                    
                except httpx.HTTPError as e:
                    log_event("api_error", payload={
                        "service": "hostify",
                        "endpoint": "listings",
                        "error": str(e)
                    })
                    print(f"[Hostify] Error fetching listings: {e}")
                    break
        
        print(f"[Hostify] Total listings fetched: {len(all_listings)}")
        return all_listings
    
    async def get_reservations_for_listing(
        self, 
        listing_id: int,
        start_date: str,
        end_date: str
    ) -> List[dict]:
        """
        Fetch all reservations for a specific listing with pagination.
        
        Args:
            listing_id: Hostify listing ID
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            List of reservation dictionaries
        """
        all_reservations = []
        page = 1
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            while True:
                params = {
                    "listing_id": listing_id,
                    "start_date": start_date,
                    "end_date": end_date,
                    "sort": "checkIn",
                    "page": page
                }
                
                try:
                    response = await client.get(
                        f"{self.base_url}/reservations",
                        headers=self.headers,
                        params=params
                    )
                    response.raise_for_status()
                    data = response.json()
                    
                    if not data.get("success"):
                        break
                    
                    reservations = data.get("reservations", [])
                    all_reservations.extend(reservations)
                    
                    # Check if there are more pages
                    if data.get("next_page") is None or len(reservations) == 0:
                        break
                    page += 1
                    
                    # Small delay to avoid rate limiting
                    await asyncio.sleep(0.05)
                    
                except httpx.HTTPError as e:
                    log_event("api_error", payload={
                        "service": "hostify",
                        "endpoint": f"reservations?listing_id={listing_id}",
                        "error": str(e)
                    })
                    break
        
        return all_reservations
    
    async def get_all_reservations(
        self, 
        start_date: str,
        end_date: str
    ) -> List[dict]:
        """
        Fetch ALL reservations across ALL listings.
        This can take a while for large portfolios.
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            List of all reservation dictionaries
        """
        # First get all listings
        listings = await self.get_listings()
        
        all_reservations = []
        
        print(f"[Hostify] Fetching reservations for {len(listings)} listings ({start_date} to {end_date})...")
        
        for i, listing in enumerate(listings):
            listing_id = listing.get("id")
            listing_name = listing.get("name", "Unknown")
            
            try:
                reservations = await self.get_reservations_for_listing(
                    listing_id,
                    start_date,
                    end_date
                )
                
                # Add listing info to each reservation
                for res in reservations:
                    res["_listing"] = listing
                
                all_reservations.extend(reservations)
                
                if reservations:
                    print(f"[Hostify] [{i+1}/{len(listings)}] {listing_name}: {len(reservations)} reservations")
                
                # Rate limiting delay
                await asyncio.sleep(0.1)
                
            except Exception as e:
                print(f"[Hostify] Error fetching reservations for {listing_name}: {e}")
                continue
        
        print(f"[Hostify] Total reservations fetched: {len(all_reservations)}")
        return all_reservations
    
    async def get_listing(self, listing_id: str) -> Optional[dict]:
        """Fetch listing details from Hostify."""
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                response = await client.get(
                    f"{self.base_url}/listings/{listing_id}",
                    headers=self.headers
                )
                response.raise_for_status()
                return response.json()
            except httpx.HTTPError as e:
                log_event("api_error", payload={
                    "service": "hostify",
                    "endpoint": f"listings/{listing_id}",
                    "error": str(e)
                })
                return None
    
    async def get_inbox_threads(self, limit: int = 100, page: int = 1) -> List[dict]:
        """Fetch all inbox threads (conversations)."""
        async with httpx.AsyncClient(timeout=60.0) as client:
            try:
                response = await client.get(
                    f"{self.base_url}/inbox",
                    headers=self.headers,
                    params={"limit": limit, "page": page}
                )
                response.raise_for_status()
                data = response.json()
                return data.get("threads", [])
            except httpx.HTTPError as e:
                log_event("api_error", payload={
                    "service": "hostify",
                    "endpoint": "inbox",
                    "error": str(e)
                })
                return []
    
    async def get_inbox_messages(self, inbox_id: int, limit: int = 50) -> List[dict]:
        """Fetch messages from a specific inbox thread."""
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                response = await client.get(
                    f"{self.base_url}/inbox/{inbox_id}/messages",
                    headers=self.headers,
                    params={"limit": limit}
                )
                response.raise_for_status()
                data = response.json()
                return data.get("messages", [])
            except httpx.HTTPError as e:
                log_event("api_error", payload={
                    "service": "hostify",
                    "endpoint": f"inbox/{inbox_id}/messages",
                    "error": str(e)
                })
                return []
    
    async def send_message(self, reservation_id: str, message: str) -> bool:
        """Send a message to a guest via Hostify inbox."""
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                # First get the reservation to find the inbox_id
                res_response = await client.get(
                    f"{self.base_url}/reservations/{reservation_id}",
                    headers=self.headers
                )
                res_response.raise_for_status()
                res_data = res_response.json()
                
                inbox_id = res_data.get("reservation", {}).get("inbox_id")
                if not inbox_id:
                    log_event("api_error", payload={
                        "service": "hostify",
                        "error": f"No inbox_id found for reservation {reservation_id}"
                    })
                    return False
                
                # Send message via inbox
                response = await client.post(
                    f"{self.base_url}/inbox/{inbox_id}/messages",
                    headers=self.headers,
                    json={"message": message}
                )
                response.raise_for_status()
                return True
            except httpx.HTTPError as e:
                log_event("api_error", payload={
                    "service": "hostify",
                    "endpoint": "inbox/messages",
                    "error": str(e)
                })
                raise


# Global client instance
hostify_client = HostifyClient()


async def sync_reservations():
    """
    Sync ALL reservations from Hostify to local cache.
    Fetches reservations from 1 year ago to 1 year ahead.
    Syncs ALL guests, even those without phone numbers (for manual matching).
    Saves incrementally as each listing is processed.
    Runs every 30 minutes via APScheduler.
    """
    print("=" * 60)
    print("[Sync] Starting Hostify reservation sync...")
    print("=" * 60)
    
    now = datetime.utcnow()
    start_date = (now - timedelta(days=365)).strftime("%Y-%m-%d")
    end_date = (now + timedelta(days=365)).strftime("%Y-%m-%d")
    
    # Get all listings first
    listings = await hostify_client.get_listings()
    
    print(f"[Sync] Fetching reservations for {len(listings)} listings ({start_date} to {end_date})...")
    
    total_reservations = 0
    synced_with_phone = 0
    synced_without_phone = 0
    active_statuses = {"accepted", "confirmed", "checked_in", "pending", "inquiry"}
    
    for i, listing in enumerate(listings):
        listing_id = listing.get("id")
        listing_name = listing.get("name", "Unknown")
        
        try:
            reservations = await hostify_client.get_reservations_for_listing(
                listing_id,
                start_date,
                end_date
            )
            
            total_reservations += len(reservations)
            
            # Save guests for this listing immediately
            if reservations:
                db = SessionLocal()
                try:
                    for res in reservations:
                        # Filter by status
                        if res.get("status", "").lower() not in active_statuses:
                            continue
                        
                        guest_data = _extract_guest_data_all(res, listing)
                        if guest_data:
                            _upsert_guest_index(db, guest_data)
                            if guest_data.get("guest_phone"):
                                synced_with_phone += 1
                            else:
                                synced_without_phone += 1
                finally:
                    db.close()
                
                print(f"[Sync] [{i+1}/{len(listings)}] {listing_name}: {len(reservations)} reservations (total saved: {synced_with_phone + synced_without_phone})")
            
            # Small delay to avoid rate limiting
            await asyncio.sleep(0.1)
            
        except Exception as e:
            print(f"[Sync] Error on listing {listing_name}: {e}")
            continue
    
    print("=" * 60)
    print(f"[Sync] ✅ Complete!")
    print(f"[Sync] Total reservations fetched: {total_reservations}")
    print(f"[Sync] Synced {synced_with_phone} guests with phone")
    print(f"[Sync] Synced {synced_without_phone} guests without phone")
    print("=" * 60)
    
    log_event("cache_synced", payload={
        "total_fetched": total_reservations,
        "synced_with_phone": synced_with_phone,
        "synced_without_phone": synced_without_phone
    })


async def force_sync_reservations():
    """
    Force an immediate sync of reservations.
    Called when a cache miss occurs.
    """
    log_event("force_sync_triggered")
    await sync_reservations()


async def sync_messages(limit_threads: int = 200):
    """
    Sync messages from Hostify inbox.
    Fetches all inbox threads and their messages.
    Creates Conversation and Message records.
    
    Args:
        limit_threads: Max number of inbox threads to sync
    """
    print("=" * 60)
    print("[Sync Messages] Starting Hostify message sync...")
    print("=" * 60)
    
    # Fetch all inbox threads (paginated)
    all_threads = []
    page = 1
    per_page = 100
    
    while len(all_threads) < limit_threads:
        threads = await hostify_client.get_inbox_threads(limit=per_page, page=page)
        if not threads:
            break
        all_threads.extend(threads)
        print(f"[Sync Messages] Fetched page {page}: {len(threads)} threads (total: {len(all_threads)})")
        if len(threads) < per_page:
            break
        page += 1
        await asyncio.sleep(0.1)
    
    print(f"[Sync Messages] Total threads to process: {len(all_threads)}")
    
    db = SessionLocal()
    conversations_created = 0
    messages_synced = 0
    
    try:
        for i, thread in enumerate(all_threads):
            try:
                inbox_id = thread.get("id")
                reservation_id = str(thread.get("reservation_id", ""))
                guest_name = thread.get("guest_name", "Guest")
                guest_phone = str(thread.get("guest_phone", "")) if thread.get("guest_phone") else None
                listing_name = thread.get("listing_title") or thread.get("listing", "")
                listing_id = str(thread.get("listing_id", ""))
                
                # Parse check-in/check-out dates and source
                checkin_str = thread.get("checkin")
                checkout_str = thread.get("checkout")
                check_in_date = _parse_datetime(checkin_str) if checkin_str else None
                check_out_date = _parse_datetime(checkout_str) if checkout_str else None
                booking_source = thread.get("integration_type_name")  # e.g., "Airbnb", "Vrbo"
                
                # Get or create conversation
                conversation = None
                if reservation_id:
                    conversation = db.query(Conversation).filter(
                        Conversation.hostify_reservation_id == reservation_id
                    ).first()
                
                if not conversation and guest_phone:
                    # Try to find by phone
                    normalized_phone = normalize_phone(guest_phone)
                    if normalized_phone:
                        conversation = db.query(Conversation).filter(
                            Conversation.guest_phone == normalized_phone
                        ).first()
                
                if not conversation:
                    # Create new conversation
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
                else:
                    # Update existing conversation with dates if missing
                    if not conversation.check_in_date and check_in_date:
                        conversation.check_in_date = check_in_date
                    if not conversation.check_out_date and check_out_date:
                        conversation.check_out_date = check_out_date
                    if not conversation.booking_source and booking_source:
                        conversation.booking_source = booking_source
                    conversations_created += 1
                
                # Fetch messages for this thread
                messages = await hostify_client.get_inbox_messages(inbox_id, limit=100)
                
                for msg_data in messages:
                    msg_id = str(msg_data.get("id", ""))
                    
                    # Skip if already exists
                    if db.query(Message).filter(Message.external_id == msg_id).first():
                        continue
                    
                    # Determine direction using Hostify's "from" field
                    # Values: "guest" = from guest, "host" = from host, "automatic" = from host (auto)
                    sender_field = msg_data.get("from", "")
                    msg_guest_name = msg_data.get("guest_name")
                    
                    if sender_field == "guest":
                        direction = "inbound"  # Explicitly from guest
                    elif sender_field == "automatic":
                        direction = "outbound"  # Automated host messages
                    elif sender_field == "host" and msg_guest_name and msg_guest_name.strip() == guest_name.strip():
                        # Hostify bug: sometimes marks guest messages as from="host" 
                        # but includes the guest's name in guest_name field
                        direction = "inbound"
                    else:
                        direction = "outbound"  # From host
                    
                    # Parse timestamp - Hostify uses "created" field
                    msg_time = msg_data.get("created") or msg_data.get("created_at") or msg_data.get("sent_at")
                    sent_at = _parse_datetime(msg_time) if msg_time else datetime.utcnow()
                    
                    content = msg_data.get("message") or msg_data.get("body") or msg_data.get("text") or ""
                    
                    message = Message(
                        conversation_id=conversation.id,
                        direction=direction,
                        source="hostify",
                        content=content,
                        external_id=msg_id,
                        sent_at=sent_at,
                        was_auto_sent=False,  # Historical messages are not auto-sent
                        was_human_edited=False
                    )
                    db.add(message)
                    messages_synced += 1
                
                # Update conversation last_message_at
                last_msg_time = thread.get("last_message")
                if last_msg_time:
                    conversation.last_message_at = _parse_datetime(last_msg_time)
                
                db.commit()
                
                if messages:
                    print(f"[Sync Messages] [{i+1}/{len(all_threads)}] {guest_name}: {len(messages)} messages")
                
                # Rate limiting
                await asyncio.sleep(0.15)
                
            except Exception as e:
                print(f"[Sync Messages] Error on thread {thread.get('id')}: {e}")
                db.rollback()
                continue
        
        print("=" * 60)
        print(f"[Sync Messages] ✅ Complete!")
        print(f"[Sync Messages] Conversations created: {conversations_created}")
        print(f"[Sync Messages] Messages synced: {messages_synced}")
        print("=" * 60)
        
        log_event("messages_synced", payload={
            "threads_processed": len(all_threads),
            "conversations_created": conversations_created,
            "messages_synced": messages_synced
        })
        
    except Exception as e:
        print(f"[Sync Messages] ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        db.rollback()
    finally:
        db.close()


async def sync_single_inbox_thread(inbox_id: str):
    """
    Sync a single inbox thread from Hostify (triggered by webhook).
    
    This is called when we receive a message_new webhook notification.
    It fetches the specific thread and creates/updates the conversation.
    
    Args:
        inbox_id: The Hostify inbox thread ID
    """
    from models import Conversation, Message, MessageDirection, SessionLocal
    
    print(f"[Webhook Sync] Syncing inbox thread {inbox_id}...")
    
    client = HostifyClient()
    db = SessionLocal()
    
    try:
        # Fetch the specific thread's messages
        messages = client.get_inbox_messages(inbox_id)
        
        if not messages:
            print(f"[Webhook Sync] No messages found for inbox {inbox_id}")
            return
        
        # Get thread metadata (we need listing info)
        # Fetch all threads and find the one we want
        threads = client.get_inbox_threads(limit=100)
        thread = next((t for t in threads if str(t.get("id")) == str(inbox_id)), None)
        
        if not thread:
            print(f"[Webhook Sync] Thread {inbox_id} not found in inbox list")
            return
        
        # Extract conversation data
        guest_name = thread.get("guest_name", "")
        listing_name = thread.get("listing_name", "")
        listing_id = str(thread.get("listing_id", ""))
        reservation_id = str(thread.get("reservation_id", "")) if thread.get("reservation_id") else None
        guest_phone = f"inbox_{inbox_id}"  # Use inbox ID as identifier
        
        # Parse dates
        check_in = _parse_datetime(thread.get("check_in"))
        check_out = _parse_datetime(thread.get("check_out"))
        source = thread.get("source") or thread.get("integration", "")
        
        # Find or create conversation
        conv = db.query(Conversation).filter(
            Conversation.guest_phone == guest_phone
        ).first()
        
        if not conv:
            conv = Conversation(
                guest_phone=guest_phone,
                guest_name=guest_name,
                listing_id=listing_id,
                listing_name=listing_name,
                hostify_reservation_id=reservation_id,
                check_in_date=check_in,
                check_out_date=check_out,
                booking_source=source
            )
            db.add(conv)
            db.commit()
            print(f"[Webhook Sync] Created conversation for {guest_name}")
        
        # Sync messages
        new_message_count = 0
        for msg_data in messages:
            msg_id = str(msg_data.get("id", ""))
            
            # Check if message already exists
            existing = db.query(Message).filter(
                Message.external_id == msg_id
            ).first()
            
            if existing:
                continue
            
            # Determine direction
            from_field = msg_data.get("from", "").lower()
            msg_guest_name = msg_data.get("guest_name", "")
            
            if from_field == "guest":
                direction = MessageDirection.inbound
            elif from_field == "host" and msg_guest_name and msg_guest_name.lower() == guest_name.lower():
                direction = MessageDirection.inbound
            else:
                direction = MessageDirection.outbound
            
            # Parse timestamp
            sent_at = _parse_datetime(msg_data.get("created")) or datetime.utcnow()
            content = msg_data.get("message", "")
            
            if not content:
                continue
            
            # Create message
            message = Message(
                conversation_id=conv.id,
                external_id=msg_id,
                direction=direction,
                channel="hostify",
                content=content,
                sent_at=sent_at,
                processed=True
            )
            db.add(message)
            new_message_count += 1
        
        db.commit()
        
        # Update conversation timestamp
        conv.last_message_at = datetime.utcnow()
        db.commit()
        
        print(f"[Webhook Sync] ✅ Synced {new_message_count} new message(s) for {guest_name}")
        
        # Generate AI suggestion for new inbound messages
        if new_message_count > 0:
            last_msg = db.query(Message).filter(
                Message.conversation_id == conv.id
            ).order_by(Message.sent_at.desc()).first()
            
            if last_msg and last_msg.direction == MessageDirection.inbound:
                log_event("new_message_needs_response", payload={
                    "conversation_id": conv.id,
                    "guest_name": guest_name,
                    "inbox_id": inbox_id
                })
                
                # Generate and save AI suggestion
                try:
                    await _generate_and_save_ai_suggestion(conv, last_msg, db)
                except Exception as e:
                    print(f"[Webhook Sync] ⚠️ Failed to generate AI suggestion: {e}")
        
    except Exception as e:
        print(f"[Webhook Sync] ❌ Error syncing inbox {inbox_id}: {e}")
        import traceback
        traceback.print_exc()
        db.rollback()
    finally:
        db.close()


async def _generate_and_save_ai_suggestion(conv, message, db):
    """
    Generate an AI suggestion for an inbound message and save it to the database.
    
    This is called when a new guest message comes in via webhook.
    The suggestion is stored so we can compare it to the actual host response later.
    """
    from brain import generate_ai_response, get_style_examples
    from models import Message, GuestIndex
    
    print(f"[AI Suggestion] Generating suggestion for message {message.id}...")
    
    # Build guest context from conversation
    guest_context = GuestIndex(
        guest_phone=conv.guest_phone or "unknown",
        guest_name=conv.guest_name,
        listing_name=conv.listing_name,
        check_in_date=conv.check_in_date,
        check_out_date=conv.check_out_date,
        source=conv.booking_source or "unknown"
    )
    
    # Get all messages for context
    all_messages = db.query(Message).filter(
        Message.conversation_id == conv.id
    ).order_by(Message.sent_at).all()
    
    # Get style examples
    style_examples = get_style_examples(message.content, n=3)
    
    # Generate AI response
    ai_response = await generate_ai_response(
        messages=all_messages,
        guest_context=guest_context,
        style_examples=style_examples,
        conversation_id=conv.id,
        use_advanced=False  # Use simple mode for speed
    )
    
    # Save the suggestion to the message
    message.ai_suggested_reply = ai_response.reply_text
    message.ai_suggestion_confidence = ai_response.confidence_score
    message.ai_suggestion_reasoning = ai_response.reasoning
    message.ai_suggestion_generated_at = datetime.utcnow()
    
    db.commit()
    
    print(f"[AI Suggestion] ✅ Saved suggestion for message {message.id} (confidence: {ai_response.confidence_score:.0%})")
    
    log_event("ai_suggestion_generated", payload={
        "message_id": message.id,
        "conversation_id": conv.id,
        "confidence": ai_response.confidence_score,
        "suggestion_preview": ai_response.reply_text[:100] if ai_response.reply_text else ""
    })


def _parse_datetime(dt_str: Optional[str]) -> Optional[datetime]:
    """Parse various datetime formats from Hostify."""
    if not dt_str:
        return None
    try:
        # Try ISO format first (with T separator)
        if "T" in dt_str:
            return datetime.fromisoformat(dt_str.replace("Z", "+00:00").replace("+00:00", ""))
        # Try space-separated datetime (Hostify format: "2026-01-20 03:17:43")
        if " " in dt_str and ":" in dt_str:
            return datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
        # Try date only
        return datetime.strptime(dt_str.split(" ")[0], "%Y-%m-%d")
    except:
        return None


def _extract_guest_data(reservation: dict, listing: dict = None) -> Optional[dict]:
    """Extract guest data from a Hostify reservation - only if phone exists."""
    data = _extract_guest_data_all(reservation, listing)
    if data and data.get("guest_phone"):
        return data
    return None


def _extract_guest_data_all(reservation: dict, listing: dict = None) -> Optional[dict]:
    """
    Extract ALL guest data from a Hostify reservation.
    Phone number is optional - guests without phones can be matched later.
    """
    try:
        if listing is None:
            listing = {}
        
        # Normalize phone if available
        phone = reservation.get("guest_phone")
        normalized_phone = None
        is_phone_verified = False
        
        if phone and isinstance(phone, str):
            normalized_phone = normalize_phone(phone)
            if normalized_phone:
                is_phone_verified = True
        
        # Build address from listing components (ensure all are strings)
        address_parts = []
        if listing.get("street"):
            address_parts.append(str(listing.get("street")))
        if listing.get("city"):
            address_parts.append(str(listing.get("city")))
        if listing.get("state"):
            address_parts.append(str(listing.get("state")))
        if listing.get("zipcode"):
            address_parts.append(str(listing.get("zipcode")))
        address = ", ".join(address_parts) if address_parts else ""
        
        # Extract check-in instructions and wifi from custom_fields if available
        custom_fields = reservation.get("custom_fields", []) or listing.get("custom_fields", []) or []
        door_code = ""
        wifi_name = ""
        wifi_password = ""
        special_instructions = reservation.get("notes") or ""
        
        for field in custom_fields:
            field_name = (field.get("name") or "").lower()
            field_value = field.get("value") or ""
            
            if "check-in" in field_name or "checkin" in field_name:
                # Append check-in instructions
                if special_instructions:
                    special_instructions += "\n\n"
                special_instructions += field_value
            elif "wifi" in field_name or "wi-fi" in field_name:
                # Try to parse wifi info from the field
                if "password" in field_name.lower():
                    wifi_password = field_value
                elif "name" in field_name.lower() or "ssid" in field_name.lower():
                    wifi_name = field_value
        
        return {
            "guest_phone": normalized_phone,
            "guest_email": reservation.get("guest_email"),
            "guest_name": reservation.get("guest_name") or "Guest",
            "reservation_id": str(reservation.get("id")),
            "listing_id": str(reservation.get("listing_id", "")),
            "listing_name": listing.get("name") or listing.get("nickname") or "",
            "listing_address": address,
            "check_in_date": _parse_date(reservation.get("checkIn")),
            "check_out_date": _parse_date(reservation.get("checkOut")),
            "door_code": door_code,
            "wifi_name": wifi_name,
            "wifi_password": wifi_password,
            "special_instructions": special_instructions[:4000] if special_instructions else None,
            "inbox_id": reservation.get("message_id"),  # Hostify calls it message_id
            "source": reservation.get("source"),
            "is_phone_verified": is_phone_verified,
        }
    except Exception as e:
        log_event("cache_extraction_error", payload={
            "reservation_id": reservation.get("id"),
            "error": str(e)
        })
        return None


def _parse_date(date_str: Optional[str]) -> Optional[datetime]:
    """Parse a date string to datetime object."""
    if not date_str:
        return None
    try:
        # Extract just the date portion (YYYY-MM-DD) if datetime is provided
        date_part = date_str.split(" ")[0].split("T")[0]
        
        # Parse as noon UTC to avoid timezone issues
        year, month, day = date_part.split("-")
        return datetime(int(year), int(month), int(day), 12, 0, 0)
    except Exception:
        return None


def _upsert_guest_index(db: Session, guest_data: dict):
    """Insert or update guest index entry."""
    try:
        existing = db.query(GuestIndex).filter(
            GuestIndex.reservation_id == guest_data["reservation_id"]
        ).first()
        
        if existing:
            # Update existing record
            for key, value in guest_data.items():
                if hasattr(existing, key):
                    setattr(existing, key, value)
            existing.synced_at = datetime.utcnow()
        else:
            # Create new record
            guest_index = GuestIndex(**guest_data)
            db.add(guest_index)
        
        db.commit()
    except Exception as e:
        db.rollback()
        print(f"[Sync] Error saving guest {guest_data.get('reservation_id')}: {e}")
    
    db.commit()


async def _detect_external_replies(db: Session):
    """
    Detect when a human has replied to a guest outside our system.
    Marks conversations as resolved and cancels pending escalation timers.
    
    Note: This requires inbox message access which may need additional implementation.
    """
    from escalation import cancel_escalation_timers
    
    # Get all active/escalated conversations
    active_conversations = db.query(Conversation).filter(
        Conversation.status.in_([
            ConversationStatus.escalated_l1,
            ConversationStatus.escalated_l2,
            ConversationStatus.active
        ])
    ).all()
    
    for conv in active_conversations:
        # Get the guest's reservation from cache
        guest_index = db.query(GuestIndex).filter(
            GuestIndex.guest_phone == conv.guest_phone
        ).first()
        
        if not guest_index or not conv.hostify_reservation_id:
            continue
        
        # TODO: Fetch inbox messages and check for external replies
        # This requires knowing the inbox_id for the reservation
        # For now, we skip this check
        pass
