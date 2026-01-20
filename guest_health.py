"""
Guest Health Analysis Module.

Provides AI-powered analysis of guest sentiment, complaints, and issues
for currently checked-in guests at monitored properties.

The goal is to help property managers identify unhappy guests early
and take proactive action to minimize negative reviews.

Data flow:
1. HostifyThread table contains all threads (18k+)
2. Guest Health filters by listing_id and check-in/out dates for monitored properties
3. Fetches fresh messages from Hostify API for each checked-in thread
4. AI analyzes messages and saves results to GuestHealthAnalysis table
"""

import json
import asyncio
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from pydantic import BaseModel

from openai import OpenAI
from sqlalchemy.orm import Session

from config import settings
from models import (
    SessionLocal, GuestIndex, GuestHealthSettings, GuestHealthAnalysis,
    SentimentLevel, Conversation, Message, MessageDirection, HostifyMessage,
    HostifyThread, InquiryAnalysis
)
from cache import HostifyClient

# Initialize OpenAI client
client = OpenAI(api_key=settings.OPENAI_API_KEY)


class ComplaintInfo(BaseModel):
    """A single complaint/issue from a guest."""
    issue: str
    department: str  # housekeeping, maintenance, amenities, communication, billing, noise, etc.
    severity: str  # low, medium, high, critical
    status: str  # open, in_progress, resolved
    mentioned_at: Optional[str] = None


class GuestAnalysisResult(BaseModel):
    """Result of AI analysis of a guest's stay."""
    sentiment: str  # very_unhappy, unhappy, neutral, happy, very_happy
    sentiment_score: float  # -1.0 to 1.0
    sentiment_reasoning: str
    complaints: List[ComplaintInfo]
    unresolved_issues: List[Dict[str, Any]]
    resolved_issues: List[Dict[str, Any]]
    risk_level: str  # low, medium, high, critical
    needs_attention: bool
    attention_reason: Optional[str]
    recommended_actions: List[Dict[str, Any]]


class TeamMistake(BaseModel):
    """A mistake made by the team during inquiry handling."""
    mistake: str
    severity: str  # minor, moderate, major, critical
    impact: str  # How it affected conversion


class InquiryAnalysisResult(BaseModel):
    """Result of AI analysis of a lost inquiry."""
    outcome: str  # no_response, price_objection, booked_elsewhere, dates_unavailable, requirements_not_met, ghost, unknown
    outcome_reasoning: str
    
    guest_requirements: List[str]  # What they were looking for
    guest_questions: List[str]  # Questions they asked
    questions_answered: bool
    unanswered_questions: List[str]
    
    team_mistakes: List[TeamMistake]
    team_strengths: List[str]
    response_quality_score: float  # 0-1
    
    conversion_likelihood: float  # 0-1, how likely this could have converted
    lost_revenue_estimate: float  # Estimated $ lost based on dates/property
    
    recommendations: List[Dict[str, Any]]  # [{action, priority, expected_impact}]
    training_opportunities: List[str]  # Areas for team improvement


async def analyze_guest_messages(
    guest_name: str,
    listing_name: str,
    check_in_date: datetime,
    check_out_date: datetime,
    messages: List[Dict[str, Any]],
    nights_stayed: int,
    nights_remaining: int
) -> GuestAnalysisResult:
    """
    Analyze a guest's message history to determine sentiment and issues.
    
    Args:
        guest_name: Name of the guest
        listing_name: Name of the property
        check_in_date: Check-in date
        check_out_date: Check-out date
        messages: List of messages [{direction, content, sent_at}]
        nights_stayed: Number of nights already stayed
        nights_remaining: Number of nights remaining
        
    Returns:
        GuestAnalysisResult with sentiment, complaints, and recommendations
    """
    
    if not messages:
        # No messages = neutral, no issues detected
        return GuestAnalysisResult(
            sentiment="neutral",
            sentiment_score=0.0,
            sentiment_reasoning="No messages exchanged with this guest yet.",
            complaints=[],
            unresolved_issues=[],
            resolved_issues=[],
            risk_level="low",
            needs_attention=False,
            attention_reason=None,
            recommended_actions=[]
        )
    
    # Build conversation history for analysis
    conversation_text = "CONVERSATION HISTORY:\n"
    for msg in messages[-30:]:  # Last 30 messages
        role = "Guest" if msg.get("direction") == "inbound" else "Host"
        timestamp = msg.get("sent_at", "")
        conversation_text += f"[{timestamp}] {role}: {msg.get('content', '')}\n"
    
    # Create analysis prompt
    system_prompt = f"""You are an expert at analyzing guest communication to assess their satisfaction and identify potential issues.

CONTEXT:
- Guest Name: {guest_name}
- Property: {listing_name}
- Check-in: {check_in_date.strftime('%B %d, %Y') if check_in_date else 'Unknown'}
- Check-out: {check_out_date.strftime('%B %d, %Y') if check_out_date else 'Unknown'}
- Nights Stayed: {nights_stayed}
- Nights Remaining: {nights_remaining}

TASK:
Analyze the conversation to determine:
1. Overall guest sentiment/happiness
2. Any complaints or issues raised BY THIS GUEST during THIS STAY
3. Which department each issue belongs to
4. Whether issues were resolved or remain open
5. Risk level for a negative review
6. Recommended actions for the property manager

CRITICAL CONTEXT RULES - READ CAREFULLY:
1. ONLY count issues that THIS GUEST is personally complaining about during THIS stay
2. DO NOT count mentions of previous reviews, past guests' experiences, or historical issues
3. If the guest MENTIONS something from a review they read, that is NOT their complaint unless they're experiencing it themselves
4. If the host mentions a past issue proactively, that's NOT a guest complaint
5. CONSOLIDATE related issues - pool heating, spa temperature, hot tub issues = ONE "Pool/Spa" issue, not multiple
6. If the same underlying problem is mentioned multiple times, count it ONCE
7. Look at WHO is saying what - guest complaints vs host explanations vs general discussion

EXAMPLES OF WHAT IS NOT A COMPLAINT:
- Guest: "I saw a review mentioning bed bugs" -> NOT a complaint unless they found bugs themselves
- Host: "Previous guests mentioned the pool was cold" -> NOT this guest's complaint
- Guest: "Someone said the WiFi was slow" -> NOT a complaint unless they're experiencing it
- Guest asking a question is NOT a complaint

DEPARTMENT CATEGORIES:
- housekeeping: Cleanliness, linens, supplies
- maintenance: Broken items, repairs needed, HVAC, plumbing, electrical
- amenities: Pool, hot tub, spa, WiFi, TV, appliances, parking (COMBINE pool+spa+hot tub issues)
- communication: Response time, unclear instructions, availability
- billing: Payment issues, refund requests, pricing disputes
- noise: Neighbor complaints, construction, traffic
- safety: Security concerns, locks, smoke detectors
- check_in: Access issues, lockbox, directions
- property_condition: General property issues, not as described

SEVERITY LEVELS:
- low: Minor inconvenience, easy fix
- medium: Impacts guest experience but manageable
- high: Significant impact on stay quality
- critical: Safety concern or major service failure

OUTPUT FORMAT (strict JSON):
{{
    "sentiment": "very_unhappy|unhappy|neutral|happy|very_happy",
    "sentiment_score": -1.0 to 1.0,
    "sentiment_reasoning": "Brief explanation of why you assessed this sentiment",
    "complaints": [
        {{
            "issue": "Description of the complaint (consolidated if related)",
            "department": "category from above",
            "severity": "low|medium|high|critical",
            "status": "open|in_progress|resolved",
            "mentioned_at": "when it was first mentioned"
        }}
    ],
    "unresolved_issues": [
        {{"issue": "description", "department": "category", "urgency": "low|medium|high"}}
    ],
    "resolved_issues": [
        {{"issue": "description", "resolution": "how it was resolved"}}
    ],
    "risk_level": "low|medium|high|critical",
    "needs_attention": true/false,
    "attention_reason": "Why this guest needs attention (null if not needed)",
    "recommended_actions": [
        {{"action": "what to do", "priority": "low|medium|high", "reason": "why"}}
    ]
}}

GUIDELINES:
- ONLY include complaints the guest PERSONALLY made about their CURRENT experience
- CONSOLIDATE related issues (all pool/spa/hot tub = one amenities issue)
- Be objective - don't inflate issues based on mentions of others' experiences
- Consider if issues were acknowledged and resolved
- High risk = likely negative review if no action taken
- Recommend proactive outreach for at-risk guests
- Consider timing (guest leaving soon vs. just arrived)"""

    try:
        response = client.chat.completions.create(
            model=settings.OPENAI_MODEL,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"{conversation_text}\n\nAnalyze this guest's satisfaction:"}
            ],
            temperature=0.3,
            max_tokens=1500
        )
        
        result = json.loads(response.choices[0].message.content)
        
        # Parse complaints into ComplaintInfo objects
        complaints = []
        for c in result.get("complaints", []):
            complaints.append(ComplaintInfo(
                issue=c.get("issue", ""),
                department=c.get("department", "general"),
                severity=c.get("severity", "medium"),
                status=c.get("status", "open"),
                mentioned_at=c.get("mentioned_at")
            ))
        
        return GuestAnalysisResult(
            sentiment=result.get("sentiment", "neutral"),
            sentiment_score=result.get("sentiment_score", 0.0),
            sentiment_reasoning=result.get("sentiment_reasoning", ""),
            complaints=complaints,
            unresolved_issues=result.get("unresolved_issues", []),
            resolved_issues=result.get("resolved_issues", []),
            risk_level=result.get("risk_level", "low"),
            needs_attention=result.get("needs_attention", False),
            attention_reason=result.get("attention_reason"),
            recommended_actions=result.get("recommended_actions", [])
        )
        
    except Exception as e:
        print(f"[GuestHealth] AI analysis error: {e}")
        # Return neutral result on error
        return GuestAnalysisResult(
            sentiment="neutral",
            sentiment_score=0.0,
            sentiment_reasoning=f"Analysis error: {str(e)}",
            complaints=[],
            unresolved_issues=[],
            resolved_issues=[],
            risk_level="low",
            needs_attention=False,
            attention_reason=None,
            recommended_actions=[]
        )


def get_monitored_properties(db: Session) -> List[Dict[str, Any]]:
    """Get list of properties that are enabled for monitoring."""
    settings_list = db.query(GuestHealthSettings).filter(
        GuestHealthSettings.is_enabled == True
    ).all()
    
    return [
        {
            "listing_id": s.listing_id,
            "listing_name": s.listing_name
        }
        for s in settings_list
    ]


def get_checked_in_guests(db: Session, listing_ids: List[str]) -> List[GuestIndex]:
    """
    Get all currently checked-in guests for the specified properties.
    (Legacy function - kept for compatibility, prefer fetch_hostify_threads)
    """
    now = datetime.utcnow()
    today = now.replace(hour=0, minute=0, second=0, microsecond=0)
    
    guests = db.query(GuestIndex).filter(
        GuestIndex.listing_id.in_(listing_ids),
        GuestIndex.check_in_date <= now,
        GuestIndex.check_out_date >= today
    ).all()
    
    return guests


def get_guest_messages(db: Session, reservation_id: str, guest_phone: str = None) -> List[Dict[str, Any]]:
    """
    Get all messages for a guest from local database.
    (Legacy function - prefer fetching directly from Hostify API)
    """
    messages = []
    
    # Try to find conversation by reservation_id first
    conversation = db.query(Conversation).filter(
        Conversation.hostify_reservation_id == reservation_id
    ).first()
    
    # If not found, try by phone
    if not conversation and guest_phone:
        conversation = db.query(Conversation).filter(
            Conversation.guest_phone == guest_phone
        ).first()
    
    if conversation:
        # Get messages from conversation
        conv_messages = db.query(Message).filter(
            Message.conversation_id == conversation.id
        ).order_by(Message.sent_at).all()
        
        for m in conv_messages:
            messages.append({
                "direction": m.direction.value if m.direction else "unknown",
                "content": m.content,
                "sent_at": m.sent_at.isoformat() if m.sent_at else ""
            })
    
    # Also check HostifyMessage table
    hostify_msgs = db.query(HostifyMessage).filter(
        HostifyMessage.reservation_id == reservation_id
    ).order_by(HostifyMessage.sent_at).all()
    
    for hm in hostify_msgs:
        # Check if this message is already in our list (by content and time)
        is_duplicate = any(
            m.get("content") == hm.content and 
            abs((datetime.fromisoformat(m.get("sent_at", "2000-01-01")) - hm.sent_at).total_seconds()) < 60
            for m in messages if m.get("sent_at")
        )
        
        if not is_duplicate:
            messages.append({
                "direction": hm.direction,
                "content": hm.content,
                "sent_at": hm.sent_at.isoformat() if hm.sent_at else ""
            })
    
    # Sort by sent_at
    messages.sort(key=lambda x: x.get("sent_at", ""))
    
    return messages


async def analyze_and_save_from_hostify(
    db: Session, 
    guest_data: Dict[str, Any], 
    messages: List[Dict[str, Any]]
) -> Optional[GuestHealthAnalysis]:
    """
    Analyze a guest using fresh data from Hostify API and save/update the analysis.
    
    Args:
        db: Database session
        guest_data: Parsed guest data from Hostify thread
        messages: List of messages from Hostify API
        
    Returns:
        GuestHealthAnalysis record or None on error
    """
    try:
        # Count messages
        total_messages = len(messages)
        guest_messages = len([m for m in messages if m.get("direction") == "inbound"])
        
        # Get last message info
        last_message_at = None
        last_message_from = None
        if messages:
            last_msg = messages[-1]
            if last_msg.get("sent_at"):
                try:
                    # Handle various timestamp formats
                    ts = last_msg["sent_at"]
                    for fmt in ["%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%SZ"]:
                        try:
                            last_message_at = datetime.strptime(ts.split(".")[0], fmt)
                            break
                        except ValueError:
                            continue
                except:
                    pass
            last_message_from = "guest" if last_msg.get("direction") == "inbound" else "host"
        
        # Run AI analysis
        analysis = await analyze_guest_messages(
            guest_name=guest_data.get("guest_name", "Guest"),
            listing_name=guest_data.get("listing_name", "Property"),
            check_in_date=guest_data.get("check_in_date"),
            check_out_date=guest_data.get("check_out_date"),
            messages=messages,
            nights_stayed=guest_data.get("nights_stayed", 0),
            nights_remaining=guest_data.get("nights_remaining", 0)
        )
        
        # Map sentiment string to enum
        sentiment_map = {
            "very_unhappy": SentimentLevel.very_unhappy,
            "unhappy": SentimentLevel.unhappy,
            "neutral": SentimentLevel.neutral,
            "happy": SentimentLevel.happy,
            "very_happy": SentimentLevel.very_happy
        }
        sentiment_enum = sentiment_map.get(analysis.sentiment, SentimentLevel.neutral)
        
        reservation_id = guest_data.get("reservation_id", "")
        
        # Find or create analysis record
        existing = db.query(GuestHealthAnalysis).filter(
            GuestHealthAnalysis.reservation_id == reservation_id
        ).first()
        
        # Find conversation_id if exists in our DB
        conversation = None
        if reservation_id:
            conversation = db.query(Conversation).filter(
                Conversation.hostify_reservation_id == reservation_id
            ).first()
        if not conversation and guest_data.get("guest_phone"):
            conversation = db.query(Conversation).filter(
                Conversation.guest_phone == guest_data.get("guest_phone")
            ).first()
        
        if existing:
            # Update existing record
            existing.guest_name = guest_data.get("guest_name")
            existing.guest_email = guest_data.get("guest_email")
            existing.guest_phone = guest_data.get("guest_phone")
            existing.listing_id = guest_data.get("listing_id")
            existing.listing_name = guest_data.get("listing_name")
            existing.check_in_date = guest_data.get("check_in_date")
            existing.check_out_date = guest_data.get("check_out_date")
            existing.nights_stayed = guest_data.get("nights_stayed")
            existing.nights_remaining = guest_data.get("nights_remaining")
            existing.booking_source = guest_data.get("booking_source")
            existing.sentiment = sentiment_enum
            existing.sentiment_score = analysis.sentiment_score
            existing.sentiment_reasoning = analysis.sentiment_reasoning
            existing.complaints = json.dumps([c.model_dump() for c in analysis.complaints])
            existing.unresolved_issues = json.dumps(analysis.unresolved_issues)
            existing.resolved_issues = json.dumps(analysis.resolved_issues)
            existing.total_messages = total_messages
            existing.guest_messages = guest_messages
            existing.last_message_at = last_message_at
            existing.last_message_from = last_message_from
            existing.risk_level = analysis.risk_level
            existing.needs_attention = analysis.needs_attention
            existing.attention_reason = analysis.attention_reason
            existing.recommended_actions = json.dumps(analysis.recommended_actions)
            existing.last_analyzed_at = datetime.utcnow()
            existing.conversation_id = conversation.id if conversation else None
            
            db.commit()
            return existing
        else:
            # Create new record
            new_analysis = GuestHealthAnalysis(
                reservation_id=reservation_id,
                guest_phone=guest_data.get("guest_phone"),
                guest_name=guest_data.get("guest_name"),
                guest_email=guest_data.get("guest_email"),
                listing_id=guest_data.get("listing_id"),
                listing_name=guest_data.get("listing_name"),
                check_in_date=guest_data.get("check_in_date"),
                check_out_date=guest_data.get("check_out_date"),
                nights_stayed=guest_data.get("nights_stayed"),
                nights_remaining=guest_data.get("nights_remaining"),
                booking_source=guest_data.get("booking_source"),
                sentiment=sentiment_enum,
                sentiment_score=analysis.sentiment_score,
                sentiment_reasoning=analysis.sentiment_reasoning,
                complaints=json.dumps([c.model_dump() for c in analysis.complaints]),
                unresolved_issues=json.dumps(analysis.unresolved_issues),
                resolved_issues=json.dumps(analysis.resolved_issues),
                total_messages=total_messages,
                guest_messages=guest_messages,
                last_message_at=last_message_at,
                last_message_from=last_message_from,
                risk_level=analysis.risk_level,
                needs_attention=analysis.needs_attention,
                attention_reason=analysis.attention_reason,
                recommended_actions=json.dumps(analysis.recommended_actions),
                last_analyzed_at=datetime.utcnow(),
                conversation_id=conversation.id if conversation else None
            )
            db.add(new_analysis)
            db.commit()
            db.refresh(new_analysis)
            return new_analysis
            
    except Exception as e:
        print(f"[GuestHealth] Error analyzing guest {guest_data.get('reservation_id')}: {e}")
        import traceback
        traceback.print_exc()
        db.rollback()
        return None


async def analyze_and_save_guest(db: Session, guest: GuestIndex) -> Optional[GuestHealthAnalysis]:
    """
    Analyze a guest's messages and save/update the analysis.
    
    Args:
        db: Database session
        guest: GuestIndex record for the guest
        
    Returns:
        GuestHealthAnalysis record or None on error
    """
    try:
        # Get messages for this guest
        messages = get_guest_messages(db, guest.reservation_id, guest.guest_phone)
        
        # Calculate nights
        now = datetime.utcnow()
        nights_stayed = 0
        nights_remaining = 0
        
        if guest.check_in_date:
            nights_stayed = max(0, (now - guest.check_in_date).days)
        if guest.check_out_date:
            nights_remaining = max(0, (guest.check_out_date - now).days)
        
        # Count messages
        total_messages = len(messages)
        guest_messages = len([m for m in messages if m.get("direction") == "inbound"])
        
        # Get last message info
        last_message_at = None
        last_message_from = None
        if messages:
            last_msg = messages[-1]
            if last_msg.get("sent_at"):
                try:
                    last_message_at = datetime.fromisoformat(last_msg["sent_at"])
                except:
                    pass
            last_message_from = "guest" if last_msg.get("direction") == "inbound" else "host"
        
        # Run AI analysis
        analysis = await analyze_guest_messages(
            guest_name=guest.guest_name or "Guest",
            listing_name=guest.listing_name or "Property",
            check_in_date=guest.check_in_date,
            check_out_date=guest.check_out_date,
            messages=messages,
            nights_stayed=nights_stayed,
            nights_remaining=nights_remaining
        )
        
        # Map sentiment string to enum
        sentiment_map = {
            "very_unhappy": SentimentLevel.very_unhappy,
            "unhappy": SentimentLevel.unhappy,
            "neutral": SentimentLevel.neutral,
            "happy": SentimentLevel.happy,
            "very_happy": SentimentLevel.very_happy
        }
        sentiment_enum = sentiment_map.get(analysis.sentiment, SentimentLevel.neutral)
        
        # Find or create analysis record
        existing = db.query(GuestHealthAnalysis).filter(
            GuestHealthAnalysis.reservation_id == guest.reservation_id
        ).first()
        
        # Find conversation_id
        conversation = None
        if guest.reservation_id:
            conversation = db.query(Conversation).filter(
                Conversation.hostify_reservation_id == guest.reservation_id
            ).first()
        if not conversation and guest.guest_phone:
            conversation = db.query(Conversation).filter(
                Conversation.guest_phone == guest.guest_phone
            ).first()
        
        if existing:
            # Update existing record
            existing.guest_name = guest.guest_name
            existing.guest_email = guest.guest_email
            existing.guest_phone = guest.guest_phone
            existing.listing_name = guest.listing_name
            existing.check_in_date = guest.check_in_date
            existing.check_out_date = guest.check_out_date
            existing.nights_stayed = nights_stayed
            existing.nights_remaining = nights_remaining
            existing.booking_source = guest.source
            existing.sentiment = sentiment_enum
            existing.sentiment_score = analysis.sentiment_score
            existing.sentiment_reasoning = analysis.sentiment_reasoning
            existing.complaints = json.dumps([c.model_dump() for c in analysis.complaints])
            existing.unresolved_issues = json.dumps(analysis.unresolved_issues)
            existing.resolved_issues = json.dumps(analysis.resolved_issues)
            existing.total_messages = total_messages
            existing.guest_messages = guest_messages
            existing.last_message_at = last_message_at
            existing.last_message_from = last_message_from
            existing.risk_level = analysis.risk_level
            existing.needs_attention = analysis.needs_attention
            existing.attention_reason = analysis.attention_reason
            existing.recommended_actions = json.dumps(analysis.recommended_actions)
            existing.last_analyzed_at = datetime.utcnow()
            existing.conversation_id = conversation.id if conversation else None
            
            db.commit()
            return existing
        else:
            # Create new record
            new_analysis = GuestHealthAnalysis(
                reservation_id=guest.reservation_id,
                guest_phone=guest.guest_phone,
                guest_name=guest.guest_name,
                guest_email=guest.guest_email,
                listing_id=guest.listing_id,
                listing_name=guest.listing_name,
                check_in_date=guest.check_in_date,
                check_out_date=guest.check_out_date,
                nights_stayed=nights_stayed,
                nights_remaining=nights_remaining,
                booking_source=guest.source,
                sentiment=sentiment_enum,
                sentiment_score=analysis.sentiment_score,
                sentiment_reasoning=analysis.sentiment_reasoning,
                complaints=json.dumps([c.model_dump() for c in analysis.complaints]),
                unresolved_issues=json.dumps(analysis.unresolved_issues),
                resolved_issues=json.dumps(analysis.resolved_issues),
                total_messages=total_messages,
                guest_messages=guest_messages,
                last_message_at=last_message_at,
                last_message_from=last_message_from,
                risk_level=analysis.risk_level,
                needs_attention=analysis.needs_attention,
                attention_reason=analysis.attention_reason,
                recommended_actions=json.dumps(analysis.recommended_actions),
                last_analyzed_at=datetime.utcnow(),
                conversation_id=conversation.id if conversation else None
            )
            db.add(new_analysis)
            db.commit()
            db.refresh(new_analysis)
            return new_analysis
            
    except Exception as e:
        print(f"[GuestHealth] Error analyzing guest {guest.reservation_id}: {e}")
        import traceback
        traceback.print_exc()
        db.rollback()
        return None


async def refresh_all_guest_health(db: Session) -> Dict[str, Any]:
    """
    Refresh guest health analysis for all checked-in guests at monitored properties.
    
    Uses HostifyThread table (which has ALL threads from Hostify) filtered by 
    listing_id and check-in/check-out dates.
    
    Returns:
        Summary of the refresh operation
    """
    # Get monitored properties
    monitored = get_monitored_properties(db)
    
    if not monitored:
        return {
            "status": "no_properties",
            "message": "No properties are configured for monitoring. Add properties in Settings.",
            "guests_analyzed": 0
        }
    
    listing_ids = [p["listing_id"] for p in monitored]
    
    # Get checked-in guests from HostifyThread table
    # HostifyThread uses listing_ids from the inbox API (matches our monitored properties)
    # A guest is checked in if: checkin <= now AND checkout >= today
    # We also filter for threads WITH a reservation_id (confirmed bookings, not inquiries)
    now = datetime.utcnow()
    today = now.replace(hour=0, minute=0, second=0, microsecond=0)
    
    # Query threads - a guest is checked in if: checkin <= now AND checkout >= today
    # Having checkin/checkout dates indicates an actual booking (not just an inquiry)
    threads = db.query(HostifyThread).filter(
        HostifyThread.listing_id.in_(listing_ids),
        HostifyThread.checkin.isnot(None),
        HostifyThread.checkin <= now,
        HostifyThread.checkout >= today
    ).all()
    
    print(f"[GuestHealth] Found {len(threads)} checked-in threads at {len(monitored)} monitored properties", flush=True)
    print(f"[GuestHealth] Monitored listing_ids: {listing_ids[:5]}...", flush=True)
    
    analyzed = 0
    errors = 0
    
    for i, thread in enumerate(threads):
        print(f"[GuestHealth] [{i+1}/{len(threads)}] Analyzing {thread.guest_name}...")
        result = await analyze_hostify_thread(db, thread)
        if result:
            analyzed += 1
        else:
            errors += 1
        # Small delay to be nice to Hostify API
        await asyncio.sleep(0.2)
    
    return {
        "status": "complete",
        "properties_monitored": len(monitored),
        "guests_found": len(threads),
        "guests_analyzed": analyzed,
        "errors": errors
    }


def _parse_datetime(dt_str: str) -> Optional[datetime]:
    """Parse datetime string from Hostify API."""
    if not dt_str:
        return None
    for fmt in ["%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"]:
        try:
            return datetime.strptime(dt_str, fmt)
        except:
            continue
    return None


async def fetch_thread_messages(db: Session, thread: HostifyThread) -> int:
    """
    Fetch messages from Hostify API for a specific thread and save to HostifyMessage table.
    Returns number of new messages saved.
    """
    hostify_client = HostifyClient()
    
    try:
        messages = await hostify_client.get_inbox_messages(thread.thread_id, limit=100)
        
        new_count = 0
        for msg_data in messages:
            msg_id = msg_data.get("id")
            if not msg_id:
                continue
            
            # Skip if already exists
            existing = db.query(HostifyMessage).filter(
                HostifyMessage.hostify_message_id == msg_id
            ).first()
            if existing:
                continue
            
            # Parse message data
            content = msg_data.get("message") or msg_data.get("body") or ""
            sender_field = msg_data.get("from", "")
            
            # Determine direction
            if sender_field == "guest":
                direction = "inbound"
                sender_type = "guest"
            else:
                direction = "outbound"
                sender_type = sender_field or "host"
            
            msg_time = msg_data.get("created") or msg_data.get("sent_at")
            sent_at = _parse_datetime(msg_time) if msg_time else datetime.utcnow()
            
            sender_name = msg_data.get("sender_name") or msg_data.get("name") or ""
            
            # Create message
            new_msg = HostifyMessage(
                hostify_message_id=msg_id,
                inbox_id=thread.thread_id,
                reservation_id=thread.reservation_id,
                content=content,
                direction=direction,
                sender_name=sender_name,
                sender_type=sender_type,
                guest_name=thread.guest_name,
                sent_at=sent_at
            )
            db.add(new_msg)
            new_count += 1
        
        # Update thread's messages_synced_at
        thread.messages_synced_at = datetime.utcnow()
        thread.message_count = len(messages)
        db.commit()
        
        if new_count > 0:
            print(f"[GuestHealth] Fetched {new_count} new messages for {thread.guest_name}")
        
        return new_count
        
    except Exception as e:
        print(f"[GuestHealth] Error fetching messages for thread {thread.thread_id}: {e}")
        db.rollback()
        return 0


async def analyze_hostify_thread(db: Session, thread: HostifyThread) -> Optional[GuestHealthAnalysis]:
    """
    Analyze a HostifyThread's messages and save/update the analysis.
    First fetches fresh messages from Hostify API, then analyzes.
    """
    try:
        # STEP 1: Fetch fresh messages from Hostify API
        await fetch_thread_messages(db, thread)
        
        # STEP 2: Get messages from HostifyMessage table for this thread
        messages_query = db.query(HostifyMessage).filter(
            HostifyMessage.inbox_id == thread.thread_id
        ).order_by(HostifyMessage.sent_at).all()
        
        messages = []
        for m in messages_query:
            # Determine direction from sender_type or direction field
            direction = m.direction if m.direction else "unknown"
            if m.sender_type == "guest":
                direction = "inbound"
            elif m.sender_type in ["host", "automatic"]:
                direction = "outbound"
            
            messages.append({
                "direction": direction,
                "content": m.content or "",
                "sent_at": m.sent_at.isoformat() if m.sent_at else ""
            })
        
        # Calculate nights
        now = datetime.utcnow()
        nights_stayed = 0
        nights_remaining = 0
        
        if thread.checkin:
            nights_stayed = max(0, (now - thread.checkin).days)
        if thread.checkout:
            nights_remaining = max(0, (thread.checkout - now).days)
        
        # Count messages
        total_messages = len(messages)
        guest_message_count = len([m for m in messages if m.get("direction") == "inbound"])
        
        # Get last message info
        last_message_at = None
        last_message_from = None
        if messages:
            last_msg = messages[-1]
            if last_msg.get("sent_at"):
                try:
                    last_message_at = datetime.fromisoformat(last_msg["sent_at"])
                except:
                    pass
            last_message_from = "guest" if last_msg.get("direction") == "inbound" else "host"
        
        # Run AI analysis
        analysis = await analyze_guest_messages(
            guest_name=thread.guest_name or "Guest",
            listing_name=thread.listing_name or "Property",
            check_in_date=thread.checkin,
            check_out_date=thread.checkout,
            messages=messages,
            nights_stayed=nights_stayed,
            nights_remaining=nights_remaining
        )
        
        # Map sentiment string to enum
        sentiment_map = {
            "very_unhappy": SentimentLevel.very_unhappy,
            "unhappy": SentimentLevel.unhappy,
            "neutral": SentimentLevel.neutral,
            "happy": SentimentLevel.happy,
            "very_happy": SentimentLevel.very_happy
        }
        sentiment_enum = sentiment_map.get(analysis.sentiment, SentimentLevel.neutral)
        
        # Use reservation_id or thread_id as the key
        reservation_id = thread.reservation_id or f"thread_{thread.thread_id}"
        
        # Find or create analysis record
        existing = db.query(GuestHealthAnalysis).filter(
            GuestHealthAnalysis.reservation_id == reservation_id
        ).first()
        
        if existing:
            # Update existing record
            existing.guest_name = thread.guest_name
            existing.listing_id = thread.listing_id
            existing.listing_name = thread.listing_name
            existing.check_in_date = thread.checkin
            existing.check_out_date = thread.checkout
            existing.nights_stayed = nights_stayed
            existing.nights_remaining = nights_remaining
            existing.sentiment = sentiment_enum
            existing.sentiment_score = analysis.sentiment_score
            existing.sentiment_reasoning = analysis.sentiment_reasoning
            existing.complaints = json.dumps([c.model_dump() for c in analysis.complaints])
            existing.unresolved_issues = json.dumps(analysis.unresolved_issues)
            existing.resolved_issues = json.dumps(analysis.resolved_issues)
            existing.total_messages = total_messages
            existing.guest_messages = guest_message_count
            existing.last_message_at = last_message_at
            existing.last_message_from = last_message_from
            existing.risk_level = analysis.risk_level
            existing.needs_attention = analysis.needs_attention
            existing.attention_reason = analysis.attention_reason
            existing.recommended_actions = json.dumps(analysis.recommended_actions)
            existing.last_analyzed_at = datetime.utcnow()
            
            db.commit()
            print(f"[GuestHealth] Updated: {thread.guest_name} - {total_messages} messages")
            return existing
        else:
            # Create new record
            new_analysis = GuestHealthAnalysis(
                reservation_id=reservation_id,
                guest_name=thread.guest_name,
                listing_id=thread.listing_id,
                listing_name=thread.listing_name,
                check_in_date=thread.checkin,
                check_out_date=thread.checkout,
                nights_stayed=nights_stayed,
                nights_remaining=nights_remaining,
                sentiment=sentiment_enum,
                sentiment_score=analysis.sentiment_score,
                sentiment_reasoning=analysis.sentiment_reasoning,
                complaints=json.dumps([c.model_dump() for c in analysis.complaints]),
                unresolved_issues=json.dumps(analysis.unresolved_issues),
                resolved_issues=json.dumps(analysis.resolved_issues),
                total_messages=total_messages,
                guest_messages=guest_message_count,
                last_message_at=last_message_at,
                last_message_from=last_message_from,
                risk_level=analysis.risk_level,
                needs_attention=analysis.needs_attention,
                attention_reason=analysis.attention_reason,
                recommended_actions=json.dumps(analysis.recommended_actions),
                last_analyzed_at=datetime.utcnow()
            )
            db.add(new_analysis)
            db.commit()
            db.refresh(new_analysis)
            print(f"[GuestHealth] Created: {thread.guest_name} - {total_messages} messages")
            return new_analysis
            
    except Exception as e:
        print(f"[GuestHealth] Error analyzing thread {thread.thread_id}: {e}")
        import traceback
        traceback.print_exc()
        db.rollback()
        return None


async def analyze_conversation(db: Session, conv: Conversation) -> Optional[GuestHealthAnalysis]:
    """
    Analyze a conversation's messages and save/update the analysis.
    Uses the Conversation record directly (which has messages).
    """
    try:
        # Get messages from this conversation
        messages_query = db.query(Message).filter(
            Message.conversation_id == conv.id
        ).order_by(Message.sent_at).all()
        
        messages = []
        for m in messages_query:
            messages.append({
                "direction": m.direction.value if m.direction else "unknown",
                "content": m.content or "",
                "sent_at": m.sent_at.isoformat() if m.sent_at else ""
            })
        
        # Calculate nights
        now = datetime.utcnow()
        nights_stayed = 0
        nights_remaining = 0
        
        if conv.check_in_date:
            nights_stayed = max(0, (now - conv.check_in_date).days)
        if conv.check_out_date:
            nights_remaining = max(0, (conv.check_out_date - now).days)
        
        # Count messages
        total_messages = len(messages)
        guest_message_count = len([m for m in messages if m.get("direction") == "inbound"])
        
        # Get last message info
        last_message_at = None
        last_message_from = None
        if messages:
            last_msg = messages[-1]
            if last_msg.get("sent_at"):
                try:
                    last_message_at = datetime.fromisoformat(last_msg["sent_at"])
                except:
                    pass
            last_message_from = "guest" if last_msg.get("direction") == "inbound" else "host"
        
        # Run AI analysis
        analysis = await analyze_guest_messages(
            guest_name=conv.guest_name or "Guest",
            listing_name=conv.listing_name or "Property",
            check_in_date=conv.check_in_date,
            check_out_date=conv.check_out_date,
            messages=messages,
            nights_stayed=nights_stayed,
            nights_remaining=nights_remaining
        )
        
        # Map sentiment string to enum
        sentiment_map = {
            "very_unhappy": SentimentLevel.very_unhappy,
            "unhappy": SentimentLevel.unhappy,
            "neutral": SentimentLevel.neutral,
            "happy": SentimentLevel.happy,
            "very_happy": SentimentLevel.very_happy
        }
        sentiment_enum = sentiment_map.get(analysis.sentiment, SentimentLevel.neutral)
        
        # Use hostify_reservation_id as the key
        reservation_id = conv.hostify_reservation_id or f"conv_{conv.id}"
        
        # Find or create analysis record
        existing = db.query(GuestHealthAnalysis).filter(
            GuestHealthAnalysis.reservation_id == reservation_id
        ).first()
        
        if existing:
            # Update existing record
            existing.guest_name = conv.guest_name
            existing.guest_phone = conv.guest_phone
            existing.listing_id = conv.listing_id
            existing.listing_name = conv.listing_name
            existing.check_in_date = conv.check_in_date
            existing.check_out_date = conv.check_out_date
            existing.nights_stayed = nights_stayed
            existing.nights_remaining = nights_remaining
            existing.booking_source = conv.booking_source
            existing.sentiment = sentiment_enum
            existing.sentiment_score = analysis.sentiment_score
            existing.sentiment_reasoning = analysis.sentiment_reasoning
            existing.complaints = json.dumps([c.model_dump() for c in analysis.complaints])
            existing.unresolved_issues = json.dumps(analysis.unresolved_issues)
            existing.resolved_issues = json.dumps(analysis.resolved_issues)
            existing.total_messages = total_messages
            existing.guest_messages = guest_message_count
            existing.last_message_at = last_message_at
            existing.last_message_from = last_message_from
            existing.risk_level = analysis.risk_level
            existing.needs_attention = analysis.needs_attention
            existing.attention_reason = analysis.attention_reason
            existing.recommended_actions = json.dumps(analysis.recommended_actions)
            existing.last_analyzed_at = datetime.utcnow()
            existing.conversation_id = conv.id
            
            db.commit()
            print(f"[GuestHealth] Updated: {conv.guest_name} - {total_messages} messages")
            return existing
        else:
            # Create new record
            new_analysis = GuestHealthAnalysis(
                reservation_id=reservation_id,
                guest_phone=conv.guest_phone,
                guest_name=conv.guest_name,
                listing_id=conv.listing_id,
                listing_name=conv.listing_name,
                check_in_date=conv.check_in_date,
                check_out_date=conv.check_out_date,
                nights_stayed=nights_stayed,
                nights_remaining=nights_remaining,
                booking_source=conv.booking_source,
                sentiment=sentiment_enum,
                sentiment_score=analysis.sentiment_score,
                sentiment_reasoning=analysis.sentiment_reasoning,
                complaints=json.dumps([c.model_dump() for c in analysis.complaints]),
                unresolved_issues=json.dumps(analysis.unresolved_issues),
                resolved_issues=json.dumps(analysis.resolved_issues),
                total_messages=total_messages,
                guest_messages=guest_message_count,
                last_message_at=last_message_at,
                last_message_from=last_message_from,
                risk_level=analysis.risk_level,
                needs_attention=analysis.needs_attention,
                attention_reason=analysis.attention_reason,
                recommended_actions=json.dumps(analysis.recommended_actions),
                last_analyzed_at=datetime.utcnow(),
                conversation_id=conv.id
            )
            db.add(new_analysis)
            db.commit()
            db.refresh(new_analysis)
            print(f"[GuestHealth] Created: {conv.guest_name} - {total_messages} messages")
            return new_analysis
            
    except Exception as e:
        print(f"[GuestHealth] Error analyzing conversation {conv.id}: {e}")
        import traceback
        traceback.print_exc()
        db.rollback()
        return None


def get_guest_health_summary(db: Session) -> List[Dict[str, Any]]:
    """
    Get a summary of all current guest health analyses.
    Sorted by risk level and needs_attention flag.
    """
    # Get all analyses for currently checked-in guests
    now = datetime.utcnow()
    today = now.replace(hour=0, minute=0, second=0, microsecond=0)
    
    analyses = db.query(GuestHealthAnalysis).filter(
        GuestHealthAnalysis.check_out_date >= today
    ).all()
    
    # Build a mapping of listing_id -> listing_name from GuestHealthSettings and HostifyThread tables
    # for cases where listing_name is not stored in the analysis
    listing_ids = list(set(a.listing_id for a in analyses if a.listing_id))
    listing_name_map = {}
    if listing_ids:
        # First try GuestHealthSettings (most reliable source for property names)
        settings_with_names = db.query(GuestHealthSettings.listing_id, GuestHealthSettings.listing_name).filter(
            GuestHealthSettings.listing_id.in_(listing_ids),
            GuestHealthSettings.listing_name.isnot(None)
        ).all()
        for lid, lname in settings_with_names:
            if lid and lname:
                listing_name_map[lid] = lname
        
        # Fall back to HostifyThread for any missing
        missing_ids = [lid for lid in listing_ids if lid not in listing_name_map]
        if missing_ids:
            threads_with_names = db.query(HostifyThread.listing_id, HostifyThread.listing_name).filter(
                HostifyThread.listing_id.in_(missing_ids),
                HostifyThread.listing_name.isnot(None)
            ).distinct().all()
            for lid, lname in threads_with_names:
                if lid and lname:
                    listing_name_map[lid] = lname
    
    result = []
    for a in analyses:
        # Use stored listing_name, or look it up from the map
        listing_name = a.listing_name or listing_name_map.get(a.listing_id)
        result.append({
            "id": a.id,
            "reservation_id": a.reservation_id,
            "guest_name": a.guest_name,
            "guest_phone": a.guest_phone,
            "listing_id": a.listing_id,
            "listing_name": listing_name,
            "check_in_date": a.check_in_date.isoformat() if a.check_in_date else None,
            "check_out_date": a.check_out_date.isoformat() if a.check_out_date else None,
            "nights_stayed": a.nights_stayed,
            "nights_remaining": a.nights_remaining,
            "booking_source": a.booking_source,
            "sentiment": a.sentiment.value if a.sentiment else "neutral",
            "sentiment_score": a.sentiment_score,
            "sentiment_reasoning": a.sentiment_reasoning,
            "complaints": json.loads(a.complaints) if a.complaints else [],
            "unresolved_issues": json.loads(a.unresolved_issues) if a.unresolved_issues else [],
            "resolved_issues": json.loads(a.resolved_issues) if a.resolved_issues else [],
            "total_messages": a.total_messages,
            "guest_messages": a.guest_messages,
            "last_message_at": a.last_message_at.isoformat() if a.last_message_at else None,
            "last_message_from": a.last_message_from,
            "risk_level": a.risk_level,
            "needs_attention": a.needs_attention,
            "attention_reason": a.attention_reason,
            "recommended_actions": json.loads(a.recommended_actions) if a.recommended_actions else [],
            "last_analyzed_at": a.last_analyzed_at.isoformat() if a.last_analyzed_at else None,
            "conversation_id": a.conversation_id
        })
    
    # Sort: needs_attention first, then by risk level, then by sentiment score
    risk_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    result.sort(key=lambda x: (
        not x["needs_attention"],
        risk_order.get(x["risk_level"], 4),
        x["sentiment_score"] if x["sentiment_score"] is not None else 0
    ))
    
    return result


# ============ INQUIRY ANALYSIS ============

async def analyze_inquiry_messages(
    guest_name: str,
    listing_name: str,
    requested_checkin: Optional[datetime],
    requested_checkout: Optional[datetime],
    messages: List[Dict[str, Any]],
    first_response_minutes: Optional[int],
    nightly_rate_estimate: float = 500.0  # Default estimate
) -> InquiryAnalysisResult:
    """
    Use AI to analyze why an inquiry didn't convert to a booking.
    
    Args:
        guest_name: Name of the potential guest
        listing_name: Property they inquired about
        requested_checkin: Dates they wanted
        requested_checkout: Dates they wanted
        messages: List of messages in the conversation
        first_response_minutes: How long until first team response
        nightly_rate_estimate: Estimated nightly rate for revenue calculation
    
    Returns:
        InquiryAnalysisResult with detailed analysis
    """
    # Format messages for AI
    conversation_text = f"INQUIRY DETAILS:\n"
    conversation_text += f"Guest: {guest_name}\n"
    conversation_text += f"Property: {listing_name}\n"
    
    if requested_checkin and requested_checkout:
        nights = (requested_checkout - requested_checkin).days
        conversation_text += f"Requested dates: {requested_checkin.strftime('%b %d')} - {requested_checkout.strftime('%b %d')} ({nights} nights)\n"
    
    if first_response_minutes is not None:
        if first_response_minutes < 60:
            conversation_text += f"First response time: {first_response_minutes} minutes\n"
        else:
            conversation_text += f"First response time: {first_response_minutes // 60} hours {first_response_minutes % 60} minutes\n"
    
    conversation_text += f"\nCONVERSATION:\n"
    
    for msg in messages:
        direction = msg.get("direction", "unknown")
        sender = "GUEST" if direction == "inbound" else "TEAM"
        content = msg.get("content", "")
        sent_at = msg.get("sent_at", "")
        conversation_text += f"[{sender}] {content}\n"
    
    system_prompt = """You are an expert at analyzing vacation rental inquiries to understand why potential guests didn't book.

Analyze this inquiry conversation and provide insights for the property manager.

OUTCOME CATEGORIES:
- "booked_elsewhere": Guest likely booked a different property
- "price_objection": Guest found price too high
- "dates_unavailable": Requested dates weren't available
- "requirements_not_met": Property didn't meet their needs (pets, amenities, etc.)
- "slow_response": Team took too long to respond
- "poor_communication": Team responses were unhelpful or unclear
- "ghost": Guest stopped responding without explanation
- "no_response": Team never responded to guest
- "still_deciding": Conversation seems ongoing/undecided
- "unknown": Can't determine reason

TEAM MISTAKES TO LOOK FOR:
- Slow initial response (>1 hour is concerning, >4 hours is bad)
- Not answering specific questions
- Not addressing concerns or objections
- Missing upsell opportunities
- Being too brief or impersonal
- Not following up when guest went quiet
- Not offering alternatives when dates unavailable
- Being pushy or aggressive
- Giving incorrect information

Respond with a JSON object:
{
    "outcome": "one of the categories above",
    "outcome_reasoning": "Detailed explanation of why they didn't book",
    
    "guest_requirements": ["list", "of", "what they wanted"],
    "guest_questions": ["specific questions they asked"],
    "questions_answered": true/false,
    "unanswered_questions": ["questions that weren't answered"],
    
    "team_mistakes": [
        {"mistake": "what went wrong", "severity": "minor|moderate|major|critical", "impact": "how it affected conversion"}
    ],
    "team_strengths": ["what the team did well"],
    "response_quality_score": 0.0-1.0,
    
    "conversion_likelihood": 0.0-1.0,
    "lost_revenue_estimate": estimated_dollars_lost,
    
    "recommendations": [
        {"action": "what to do differently", "priority": "low|medium|high", "expected_impact": "how it would help"}
    ],
    "training_opportunities": ["areas where team could improve"]
}

Be objective and constructive. The goal is to help the team improve, not to blame them."""

    try:
        response = client.chat.completions.create(
            model=settings.OPENAI_MODEL,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"{conversation_text}\n\nAnalyze why this inquiry didn't convert:"}
            ],
            temperature=0.3,
            max_tokens=2000
        )
        
        result = json.loads(response.choices[0].message.content)
        
        # Parse team mistakes
        team_mistakes = []
        for m in result.get("team_mistakes", []):
            team_mistakes.append(TeamMistake(
                mistake=m.get("mistake", ""),
                severity=m.get("severity", "minor"),
                impact=m.get("impact", "")
            ))
        
        # Calculate lost revenue if not provided
        lost_revenue = result.get("lost_revenue_estimate", 0)
        if not lost_revenue and requested_checkin and requested_checkout:
            nights = (requested_checkout - requested_checkin).days
            lost_revenue = nights * nightly_rate_estimate * result.get("conversion_likelihood", 0.5)
        
        return InquiryAnalysisResult(
            outcome=result.get("outcome", "unknown"),
            outcome_reasoning=result.get("outcome_reasoning", ""),
            guest_requirements=result.get("guest_requirements", []),
            guest_questions=result.get("guest_questions", []),
            questions_answered=result.get("questions_answered", True),
            unanswered_questions=result.get("unanswered_questions", []),
            team_mistakes=team_mistakes,
            team_strengths=result.get("team_strengths", []),
            response_quality_score=result.get("response_quality_score", 0.5),
            conversion_likelihood=result.get("conversion_likelihood", 0.5),
            lost_revenue_estimate=lost_revenue,
            recommendations=result.get("recommendations", []),
            training_opportunities=result.get("training_opportunities", [])
        )
        
    except Exception as e:
        print(f"[InquiryAnalysis] AI error: {e}")
        return InquiryAnalysisResult(
            outcome="unknown",
            outcome_reasoning="Analysis failed",
            guest_requirements=[],
            guest_questions=[],
            questions_answered=True,
            unanswered_questions=[],
            team_mistakes=[],
            team_strengths=[],
            response_quality_score=0.5,
            conversion_likelihood=0.5,
            lost_revenue_estimate=0,
            recommendations=[],
            training_opportunities=[]
        )


async def analyze_and_save_inquiry(db: Session, thread: HostifyThread) -> Optional[InquiryAnalysis]:
    """
    Analyze an inquiry thread and save the analysis.
    """
    try:
        # Fetch messages for this thread
        await fetch_thread_messages(db, thread)
        
        # Get messages
        messages_query = db.query(HostifyMessage).filter(
            HostifyMessage.inbox_id == thread.thread_id
        ).order_by(HostifyMessage.sent_at).all()
        
        messages = []
        first_guest_msg_time = None
        first_team_response_time = None
        
        for m in messages_query:
            direction = m.direction if m.direction else "unknown"
            if m.sender_type == "guest":
                direction = "inbound"
                if first_guest_msg_time is None:
                    first_guest_msg_time = m.sent_at
            elif m.sender_type in ["host", "automatic"]:
                direction = "outbound"
                if first_team_response_time is None and first_guest_msg_time is not None:
                    first_team_response_time = m.sent_at
            
            messages.append({
                "direction": direction,
                "content": m.content or "",
                "sent_at": m.sent_at.isoformat() if m.sent_at else ""
            })
        
        # Calculate first response time
        first_response_minutes = None
        if first_guest_msg_time and first_team_response_time:
            delta = first_team_response_time - first_guest_msg_time
            first_response_minutes = int(delta.total_seconds() / 60)
        
        # Count messages
        total_messages = len(messages)
        guest_messages = len([m for m in messages if m.get("direction") == "inbound"])
        team_messages = total_messages - guest_messages
        
        # Calculate conversation duration
        conversation_duration_hours = None
        if messages:
            first_msg_time = None
            last_msg_time = None
            for m in messages:
                if m.get("sent_at"):
                    try:
                        msg_time = datetime.fromisoformat(m["sent_at"])
                        if first_msg_time is None:
                            first_msg_time = msg_time
                        last_msg_time = msg_time
                    except:
                        pass
            if first_msg_time and last_msg_time:
                conversation_duration_hours = (last_msg_time - first_msg_time).total_seconds() / 3600
        
        # Run AI analysis
        analysis = await analyze_inquiry_messages(
            guest_name=thread.guest_name or "Guest",
            listing_name=thread.listing_name or "Property",
            requested_checkin=thread.checkin,
            requested_checkout=thread.checkout,
            messages=messages,
            first_response_minutes=first_response_minutes
        )
        
        # Find or update existing record
        existing = db.query(InquiryAnalysis).filter(
            InquiryAnalysis.thread_id == thread.thread_id
        ).first()
        
        if existing:
            existing.guest_name = thread.guest_name
            existing.guest_email = thread.guest_email
            existing.listing_id = thread.listing_id
            existing.listing_name = thread.listing_name
            existing.inquiry_date = thread.last_message_at
            existing.requested_checkin = thread.checkin
            existing.requested_checkout = thread.checkout
            existing.first_response_minutes = first_response_minutes
            existing.total_messages = total_messages
            existing.team_messages = team_messages
            existing.guest_messages = guest_messages
            existing.conversation_duration_hours = conversation_duration_hours
            existing.outcome = analysis.outcome
            existing.outcome_reasoning = analysis.outcome_reasoning
            existing.guest_requirements = json.dumps(analysis.guest_requirements)
            existing.guest_questions = json.dumps(analysis.guest_questions)
            existing.questions_answered = analysis.questions_answered
            existing.unanswered_questions = json.dumps(analysis.unanswered_questions)
            existing.team_mistakes = json.dumps([m.model_dump() for m in analysis.team_mistakes])
            existing.team_strengths = json.dumps(analysis.team_strengths)
            existing.response_quality_score = analysis.response_quality_score
            existing.conversion_likelihood = analysis.conversion_likelihood
            existing.lost_revenue_estimate = analysis.lost_revenue_estimate
            existing.recommendations = json.dumps(analysis.recommendations)
            existing.training_opportunities = json.dumps(analysis.training_opportunities)
            existing.analyzed_at = datetime.utcnow()
            
            db.commit()
            print(f"[InquiryAnalysis] Updated: {thread.guest_name} - {analysis.outcome}")
            return existing
        else:
            new_analysis = InquiryAnalysis(
                thread_id=thread.thread_id,
                guest_name=thread.guest_name,
                guest_email=thread.guest_email,
                listing_id=thread.listing_id,
                listing_name=thread.listing_name,
                inquiry_date=thread.last_message_at,
                requested_checkin=thread.checkin,
                requested_checkout=thread.checkout,
                first_response_minutes=first_response_minutes,
                total_messages=total_messages,
                team_messages=team_messages,
                guest_messages=guest_messages,
                conversation_duration_hours=conversation_duration_hours,
                outcome=analysis.outcome,
                outcome_reasoning=analysis.outcome_reasoning,
                guest_requirements=json.dumps(analysis.guest_requirements),
                guest_questions=json.dumps(analysis.guest_questions),
                questions_answered=analysis.questions_answered,
                unanswered_questions=json.dumps(analysis.unanswered_questions),
                team_mistakes=json.dumps([m.model_dump() for m in analysis.team_mistakes]),
                team_strengths=json.dumps(analysis.team_strengths),
                response_quality_score=analysis.response_quality_score,
                conversion_likelihood=analysis.conversion_likelihood,
                lost_revenue_estimate=analysis.lost_revenue_estimate,
                recommendations=json.dumps(analysis.recommendations),
                training_opportunities=json.dumps(analysis.training_opportunities),
                analyzed_at=datetime.utcnow()
            )
            db.add(new_analysis)
            db.commit()
            db.refresh(new_analysis)
            print(f"[InquiryAnalysis] Created: {thread.guest_name} - {analysis.outcome}")
            return new_analysis
            
    except Exception as e:
        print(f"[InquiryAnalysis] Error analyzing thread {thread.thread_id}: {e}")
        import traceback
        traceback.print_exc()
        db.rollback()
        return None


async def refresh_inquiry_analysis(db: Session, days_back: int = 30, limit: int = 50) -> Dict[str, Any]:
    """
    Analyze recent inquiries that didn't convert to bookings.
    
    Args:
        db: Database session
        days_back: How many days of inquiries to analyze
        limit: Max number of inquiries to analyze
    
    Returns:
        Summary of analysis
    """
    # Get monitored properties
    monitored = get_monitored_properties(db)
    
    if not monitored:
        return {
            "status": "no_properties",
            "message": "No properties are configured for monitoring.",
            "inquiries_analyzed": 0
        }
    
    listing_ids = [p["listing_id"] for p in monitored]
    
    # Get confirmed reservation IDs to exclude
    now = datetime.utcnow()
    cutoff_date = now - timedelta(days=days_back)
    
    confirmed_reservation_ids = [
        g.reservation_id for g in db.query(GuestIndex.reservation_id).filter(
            GuestIndex.listing_id.in_(listing_ids),
            GuestIndex.reservation_id.isnot(None)
        ).all()
    ]
    
    # Find threads that are NOT confirmed reservations (inquiries)
    # Look for threads from the past N days at monitored properties
    inquiry_threads = db.query(HostifyThread).filter(
        HostifyThread.listing_id.in_(listing_ids),
        HostifyThread.checkin >= cutoff_date,  # Inquired for dates after cutoff
        ~HostifyThread.reservation_id.in_(confirmed_reservation_ids) if confirmed_reservation_ids else True
    ).order_by(HostifyThread.last_message_at.desc()).limit(limit).all()
    
    print(f"[InquiryAnalysis] Found {len(inquiry_threads)} inquiry threads to analyze")
    
    analyzed = 0
    errors = 0
    
    for i, thread in enumerate(inquiry_threads):
        print(f"[InquiryAnalysis] [{i+1}/{len(inquiry_threads)}] Analyzing {thread.guest_name}...")
        result = await analyze_and_save_inquiry(db, thread)
        if result:
            analyzed += 1
        else:
            errors += 1
        await asyncio.sleep(0.3)  # Rate limit
    
    return {
        "status": "complete",
        "properties_monitored": len(monitored),
        "inquiries_found": len(inquiry_threads),
        "inquiries_analyzed": analyzed,
        "errors": errors
    }


def get_inquiry_analyses(db: Session, listing_id: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
    """
    Get inquiry analyses, optionally filtered by listing.
    """
    query = db.query(InquiryAnalysis)
    
    if listing_id:
        query = query.filter(InquiryAnalysis.listing_id == listing_id)
    
    analyses = query.order_by(InquiryAnalysis.analyzed_at.desc()).limit(limit).all()
    
    result = []
    for a in analyses:
        result.append({
            "id": a.id,
            "thread_id": a.thread_id,
            "guest_name": a.guest_name,
            "guest_email": a.guest_email,
            "listing_id": a.listing_id,
            "listing_name": a.listing_name,
            "inquiry_date": a.inquiry_date.isoformat() if a.inquiry_date else None,
            "requested_checkin": a.requested_checkin.isoformat() if a.requested_checkin else None,
            "requested_checkout": a.requested_checkout.isoformat() if a.requested_checkout else None,
            "first_response_minutes": a.first_response_minutes,
            "total_messages": a.total_messages,
            "team_messages": a.team_messages,
            "guest_messages": a.guest_messages,
            "conversation_duration_hours": a.conversation_duration_hours,
            "outcome": a.outcome,
            "outcome_reasoning": a.outcome_reasoning,
            "guest_requirements": json.loads(a.guest_requirements) if a.guest_requirements else [],
            "guest_questions": json.loads(a.guest_questions) if a.guest_questions else [],
            "questions_answered": a.questions_answered,
            "unanswered_questions": json.loads(a.unanswered_questions) if a.unanswered_questions else [],
            "team_mistakes": json.loads(a.team_mistakes) if a.team_mistakes else [],
            "team_strengths": json.loads(a.team_strengths) if a.team_strengths else [],
            "response_quality_score": a.response_quality_score,
            "conversion_likelihood": a.conversion_likelihood,
            "lost_revenue_estimate": a.lost_revenue_estimate,
            "recommendations": json.loads(a.recommendations) if a.recommendations else [],
            "training_opportunities": json.loads(a.training_opportunities) if a.training_opportunities else [],
            "analyzed_at": a.analyzed_at.isoformat() if a.analyzed_at else None
        })
    
    # Sort by lost revenue (highest first), then by conversion likelihood
    result.sort(key=lambda x: (
        -(x["lost_revenue_estimate"] or 0),
        -(x["conversion_likelihood"] or 0)
    ))
    
    return result


def get_inquiry_summary(db: Session) -> Dict[str, Any]:
    """
    Get summary statistics for inquiry analysis.
    """
    analyses = db.query(InquiryAnalysis).all()
    
    if not analyses:
        return {
            "total_inquiries": 0,
            "total_lost_revenue": 0,
            "avg_response_time": 0,
            "avg_conversion_likelihood": 0,
            "outcomes": {},
            "common_mistakes": [],
            "top_training_needs": []
        }
    
    # Calculate stats
    total_lost_revenue = sum(a.lost_revenue_estimate or 0 for a in analyses)
    response_times = [a.first_response_minutes for a in analyses if a.first_response_minutes is not None]
    avg_response_time = sum(response_times) / len(response_times) if response_times else 0
    
    likelihoods = [a.conversion_likelihood for a in analyses if a.conversion_likelihood is not None]
    avg_conversion_likelihood = sum(likelihoods) / len(likelihoods) if likelihoods else 0
    
    # Count outcomes
    outcomes = {}
    for a in analyses:
        outcome = a.outcome or "unknown"
        outcomes[outcome] = outcomes.get(outcome, 0) + 1
    
    # Aggregate common mistakes
    mistake_counts = {}
    for a in analyses:
        if a.team_mistakes:
            try:
                mistakes = json.loads(a.team_mistakes)
                for m in mistakes:
                    mistake = m.get("mistake", "")
                    if mistake:
                        mistake_counts[mistake] = mistake_counts.get(mistake, 0) + 1
            except:
                pass
    
    common_mistakes = sorted(mistake_counts.items(), key=lambda x: -x[1])[:10]
    
    # Aggregate training needs
    training_counts = {}
    for a in analyses:
        if a.training_opportunities:
            try:
                opportunities = json.loads(a.training_opportunities)
                for t in opportunities:
                    training_counts[t] = training_counts.get(t, 0) + 1
            except:
                pass
    
    top_training_needs = sorted(training_counts.items(), key=lambda x: -x[1])[:10]
    
    return {
        "total_inquiries": len(analyses),
        "total_lost_revenue": total_lost_revenue,
        "avg_response_time_minutes": round(avg_response_time, 1),
        "avg_conversion_likelihood": round(avg_conversion_likelihood, 2),
        "outcomes": outcomes,
        "common_mistakes": [{"mistake": m, "count": c} for m, c in common_mistakes],
        "top_training_needs": [{"topic": t, "count": c} for t, c in top_training_needs]
    }
