"""
Agent System for Property Manager.
Provides intent classification, multi-step reasoning, and specialized handlers.
"""

import json
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

from openai import OpenAI
from pydantic import BaseModel

from config import settings
from models import Message, GuestIndex, MessageDirection


# Initialize OpenAI client
client = OpenAI(api_key=settings.OPENAI_API_KEY)


# ============ INTENT CLASSIFICATION ============

class Intent(str, Enum):
    """Possible guest intents."""
    # Information requests
    WIFI_INFO = "wifi_info"
    DOOR_CODE = "door_code"
    CHECK_IN_TIME = "check_in_time"
    CHECK_OUT_TIME = "check_out_time"
    PARKING_INFO = "parking_info"
    AMENITIES_INFO = "amenities_info"
    DIRECTIONS = "directions"
    LOCAL_RECOMMENDATIONS = "local_recommendations"
    HOUSE_RULES = "house_rules"
    
    # Requests
    EARLY_CHECK_IN = "early_check_in"
    LATE_CHECK_OUT = "late_check_out"
    EXTEND_STAY = "extend_stay"
    
    # Issues
    ISSUE_WIFI = "issue_wifi"
    ISSUE_HVAC = "issue_hvac"
    ISSUE_APPLIANCE = "issue_appliance"
    ISSUE_ACCESS = "issue_access"
    ISSUE_CLEANLINESS = "issue_cleanliness"
    ISSUE_NOISE = "issue_noise"
    ISSUE_OTHER = "issue_other"
    
    # Complaints / Escalations
    COMPLAINT = "complaint"
    REFUND_REQUEST = "refund_request"
    EMERGENCY = "emergency"
    
    # Social
    GREETING = "greeting"
    THANKS = "thanks"
    GOODBYE = "goodbye"
    
    # Other
    GENERAL_QUESTION = "general_question"
    UNKNOWN = "unknown"


class IntentClassification(BaseModel):
    """Result of intent classification."""
    primary_intent: str
    secondary_intents: List[str] = []
    confidence: float
    requires_action: bool
    is_urgent: bool
    entities: Dict[str, Any] = {}  # Extracted entities like dates, times, etc.


def classify_intent(message: str, conversation_context: str = "") -> IntentClassification:
    """
    Classify the intent of a guest message.
    
    Args:
        message: The guest's message
        conversation_context: Recent conversation for context
        
    Returns:
        IntentClassification with primary and secondary intents
    """
    
    intent_list = [intent.value for intent in Intent]
    
    system_prompt = f"""You are an intent classifier for a property management system.
Analyze the guest message and classify the intent(s).

AVAILABLE INTENTS:
{json.dumps(intent_list, indent=2)}

RULES:
1. Identify the PRIMARY intent (most important)
2. List any SECONDARY intents if the message has multiple requests
3. Set is_urgent=true for emergencies, safety issues, or angry complaints
4. Set requires_action=true if the guest needs a response or action
5. Extract relevant entities (dates, times, appliances mentioned, etc.)

OUTPUT FORMAT (strict JSON):
{{
    "primary_intent": "intent_name",
    "secondary_intents": ["intent2", "intent3"],
    "confidence": 0.0-1.0,
    "requires_action": true/false,
    "is_urgent": true/false,
    "entities": {{"key": "value"}}
}}"""

    user_prompt = message
    if conversation_context:
        user_prompt = f"CONVERSATION CONTEXT:\n{conversation_context}\n\nNEW MESSAGE:\n{message}"

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Use mini for fast classification
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
            max_tokens=300
        )
        
        result = json.loads(response.choices[0].message.content)
        return IntentClassification(**result)
        
    except Exception as e:
        # Fallback classification
        return IntentClassification(
            primary_intent=Intent.UNKNOWN.value,
            secondary_intents=[],
            confidence=0.0,
            requires_action=True,
            is_urgent=False,
            entities={}
        )


# ============ CONTEXT RETRIEVAL ============

@dataclass
class RetrievedContext:
    """Context retrieved from various sources for response generation."""
    # Property knowledge
    property_docs: List[Dict[str, Any]] = field(default_factory=list)
    
    # Similar past conversations
    similar_conversations: List[Dict[str, Any]] = field(default_factory=list)
    
    # Style examples
    style_examples: List[Dict[str, Any]] = field(default_factory=list)
    
    # Corrections to learn from
    relevant_corrections: List[Dict[str, Any]] = field(default_factory=list)
    
    # Conversation summary (if long conversation)
    conversation_summary: Optional[str] = None
    
    # Guest profile
    guest_profile: Optional[Dict[str, Any]] = None
    
    # Live data (from APIs)
    live_data: Dict[str, Any] = field(default_factory=dict)


def retrieve_context(
    message: str,
    intent: IntentClassification,
    guest_context: Optional[GuestIndex],
    messages: List[Message]
) -> RetrievedContext:
    """
    Retrieve all relevant context for response generation.
    
    This is the main RAG function that gathers context from multiple sources.
    """
    from embeddings import (
        search_property_knowledge,
        search_similar_conversations,
        search_style_examples,
        search_corrections
    )
    
    context = RetrievedContext()
    property_id = guest_context.listing_id if guest_context else None
    
    # 1. Search property knowledge base
    if property_id:
        # Determine which doc types to search based on intent
        doc_types = _get_doc_types_for_intent(intent.primary_intent)
        context.property_docs = search_property_knowledge(
            query=message,
            property_id=property_id,
            doc_types=doc_types,
            top_k=5
        )
    
    # 2. Search similar past conversations
    context.similar_conversations = search_similar_conversations(
        query=message,
        property_id=property_id,
        intent=intent.primary_intent,
        successful_only=True,
        top_k=3
    )
    
    # 3. Get semantically similar style examples
    context.style_examples = search_style_examples(
        query=message,
        top_k=3
    )
    
    # 4. Search for relevant corrections (to avoid past mistakes)
    context.relevant_corrections = search_corrections(
        query=message,
        property_id=property_id,
        top_k=2
    )
    
    # 5. Generate conversation summary if needed
    if len(messages) > 15:
        context.conversation_summary = _summarize_conversation(messages)
    
    # 6. Load guest profile if available
    if guest_context:
        context.guest_profile = _load_guest_profile(guest_context.guest_phone)
    
    return context


def _get_doc_types_for_intent(intent: str) -> Optional[List[str]]:
    """Map intent to relevant document types."""
    intent_to_docs = {
        Intent.WIFI_INFO.value: ["faq", "appliance_guide"],
        Intent.ISSUE_WIFI.value: ["faq", "appliance_guide", "known_issues"],
        Intent.DOOR_CODE.value: ["check_in", "faq"],
        Intent.ISSUE_ACCESS.value: ["check_in", "faq", "known_issues"],
        Intent.CHECK_IN_TIME.value: ["check_in", "house_rules"],
        Intent.CHECK_OUT_TIME.value: ["check_out", "house_rules"],
        Intent.PARKING_INFO.value: ["faq", "house_rules"],
        Intent.AMENITIES_INFO.value: ["faq", "appliance_guide"],
        Intent.LOCAL_RECOMMENDATIONS.value: ["local_tips"],
        Intent.HOUSE_RULES.value: ["house_rules"],
        Intent.ISSUE_HVAC.value: ["appliance_guide", "known_issues"],
        Intent.ISSUE_APPLIANCE.value: ["appliance_guide", "known_issues"],
        Intent.ISSUE_CLEANLINESS.value: ["known_issues", "house_rules"],
    }
    return intent_to_docs.get(intent)


def _summarize_conversation(messages: List[Message]) -> str:
    """Generate a summary of a long conversation."""
    # Build message text
    message_text = ""
    for msg in messages[-30:]:  # Last 30 messages
        role = "Guest" if msg.direction == MessageDirection.inbound else "Host"
        message_text += f"{role}: {msg.content}\n"
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": """Summarize this conversation between a guest and property host.
Focus on:
1. Main topics discussed
2. Any issues raised and their resolution status
3. Any pending requests or questions
4. Guest's apparent mood/satisfaction

Keep the summary concise (3-5 sentences)."""
                },
                {"role": "user", "content": message_text}
            ],
            temperature=0.3,
            max_tokens=300
        )
        return response.choices[0].message.content
    except Exception:
        return None


def _load_guest_profile(guest_phone: str) -> Optional[Dict[str, Any]]:
    """Load guest profile from database."""
    from models import SessionLocal, GuestProfile
    
    db = SessionLocal()
    try:
        profile = db.query(GuestProfile).filter(
            GuestProfile.guest_phone == guest_phone
        ).first()
        
        if profile:
            return {
                "name": profile.guest_name,
                "total_stays": profile.total_stays,
                "communication_style": profile.communication_style,
                "overall_sentiment": profile.overall_sentiment,
                "preferences": json.loads(profile.preferences) if profile.preferences else {},
                "past_issues": json.loads(profile.past_issues) if profile.past_issues else [],
                "is_vip": profile.is_vip,
                "special_handling_notes": profile.special_handling_notes
            }
        return None
    finally:
        db.close()


# ============ SPECIALIZED HANDLERS ============

class HandlerResult(BaseModel):
    """Result from a specialized handler."""
    can_handle: bool
    response: Optional[str] = None
    confidence: float = 0.0
    needs_escalation: bool = False
    escalation_reason: Optional[str] = None
    additional_context: Dict[str, Any] = {}


class BaseHandler:
    """Base class for specialized intent handlers."""
    
    supported_intents: List[str] = []
    
    def can_handle(self, intent: IntentClassification) -> bool:
        """Check if this handler can handle the given intent."""
        return intent.primary_intent in self.supported_intents
    
    async def handle(
        self,
        message: str,
        intent: IntentClassification,
        guest_context: Optional[GuestIndex],
        retrieved_context: RetrievedContext
    ) -> HandlerResult:
        """Handle the message. Override in subclasses."""
        raise NotImplementedError


class WifiHandler(BaseHandler):
    """Handler for WiFi-related questions and issues."""
    
    supported_intents = [Intent.WIFI_INFO.value, Intent.ISSUE_WIFI.value]
    
    async def handle(
        self,
        message: str,
        intent: IntentClassification,
        guest_context: Optional[GuestIndex],
        retrieved_context: RetrievedContext
    ) -> HandlerResult:
        # Check if we have WiFi info
        if not guest_context or not guest_context.wifi_password:
            return HandlerResult(
                can_handle=True,
                needs_escalation=True,
                escalation_reason="WiFi information not available in system"
            )
        
        # For simple info request
        if intent.primary_intent == Intent.WIFI_INFO.value:
            response = f"The WiFi network is '{guest_context.wifi_name}' and the password is '{guest_context.wifi_password}'. Let me know if you have any trouble connecting!"
            return HandlerResult(
                can_handle=True,
                response=response,
                confidence=0.95
            )
        
        # For issues, include troubleshooting from knowledge base
        troubleshooting = ""
        for doc in retrieved_context.property_docs:
            if "wifi" in doc.get("metadata", {}).get("title", "").lower():
                troubleshooting = doc.get("content", "")
                break
        
        if troubleshooting:
            return HandlerResult(
                can_handle=True,
                additional_context={"troubleshooting_steps": troubleshooting},
                confidence=0.80
            )
        
        # Default troubleshooting
        return HandlerResult(
            can_handle=True,
            additional_context={
                "troubleshooting_steps": "Try: 1) Forget the network and reconnect, 2) Restart the router (unplug for 30 seconds), 3) Move closer to the router"
            },
            confidence=0.75
        )


class AccessHandler(BaseHandler):
    """Handler for access/door code questions and issues."""
    
    supported_intents = [Intent.DOOR_CODE.value, Intent.ISSUE_ACCESS.value]
    
    async def handle(
        self,
        message: str,
        intent: IntentClassification,
        guest_context: Optional[GuestIndex],
        retrieved_context: RetrievedContext
    ) -> HandlerResult:
        if not guest_context or not guest_context.door_code:
            return HandlerResult(
                can_handle=True,
                needs_escalation=True,
                escalation_reason="Door code not available in system"
            )
        
        if intent.primary_intent == Intent.DOOR_CODE.value:
            response = f"The door code is {guest_context.door_code}. Just enter it on the keypad to unlock the door."
            return HandlerResult(
                can_handle=True,
                response=response,
                confidence=0.95
            )
        
        # For access issues, escalate for safety
        return HandlerResult(
            can_handle=True,
            needs_escalation=True,
            escalation_reason="Guest reporting access issues - may need immediate assistance",
            additional_context={"door_code": guest_context.door_code}
        )


class CheckInOutHandler(BaseHandler):
    """Handler for check-in/check-out time questions."""
    
    supported_intents = [
        Intent.CHECK_IN_TIME.value, 
        Intent.CHECK_OUT_TIME.value,
        Intent.EARLY_CHECK_IN.value,
        Intent.LATE_CHECK_OUT.value
    ]
    
    async def handle(
        self,
        message: str,
        intent: IntentClassification,
        guest_context: Optional[GuestIndex],
        retrieved_context: RetrievedContext
    ) -> HandlerResult:
        
        if intent.primary_intent == Intent.CHECK_IN_TIME.value:
            check_in_time = "4:00 PM"  # Default
            if guest_context and guest_context.check_in_date:
                date_str = guest_context.check_in_date.strftime("%B %d")
                response = f"Check-in is at {check_in_time} on {date_str}. I'll send you the door code about an hour before!"
            else:
                response = f"Check-in is at {check_in_time}. I'll send you the door code about an hour before!"
            
            return HandlerResult(
                can_handle=True,
                response=response,
                confidence=0.90
            )
        
        if intent.primary_intent == Intent.CHECK_OUT_TIME.value:
            check_out_time = "11:00 AM"  # Default
            response = f"Check-out is at {check_out_time}. Just leave the keys on the counter and lock up behind you. Safe travels!"
            return HandlerResult(
                can_handle=True,
                response=response,
                confidence=0.90
            )
        
        # Early check-in and late check-out need human approval
        return HandlerResult(
            can_handle=True,
            needs_escalation=True,
            escalation_reason=f"Guest requesting {intent.primary_intent.replace('_', ' ')} - needs availability check"
        )


class EmergencyHandler(BaseHandler):
    """Handler for emergencies - always escalates."""
    
    supported_intents = [Intent.EMERGENCY.value, Intent.COMPLAINT.value, Intent.REFUND_REQUEST.value]
    
    async def handle(
        self,
        message: str,
        intent: IntentClassification,
        guest_context: Optional[GuestIndex],
        retrieved_context: RetrievedContext
    ) -> HandlerResult:
        return HandlerResult(
            can_handle=True,
            needs_escalation=True,
            escalation_reason=f"Urgent: {intent.primary_intent.replace('_', ' ').title()}"
        )


# Registry of all handlers
HANDLERS = [
    WifiHandler(),
    AccessHandler(),
    CheckInOutHandler(),
    EmergencyHandler(),
]


def get_handler_for_intent(intent: IntentClassification) -> Optional[BaseHandler]:
    """Find a handler that can handle the given intent."""
    for handler in HANDLERS:
        if handler.can_handle(intent):
            return handler
    return None


# ============ AGENT ROUTER ============

@dataclass
class AgentResponse:
    """Final response from the agent system."""
    reply_text: str
    confidence_score: float
    requires_human: bool
    reasoning: str
    escalation_reason: Optional[str] = None
    intent: Optional[str] = None
    handler_used: Optional[str] = None
    context_sources: List[str] = field(default_factory=list)


async def route_and_respond(
    message: str,
    messages: List[Message],
    guest_context: Optional[GuestIndex],
    conversation_id: int
) -> AgentResponse:
    """
    Main agent router that:
    1. Classifies intent
    2. Retrieves relevant context
    3. Routes to specialized handler or general LLM
    4. Generates response
    
    Args:
        message: The guest's message
        messages: Conversation history
        guest_context: Guest reservation info
        conversation_id: The conversation ID
        
    Returns:
        AgentResponse with the generated reply
    """
    
    # 1. Build conversation context for classification
    conv_context = ""
    for msg in messages[-5:]:
        role = "Guest" if msg.direction == MessageDirection.inbound else "Host"
        conv_context += f"{role}: {msg.content}\n"
    
    # 2. Classify intent
    intent = classify_intent(message, conv_context)
    
    # 3. Retrieve context
    retrieved_context = retrieve_context(message, intent, guest_context, messages)
    
    # 4. Track which context sources were used
    context_sources = []
    if retrieved_context.property_docs:
        context_sources.append("property_knowledge")
    if retrieved_context.similar_conversations:
        context_sources.append("similar_conversations")
    if retrieved_context.style_examples:
        context_sources.append("style_examples")
    if retrieved_context.relevant_corrections:
        context_sources.append("corrections")
    if retrieved_context.conversation_summary:
        context_sources.append("conversation_summary")
    if retrieved_context.guest_profile:
        context_sources.append("guest_profile")
    
    # 5. Try specialized handler first
    handler = get_handler_for_intent(intent)
    if handler:
        result = await handler.handle(message, intent, guest_context, retrieved_context)
        
        if result.needs_escalation:
            return AgentResponse(
                reply_text="",
                confidence_score=0.0,
                requires_human=True,
                reasoning=result.escalation_reason or "Handler requested escalation",
                escalation_reason=result.escalation_reason,
                intent=intent.primary_intent,
                handler_used=handler.__class__.__name__,
                context_sources=context_sources
            )
        
        if result.response and result.confidence >= 0.85:
            # Handler provided a complete response
            return AgentResponse(
                reply_text=result.response,
                confidence_score=result.confidence,
                requires_human=False,
                reasoning=f"Handled by {handler.__class__.__name__}",
                intent=intent.primary_intent,
                handler_used=handler.__class__.__name__,
                context_sources=context_sources
            )
    
    # 6. Fall back to general LLM with full RAG context
    return await generate_rag_response(
        message=message,
        messages=messages,
        guest_context=guest_context,
        intent=intent,
        retrieved_context=retrieved_context,
        context_sources=context_sources
    )


async def generate_rag_response(
    message: str,
    messages: List[Message],
    guest_context: Optional[GuestIndex],
    intent: IntentClassification,
    retrieved_context: RetrievedContext,
    context_sources: List[str]
) -> AgentResponse:
    """
    Generate a response using full RAG context.
    This is the fallback when specialized handlers can't fully handle the request.
    """
    
    # Build comprehensive prompt
    prompt_sections = []
    
    # 1. Guest information
    if guest_context:
        prompt_sections.append(f"""GUEST INFORMATION:
- Name: {guest_context.guest_name}
- Check-in: {guest_context.check_in_date.strftime('%B %d, %Y at 4:00 PM') if guest_context.check_in_date else 'Unknown'}
- Check-out: {guest_context.check_out_date.strftime('%B %d, %Y at 11:00 AM') if guest_context.check_out_date else 'Unknown'}
- Property: {guest_context.listing_name}
- Address: {guest_context.listing_address}
- Door Code: {guest_context.door_code or 'Not set'}
- WiFi Network: {guest_context.wifi_name or 'Not set'}
- WiFi Password: {guest_context.wifi_password or 'Not set'}
- Special Instructions: {guest_context.special_instructions or 'None'}""")
    else:
        prompt_sections.append("GUEST INFORMATION: Unknown guest - no reservation found.")
    
    # 2. Guest profile (if returning guest)
    if retrieved_context.guest_profile:
        profile = retrieved_context.guest_profile
        profile_text = f"""GUEST PROFILE (Returning Guest):
- Previous stays: {profile.get('total_stays', 0)}
- Communication style: {profile.get('communication_style', 'unknown')}
- Overall sentiment: {profile.get('overall_sentiment', 'neutral')}"""
        if profile.get('past_issues'):
            profile_text += f"\n- Past issues: {', '.join(profile['past_issues'][:3])}"
        if profile.get('is_vip'):
            profile_text += "\n- VIP GUEST - Prioritize exceptional service"
        prompt_sections.append(profile_text)
    
    # 3. Property knowledge (RAG results)
    if retrieved_context.property_docs:
        docs_text = "RELEVANT PROPERTY INFORMATION:\n"
        for doc in retrieved_context.property_docs[:3]:
            title = doc.get("metadata", {}).get("title", "Info")
            content = doc.get("content", "")[:500]
            docs_text += f"\n[{title}]\n{content}\n"
        prompt_sections.append(docs_text)
    
    # 4. Similar past conversations
    if retrieved_context.similar_conversations:
        similar_text = "SIMILAR PAST CONVERSATIONS (for reference):\n"
        for conv in retrieved_context.similar_conversations[:2]:
            similar_text += f"\nGuest asked: \"{conv['guest_message']}\"\n"
            similar_text += f"Good response: \"{conv['response']}\"\n"
        prompt_sections.append(similar_text)
    
    # 5. Style examples
    if retrieved_context.style_examples:
        style_text = "RESPONSE STYLE EXAMPLES:\n"
        for ex in retrieved_context.style_examples[:2]:
            style_text += f"\nGuest: \"{ex['guest_message']}\"\n"
            style_text += f"Reply: \"{ex['your_reply']}\"\n"
        prompt_sections.append(style_text)
    
    # 6. Corrections to learn from
    if retrieved_context.relevant_corrections:
        corr_text = "PAST CORRECTIONS (avoid these mistakes):\n"
        for corr in retrieved_context.relevant_corrections[:2]:
            corr_text += f"\n- AI said: \"{corr['original_ai_response'][:100]}...\"\n"
            corr_text += f"- Should be: \"{corr['corrected_response'][:100]}...\"\n"
        prompt_sections.append(corr_text)
    
    # 7. Conversation summary or history
    if retrieved_context.conversation_summary:
        prompt_sections.append(f"CONVERSATION SUMMARY:\n{retrieved_context.conversation_summary}")
    
    # Conversation history (last 10 messages)
    history = "RECENT CONVERSATION:\n"
    for msg in messages[-10:]:
        role = "Guest" if msg.direction == MessageDirection.inbound else "You"
        history += f"{role}: {msg.content}\n"
    prompt_sections.append(history)
    
    # 8. Intent information
    prompt_sections.append(f"""DETECTED INTENT:
- Primary: {intent.primary_intent}
- Secondary: {', '.join(intent.secondary_intents) if intent.secondary_intents else 'None'}
- Urgency: {'URGENT' if intent.is_urgent else 'Normal'}""")
    
    # Build system prompt
    system_prompt = f"""You are an expert property manager assistant with access to comprehensive context.

{chr(10).join(prompt_sections)}

RESPONSE RULES:
1. Use the provided context to give accurate, helpful responses
2. Match the tone and style of the examples provided
3. If you learned from past corrections, apply those lessons
4. Be warm, friendly, and concise (2-4 sentences typically)
5. Only provide information available in the context - never make things up
6. If information is missing or you're unsure, set requires_human to true
7. For complaints, emergencies, or refund requests, ALWAYS set requires_human to true

CONFIDENCE SCORING:
- 0.90-1.00: Clear answer available in context
- 0.80-0.89: Good context available, standard question
- 0.70-0.79: Some interpretation needed
- 0.50-0.69: Uncertain, limited context
- Below 0.50: Escalate to human

OUTPUT FORMAT (strict JSON):
{{
    "reply_text": "Your response to the guest",
    "confidence_score": 0.0 to 1.0,
    "requires_human": true or false,
    "reasoning": "Brief explanation of your confidence level",
    "escalation_reason": "If requires_human is true, explain why (null otherwise)"
}}"""

    try:
        response = client.chat.completions.create(
            model=settings.OPENAI_MODEL,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Current guest message: {message}\n\nGenerate your response:"}
            ],
            temperature=0.7,
            max_tokens=500
        )
        
        result = json.loads(response.choices[0].message.content)
        
        return AgentResponse(
            reply_text=result.get("reply_text", ""),
            confidence_score=result.get("confidence_score", 0.0),
            requires_human=result.get("requires_human", False),
            reasoning=result.get("reasoning", ""),
            escalation_reason=result.get("escalation_reason"),
            intent=intent.primary_intent,
            handler_used="RAG_LLM",
            context_sources=context_sources
        )
        
    except Exception as e:
        return AgentResponse(
            reply_text="",
            confidence_score=0.0,
            requires_human=True,
            reasoning=f"AI generation error: {str(e)}",
            escalation_reason=f"AI generation failed: {str(e)}",
            intent=intent.primary_intent,
            context_sources=context_sources
        )
