"""
AI Brain - Advanced Response Generation with RAG and Agents.

This module provides:
1. Intent classification for understanding guest requests
2. RAG (Retrieval-Augmented Generation) for context-aware responses
3. Specialized handlers for common intents
4. Learning from human feedback
5. Guest profile-aware personalization

The system uses:
- ChromaDB for vector storage and semantic search
- OpenAI GPT-4o for response generation
- OpenAI text-embedding-3-small for embeddings
"""

import json
from typing import List, Optional, Dict, Any
from pydantic import BaseModel

from openai import OpenAI

from config import settings
from models import Message, GuestIndex, MessageDirection


# Initialize OpenAI client
client = OpenAI(api_key=settings.OPENAI_API_KEY)


class AIResponse(BaseModel):
    """Structured response from the AI."""
    reply_text: str
    confidence_score: float
    requires_human: bool
    reasoning: str
    escalation_reason: Optional[str] = None
    # Enhanced fields
    intent: Optional[str] = None
    handler_used: Optional[str] = None
    context_sources: List[str] = []


def get_style_examples(guest_message: str, n: int = 3) -> List[dict]:
    """
    Return n most relevant style examples for the current message.
    
    Uses SEMANTIC SEARCH via embeddings for better matching.
    Falls back to keyword matching if vector DB not available.
    """
    # Try semantic search first
    try:
        from embeddings import search_style_examples
        results = search_style_examples(guest_message, top_k=n)
        if results:
            return [
                {
                    "guest_message": r["guest_message"],
                    "your_reply": r["your_reply"],
                    "category": r.get("category", "general"),
                    "score": r.get("score", 0)
                }
                for r in results
            ]
    except Exception as e:
        print(f"[Brain] Semantic style search failed, falling back to keyword: {e}")
    
    # Fallback to keyword matching
    try:
        with open("style_guide.json", "r") as f:
            all_examples = json.load(f)
    except FileNotFoundError:
        return []
    
    if not all_examples:
        return []
    
    message_lower = guest_message.lower()
    
    # Score examples by keyword relevance
    scored_examples = []
    for example in all_examples:
        score = 0
        category = example.get("category", "").lower()
        guest_msg = example.get("guest_message", "").lower()
        tags = example.get("tags", [])
        
        # Category keyword matching
        if category in message_lower:
            score += 3
        
        # Word overlap
        message_words = set(message_lower.split())
        example_words = set(guest_msg.split())
        overlap = len(message_words & example_words)
        score += overlap
        
        # Tag matching for common topics
        wifi_keywords = ["wifi", "internet", "password", "network"]
        checkin_keywords = ["check", "arrive", "early", "time"]
        issue_keywords = ["not working", "broken", "problem", "help"]
        
        if any(kw in message_lower for kw in wifi_keywords) and "wifi" in category:
            score += 5
        if any(kw in message_lower for kw in checkin_keywords) and "check" in category:
            score += 5
        if any(kw in message_lower for kw in issue_keywords) and "issue" in category:
            score += 5
        
        scored_examples.append((score, example))
    
    # Sort by score and return top n
    scored_examples.sort(key=lambda x: x[0], reverse=True)
    return [ex for score, ex in scored_examples[:n]]


async def generate_unknown_guest_response(
    messages: List[Message],
    available_properties: List[str]
) -> AIResponse:
    """
    Generate a response for an unknown guest asking for property/dates.
    
    Args:
        messages: Recent conversation messages
        available_properties: List of property names to help identify
        
    Returns:
        AIResponse asking guest to identify themselves
    """
    # Build conversation history
    history = "CONVERSATION HISTORY:\n"
    for msg in messages[-5:]:
        role = "Guest" if msg.direction == MessageDirection.inbound else "You"
        history += f"{role}: {msg.content}\n"
    
    # Create a property hint if we have properties
    property_hint = ""
    if available_properties:
        sample = available_properties[:5]
        property_hint = f"\nSome of our properties: {', '.join(sample)}"
    
    system_prompt = f"""You are a friendly property manager assistant. A guest has messaged you but we don't know which reservation they belong to.

Your task is to politely ask them to identify themselves so you can help them properly.

RULES:
1. Be warm and apologetic that you don't recognize their number
2. Ask them which property they're staying at (or have a reservation for)
3. Ask for their check-in date or name on the reservation
4. Keep it brief and friendly
5. Don't make up any property details
{property_hint}

OUTPUT FORMAT (strict JSON):
{{
    "reply_text": "Your friendly response asking them to identify themselves",
    "confidence_score": 0.85,
    "requires_human": false,
    "reasoning": "Asking unknown guest to identify their reservation",
    "escalation_reason": null
}}"""
    
    try:
        response = client.chat.completions.create(
            model=settings.OPENAI_MODEL,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"{history}\n\nGenerate your response:"}
            ],
            temperature=0.7,
            max_tokens=300
        )
        
        result = json.loads(response.choices[0].message.content)
        return AIResponse(**result)
        
    except Exception as e:
        # Fallback response if AI fails
        return AIResponse(
            reply_text="Hi there! I don't seem to have your number on file. Could you please let me know which property you're staying at and your check-in date? I'll be happy to help once I can look up your reservation!",
            confidence_score=0.80,
            requires_human=False,
            reasoning="Fallback response for unknown guest",
            escalation_reason=None
        )


async def generate_ai_response(
    messages: List[Message],
    guest_context: Optional[GuestIndex],
    style_examples: List[dict],
    conversation_id: int = None,
    use_advanced: bool = True
) -> AIResponse:
    """
    Generate an AI response with confidence scoring.
    
    This is the main entry point for response generation. It can use either:
    1. Advanced mode (default): Full RAG + agents pipeline with intent classification,
       semantic search, property knowledge, guest profiles, and learning from corrections
    2. Simple mode: Basic GPT-4o with conversation history and style examples
    
    Args:
        messages: Recent conversation messages
        guest_context: Guest and reservation information (can be None for unknown guests)
        style_examples: Few-shot examples of good replies (used in simple mode)
        conversation_id: The conversation ID (needed for advanced mode)
        use_advanced: Whether to use the advanced RAG pipeline (default True)
        
    Returns:
        AIResponse with reply text, confidence score, and metadata
    """
    
    # Use advanced RAG + agents pipeline if enabled
    if use_advanced and conversation_id is not None:
        try:
            return await generate_advanced_response(
                messages=messages,
                guest_context=guest_context,
                conversation_id=conversation_id
            )
        except Exception as e:
            print(f"[Brain] Advanced pipeline failed, falling back to simple: {e}")
            # Fall through to simple mode
    
    # Simple mode - original implementation
    return await generate_simple_response(messages, guest_context, style_examples)


async def generate_advanced_response(
    messages: List[Message],
    guest_context: Optional[GuestIndex],
    conversation_id: int
) -> AIResponse:
    """
    Generate a response using the full RAG + agents pipeline.
    
    This includes:
    1. Intent classification
    2. Retrieval from property knowledge base
    3. Semantic search for similar past conversations
    4. Guest profile personalization
    5. Specialized handlers for common intents
    6. Learning from past corrections
    
    Args:
        messages: Recent conversation messages
        guest_context: Guest and reservation information
        conversation_id: The conversation ID
        
    Returns:
        AIResponse with full context about how the response was generated
    """
    from agents import route_and_respond
    from knowledge import guest_profile_manager, conversation_summarizer
    
    # Get the latest guest message
    last_message = ""
    for msg in reversed(messages):
        if msg.direction == MessageDirection.inbound:
            last_message = msg.content
            break
    
    if not last_message:
        return AIResponse(
            reply_text="",
            confidence_score=0.0,
            requires_human=True,
            reasoning="No guest message found",
            escalation_reason="No guest message to respond to"
        )
    
    # Update guest profile with conversation data
    if guest_context:
        guest_profile_manager.update_from_conversation(
            guest_context.guest_phone,
            messages,
            guest_context
        )
        # Analyze sentiment of the latest message
        guest_profile_manager.analyze_sentiment(guest_context.guest_phone, last_message)
    
    # Check if we need to summarize a long conversation
    if conversation_summarizer.should_summarize(conversation_id):
        conversation_summarizer.generate_summary(conversation_id)
    
    # Route through the agent system
    agent_response = await route_and_respond(
        message=last_message,
        messages=messages,
        guest_context=guest_context,
        conversation_id=conversation_id
    )
    
    # Convert to AIResponse
    return AIResponse(
        reply_text=agent_response.reply_text,
        confidence_score=agent_response.confidence_score,
        requires_human=agent_response.requires_human,
        reasoning=agent_response.reasoning,
        escalation_reason=agent_response.escalation_reason,
        intent=agent_response.intent,
        handler_used=agent_response.handler_used,
        context_sources=agent_response.context_sources
    )


async def generate_simple_response(
    messages: List[Message],
    guest_context: Optional[GuestIndex],
    style_examples: List[dict]
) -> AIResponse:
    """
    Generate a response using the simple (original) approach.
    Uses just conversation history and style examples with GPT-4o.
    
    This is the fallback when the advanced pipeline is disabled or fails.
    """
    
    # Build context string with guest information
    if guest_context:
        context = f"""
GUEST INFORMATION:
- Name: {guest_context.guest_name}
- Check-in: {guest_context.check_in_date.strftime('%B %d, %Y at 4:00 PM') if guest_context.check_in_date else 'Unknown'}
- Check-out: {guest_context.check_out_date.strftime('%B %d, %Y at 11:00 AM') if guest_context.check_out_date else 'Unknown'}
- Property: {guest_context.listing_name}
- Address: {guest_context.listing_address}
- Door Code: {guest_context.door_code or 'Not set'}
- WiFi Network: {guest_context.wifi_name or 'Not set'}
- WiFi Password: {guest_context.wifi_password or 'Not set'}
- Special Instructions: {guest_context.special_instructions or 'None'}
"""
    else:
        context = """
GUEST INFORMATION:
- This is an UNKNOWN guest - we don't have their reservation on file
- Be helpful but explain that you need to identify their reservation first
- Ask for their property name, check-in date, and/or name on reservation
"""
    
    # Build style examples section
    style_section = ""
    if style_examples:
        style_section = "EXAMPLE REPLIES (match this tone and style):\n"
        for ex in style_examples:
            style_section += f"Guest: \"{ex.get('guest_message', '')}\"\n"
            style_section += f"You replied: \"{ex.get('your_reply', '')}\"\n\n"
    
    # Build conversation history
    history = "CONVERSATION HISTORY:\n"
    for msg in messages[-10:]:  # Last 10 messages
        role = "Guest" if msg.direction == MessageDirection.inbound else "You"
        history += f"{role}: {msg.content}\n"
    
    # System prompt
    system_prompt = f"""You are an expert property manager assistant. Your job is to help guests with their stay.

{context}

{style_section}

RULES:
1. Be warm, friendly, and helpful
2. Only provide information you have in the context above
3. If you don't have the information or are unsure, set requires_human to true
4. For complaints, emergencies, or refund requests, ALWAYS set requires_human to true
5. Never make up information (door codes, wifi passwords, policies)
6. Keep responses concise but friendly (2-4 sentences typically)
7. Use the guest's name occasionally for a personal touch

CONFIDENCE SCORING GUIDELINES:
- 0.90-1.00: Simple, factual questions with clear answers in context (wifi password, check-in time)
- 0.80-0.89: Standard questions with good context available
- 0.70-0.79: Questions requiring some interpretation or partial context
- 0.50-0.69: Uncertain situations, missing info, or complex requests
- Below 0.50: Escalate - complaints, emergencies, refunds, no context

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
                {"role": "user", "content": f"{history}\n\nGenerate your response:"}
            ],
            temperature=0.7,
            max_tokens=500
        )
        
        result = json.loads(response.choices[0].message.content)
        return AIResponse(**result)
        
    except Exception as e:
        # On error, return a safe escalation response
        return AIResponse(
            reply_text="",
            confidence_score=0.0,
            requires_human=True,
            reasoning=f"AI generation error: {str(e)}",
            escalation_reason=f"AI generation failed: {str(e)}"
        )
