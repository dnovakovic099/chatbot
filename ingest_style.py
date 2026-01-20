"""
Style Guide Generator.

Ingests past conversation data and generates a style guide for AI few-shot prompting.

Usage:
    python ingest_style.py --input conversations.json --output style_guide.json
"""

import json
import argparse
from typing import List, Dict, Optional
from collections import defaultdict


def categorize_message(message: str) -> str:
    """
    Categorize a message based on its content.
    Returns a category string for grouping similar exchanges.
    """
    message_lower = message.lower()
    
    # WiFi related
    if any(kw in message_lower for kw in ["wifi", "internet", "password", "network", "connect"]):
        return "wifi"
    
    # Check-in related
    if any(kw in message_lower for kw in ["check in", "check-in", "checkin", "arrive", "arrival", "early"]):
        return "check_in"
    
    # Check-out related
    if any(kw in message_lower for kw in ["check out", "check-out", "checkout", "leaving", "departure", "late"]):
        return "check_out"
    
    # Door/access related
    if any(kw in message_lower for kw in ["door", "code", "key", "lock", "access", "enter"]):
        return "access"
    
    # Parking related
    if any(kw in message_lower for kw in ["park", "parking", "car", "garage", "driveway"]):
        return "parking"
    
    # Amenities
    if any(kw in message_lower for kw in ["pool", "hot tub", "gym", "amenities", "laundry"]):
        return "amenities"
    
    # Issues/problems
    if any(kw in message_lower for kw in ["not working", "broken", "doesn't work", "problem", "issue", "help"]):
        return "issue"
    
    # Directions/location
    if any(kw in message_lower for kw in ["where", "address", "directions", "location", "find"]):
        return "directions"
    
    # Thanks/appreciation
    if any(kw in message_lower for kw in ["thank", "thanks", "appreciate", "great stay"]):
        return "thanks"
    
    # General inquiry
    return "general"


def extract_tags(reply: str) -> List[str]:
    """Extract descriptive tags from a reply."""
    tags = []
    reply_lower = reply.lower()
    
    # Tone tags
    if any(word in reply_lower for word in ["sorry", "apologize", "apologies"]):
        tags.append("empathetic")
    if "!" in reply and len(reply) < 200:
        tags.append("friendly")
    if any(word in reply_lower for word in ["please", "could you", "would you"]):
        tags.append("polite")
    
    # Content tags
    if any(word in reply_lower for word in ["first", "then", "next", "step"]):
        tags.append("instructional")
    if "?" in reply:
        tags.append("clarifying")
    if len(reply) < 100:
        tags.append("quick_info")
    if any(word in reply_lower for word in ["let me know", "reach out", "contact"]):
        tags.append("supportive")
    if any(word in reply_lower for word in ["check", "verify", "make sure"]):
        tags.append("troubleshoot")
    
    return tags or ["general"]


def score_reply_quality(
    guest_message: str,
    your_reply: str,
    conversation_length: int = 0,
    resolution_speed: Optional[int] = None
) -> float:
    """
    Score a reply's quality based on heuristics.
    
    Returns a score from 0-1 where higher is better.
    """
    score = 0.5  # Base score
    
    # Penalize very short or very long replies
    reply_len = len(your_reply)
    if 50 <= reply_len <= 300:
        score += 0.2
    elif reply_len < 30 or reply_len > 500:
        score -= 0.1
    
    # Bonus for quick resolution (few messages)
    if conversation_length and conversation_length <= 3:
        score += 0.2
    
    # Bonus for containing relevant info
    guest_lower = guest_message.lower()
    reply_lower = your_reply.lower()
    
    # If guest asks about wifi and reply contains wifi-related words
    if "wifi" in guest_lower and any(w in reply_lower for w in ["network", "password", "connect"]):
        score += 0.1
    
    # If guest asks about check-in and reply contains time/instructions
    if "check" in guest_lower and any(w in reply_lower for w in ["pm", "am", "time", "code"]):
        score += 0.1
    
    # Penalize generic responses
    generic_phrases = ["let me check", "i'll get back to you", "one moment"]
    if any(phrase in reply_lower for phrase in generic_phrases):
        score -= 0.2
    
    return max(0.0, min(1.0, score))


def process_conversations(conversations: List[Dict]) -> List[Dict]:
    """
    Process raw conversation data into style guide examples.
    
    Expected input format (flexible):
    - List of conversation objects
    - Each conversation has messages array
    - Messages have: content, direction/type, timestamp
    """
    examples = []
    
    for conv in conversations:
        messages = conv.get("messages", [])
        if len(messages) < 2:
            continue
        
        # Find guest message -> your reply pairs
        for i in range(len(messages) - 1):
            msg = messages[i]
            next_msg = messages[i + 1]
            
            # Check if this is a guest message followed by your reply
            msg_direction = msg.get("direction", msg.get("type", ""))
            next_direction = next_msg.get("direction", next_msg.get("type", ""))
            
            is_guest_msg = msg_direction in ["inbound", "received", "guest"]
            is_your_reply = next_direction in ["outbound", "sent", "host", "you"]
            
            if not (is_guest_msg and is_your_reply):
                continue
            
            guest_message = msg.get("content", msg.get("body", msg.get("text", "")))
            your_reply = next_msg.get("content", next_msg.get("body", next_msg.get("text", "")))
            
            if not guest_message or not your_reply:
                continue
            
            # Filter by length
            if len(your_reply) < 50 or len(your_reply) > 300:
                continue
            
            # Score the reply
            quality_score = score_reply_quality(
                guest_message,
                your_reply,
                conversation_length=len(messages)
            )
            
            # Only keep good examples
            if quality_score < 0.5:
                continue
            
            example = {
                "category": categorize_message(guest_message),
                "guest_message": guest_message.strip(),
                "your_reply": your_reply.strip(),
                "tags": extract_tags(your_reply),
                "quality_score": quality_score
            }
            examples.append(example)
    
    return examples


def deduplicate_and_select(examples: List[Dict], max_per_category: int = 5) -> List[Dict]:
    """
    Deduplicate examples and select the best ones per category.
    """
    # Group by category
    by_category = defaultdict(list)
    for ex in examples:
        by_category[ex["category"]].append(ex)
    
    # Sort each category by quality score and take top N
    final_examples = []
    for category, cat_examples in by_category.items():
        # Sort by quality score descending
        sorted_examples = sorted(cat_examples, key=lambda x: x["quality_score"], reverse=True)
        
        # Take top N, removing quality_score from output
        for ex in sorted_examples[:max_per_category]:
            final_ex = {
                "category": ex["category"],
                "guest_message": ex["guest_message"],
                "your_reply": ex["your_reply"],
                "tags": ex["tags"]
            }
            final_examples.append(final_ex)
    
    return final_examples


def main():
    parser = argparse.ArgumentParser(description="Generate style guide from conversation history")
    parser.add_argument("--input", "-i", required=True, help="Input JSON file with conversations")
    parser.add_argument("--output", "-o", default="style_guide.json", help="Output style guide JSON file")
    parser.add_argument("--max-per-category", type=int, default=5, help="Max examples per category")
    
    args = parser.parse_args()
    
    # Load conversations
    print(f"Loading conversations from {args.input}...")
    with open(args.input, "r") as f:
        data = json.load(f)
    
    # Handle different input formats
    if isinstance(data, list):
        conversations = data
    elif isinstance(data, dict):
        conversations = data.get("conversations", data.get("data", [data]))
    else:
        print("Error: Unexpected data format")
        return
    
    print(f"Found {len(conversations)} conversations")
    
    # Process into examples
    examples = process_conversations(conversations)
    print(f"Extracted {len(examples)} candidate examples")
    
    # Deduplicate and select best
    final_examples = deduplicate_and_select(examples, args.max_per_category)
    print(f"Selected {len(final_examples)} final examples")
    
    # Show category distribution
    categories = defaultdict(int)
    for ex in final_examples:
        categories[ex["category"]] += 1
    
    print("\nCategory distribution:")
    for cat, count in sorted(categories.items()):
        print(f"  {cat}: {count}")
    
    # Save output
    with open(args.output, "w") as f:
        json.dump(final_examples, f, indent=2)
    
    print(f"\nSaved style guide to {args.output}")


if __name__ == "__main__":
    main()
