"""
RAG Rules Generator - Extracts knowledge and rules from past conversations.

This script:
1. Fetches messages from Hostify, one property at a time
2. Analyzes host responses to identify common Q&A patterns
3. Creates RAG rules for each property based on past responses
4. Tracks cross-property FAQs to create a master rule sheet
5. Saves everything to files for review (not DB)

Usage:
    python generate_rag_rules.py --properties 5
"""

import asyncio
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
from dataclasses import dataclass, asdict

from openai import OpenAI

from config import settings
from cache import HostifyClient
from models import SessionLocal, GuestIndex

# Output directory for generated rules
OUTPUT_DIR = Path("./generated_rag_rules")
OUTPUT_DIR.mkdir(exist_ok=True)


@dataclass
class QAPair:
    """A question-answer pair extracted from conversations."""
    guest_question: str
    host_answer: str
    timestamp: str
    thread_id: int
    guest_name: str = ""


@dataclass
class RAGRule:
    """A generated RAG rule/knowledge entry."""
    rule_id: str
    category: str  # wifi, check_in, amenities, parking, appliances, local_tips, policies, troubleshooting, other
    title: str
    trigger_questions: List[str]  # Questions that should trigger this rule
    response_template: str  # How to respond
    context_notes: str  # Additional context for the AI
    confidence: float  # How confident we are in this rule
    source_conversations: int  # How many conversations this was derived from
    property_specific: bool  # True if this is property-specific, False for global


@dataclass
class PropertyRules:
    """Collection of rules for a single property."""
    property_id: str
    property_name: str
    rules: List[RAGRule]
    total_conversations_analyzed: int
    total_messages_analyzed: int
    generated_at: str


@dataclass
class GlobalRules:
    """Cross-property rules (master sheet)."""
    rules: List[RAGRule]
    properties_analyzed: List[str]
    common_question_categories: Dict[str, int]  # category -> count
    generated_at: str


class RAGRulesGenerator:
    """Generates RAG rules from conversation history."""
    
    CATEGORIES = [
        "wifi",
        "check_in",
        "check_out", 
        "amenities",
        "parking",
        "appliances",
        "local_tips",
        "policies",
        "troubleshooting",
        "emergency",
        "booking",
        "other"
    ]
    
    def __init__(self):
        self.hostify = HostifyClient()
        self.openai = OpenAI(api_key=settings.OPENAI_API_KEY)
        self.all_qa_pairs: Dict[str, List[QAPair]] = defaultdict(list)  # property_id -> pairs
        self.global_qa_pairs: List[Tuple[str, QAPair]] = []  # (property_id, pair)
    
    async def fetch_property_messages(
        self, 
        property_id: str,
        property_name: str,
        max_threads: int = 100
    ) -> List[QAPair]:
        """
        Fetch all messages for a specific property and extract Q&A pairs.
        Note: Hostify API ignores listing_id filter, so we must filter client-side.
        """
        print(f"\n{'='*60}")
        print(f"📥 Fetching messages for: {property_name}")
        print(f"   Property ID: {property_id}")
        print(f"{'='*60}")
        
        qa_pairs = []
        
        # Fetch all threads and filter client-side (API ignores listing_id param)
        property_threads = []
        page = 1
        max_pages = 100  # Up to 2000 threads scanned
        total_scanned = 0
        
        import httpx
        async with httpx.AsyncClient(timeout=60.0) as client:
            while page <= max_pages and len(property_threads) < max_threads:
                try:
                    response = await client.get(
                        f"{self.hostify.base_url}/inbox",
                        headers=self.hostify.headers,
                        params={"limit": 100, "page": page}
                    )
                    response.raise_for_status()
                    data = response.json()
                    threads = data.get("threads", [])
                    
                    if not threads:
                        break
                    
                    total_scanned += len(threads)
                    
                    # Filter for this property
                    matching = [t for t in threads if str(t.get("listing_id", "")) == str(property_id)]
                    property_threads.extend(matching)
                    
                    if page % 10 == 0:
                        print(f"   Scanned {total_scanned} threads, found {len(property_threads)} for this property...")
                    
                    if not data.get("next_page") or len(property_threads) >= max_threads:
                        break
                    page += 1
                    await asyncio.sleep(0.03)
                except Exception as e:
                    print(f"   Error fetching page {page}: {e}")
                    break
        
        all_threads = property_threads[:max_threads]
        print(f"   Scanned {total_scanned} total threads, found {len(all_threads)} for this property")
        
        # Process each thread
        total_messages = 0
        for i, thread in enumerate(all_threads):
            thread_id = thread.get("id")
            guest_name = thread.get("guest_name", "Guest")
            
            # Fetch messages for this thread
            messages = await self.hostify.get_inbox_messages(thread_id, limit=100)
            total_messages += len(messages)
            
            if not messages:
                continue
            
            # Sort messages by timestamp
            messages.sort(key=lambda m: m.get("created", ""))
            
            # Extract Q&A pairs (guest question -> host answer)
            for j, msg in enumerate(messages):
                # Determine if this is a guest message using Hostify's complex logic
                # Values: "guest" = from guest, "host" = from host, "automatic" = from host (auto)
                sender_field = str(msg.get("from") or "")
                msg_guest_name = str(msg.get("guest_name") or "")
                
                # Determine direction:
                # 1. from="guest" → guest message (inbound)
                # 2. from="automatic" → host automated message (outbound)
                # 3. from="host" BUT msg_guest_name matches thread guest_name → 
                #    actually a guest message (Hostify bug workaround)
                # 4. Otherwise → host message (outbound)
                is_guest_message = False
                if sender_field == "guest":
                    is_guest_message = True
                elif sender_field == "host" and msg_guest_name and msg_guest_name.strip().lower() == guest_name.strip().lower():
                    # Hostify bug: sometimes marks guest messages as from="host"
                    # but includes the guest's name in guest_name field
                    is_guest_message = True
                
                # If this is a guest message, look for host response
                if is_guest_message:
                    raw_question = msg.get("message")
                    guest_question = str(raw_question).strip() if raw_question else ""
                    if not guest_question or len(guest_question) < 10:
                        continue
                    
                    # Look for the next host response
                    for k in range(j + 1, min(j + 5, len(messages))):
                        next_sender_field = messages[k].get("from", "")
                        next_msg_guest_name = messages[k].get("guest_name") or ""
                        
                        # Check if next message is from host (not guest)
                        is_host_response = False
                        if next_sender_field == "automatic":
                            is_host_response = True
                        elif next_sender_field == "host":
                            # Only host if guest_name doesn't match (otherwise it's guest due to bug)
                            if not next_msg_guest_name or str(next_msg_guest_name).strip().lower() != guest_name.strip().lower():
                                is_host_response = True
                        
                        if is_host_response:
                            raw_answer = messages[k].get("message")
                            host_answer = str(raw_answer).strip() if raw_answer else ""
                            if host_answer and len(host_answer) > 10:
                                qa_pairs.append(QAPair(
                                    guest_question=guest_question,
                                    host_answer=host_answer,
                                    timestamp=msg.get("created", ""),
                                    thread_id=thread_id,
                                    guest_name=guest_name
                                ))
                            break
            
            if (i + 1) % 10 == 0:
                print(f"   Processed {i + 1}/{len(all_threads)} threads...")
            
            await asyncio.sleep(0.1)
        
        print(f"   ✅ Found {len(qa_pairs)} Q&A pairs from {total_messages} messages")
        return qa_pairs
    
    def _generate_rules_prompt(self, property_name: str, qa_pairs: List[QAPair], is_global: bool = False) -> str:
        """Generate the prompt for rule extraction."""
        
        pairs_text = "\n\n---\n\n".join([
            f"Guest: {qa.guest_question}\nHost: {qa.host_answer}"
            for qa in qa_pairs[:75]  # Limit to avoid token limits
        ])
        
        context = "across multiple properties" if is_global else f"for {property_name}"
        
        return f"""Analyze these vacation rental conversations {context} and extract reusable RAG rules.

CONVERSATIONS:
{pairs_text}

INSTRUCTIONS:
1. Identify common question patterns and the best host responses
2. Group similar questions together into rules
3. Extract property-specific information (WiFi passwords, door codes, etc.) if mentioned
4. Note any troubleshooting patterns
5. Identify questions that could be auto-answered vs. need human review

CATEGORIES to use:
- wifi: WiFi network names, passwords, connection issues
- check_in: Check-in times, procedures, door codes, key locations
- check_out: Check-out procedures, times, cleaning expectations
- amenities: Pool, hot tub, grill, gym, etc.
- parking: Parking locations, permits, restrictions
- appliances: How to use TV, thermostat, washer/dryer, etc.
- local_tips: Restaurants, grocery stores, attractions
- policies: House rules, quiet hours, pet policies, smoking
- troubleshooting: Common issues and solutions
- emergency: Emergency contacts, urgent situations
- booking: Reservation changes, extensions, cancellations
- other: Anything that doesn't fit above

OUTPUT FORMAT (JSON):
{{
    "rules": [
        {{
            "category": "wifi",
            "title": "WiFi Connection",
            "trigger_questions": [
                "What's the WiFi password?",
                "How do I connect to the internet?",
                "WiFi not working"
            ],
            "response_template": "The WiFi network is [NETWORK_NAME] and the password is [PASSWORD]. If you're having trouble connecting, try restarting the router located [LOCATION].",
            "context_notes": "Always include both network name and password. Mention router restart as troubleshooting step.",
            "confidence": 0.9,
            "source_conversations": 5,
            "property_specific": true
        }}
    ],
    "insights": {{
        "most_common_topics": ["wifi", "check_in", "parking"],
        "questions_needing_human": ["refund requests", "damage reports"],
        "suggested_improvements": "Consider adding clearer check-in instructions"
    }}
}}

Create 5-15 rules based on the patterns you see. Focus on:
1. Questions that appear multiple times
2. Information that would be useful for future guests
3. Troubleshooting steps that worked
4. Property-specific details mentioned by the host"""

    async def generate_property_rules(
        self, 
        property_id: str, 
        property_name: str,
        qa_pairs: List[QAPair]
    ) -> PropertyRules:
        """Generate RAG rules for a single property."""
        
        print(f"\n🤖 Generating rules for {property_name}...")
        
        if len(qa_pairs) < 3:
            print(f"   ⚠️ Not enough Q&A pairs ({len(qa_pairs)}) to generate meaningful rules")
            return PropertyRules(
                property_id=property_id,
                property_name=property_name,
                rules=[],
                total_conversations_analyzed=len(set(qa.thread_id for qa in qa_pairs)),
                total_messages_analyzed=len(qa_pairs) * 2,
                generated_at=datetime.utcnow().isoformat()
            )
        
        prompt = self._generate_rules_prompt(property_name, qa_pairs, is_global=False)
        
        try:
            response = self.openai.chat.completions.create(
                model="gpt-4o",
                response_format={"type": "json_object"},
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at analyzing vacation rental conversations and extracting reusable knowledge for an AI assistant. Output valid JSON only."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=4000
            )
            
            result = json.loads(response.choices[0].message.content)
            raw_rules = result.get("rules", [])
            
            # Convert to RAGRule objects
            rules = []
            for i, r in enumerate(raw_rules):
                rule = RAGRule(
                    rule_id=f"{property_id}_rule_{i+1}",
                    category=r.get("category", "other"),
                    title=r.get("title", "Untitled"),
                    trigger_questions=r.get("trigger_questions", []),
                    response_template=r.get("response_template", ""),
                    context_notes=r.get("context_notes", ""),
                    confidence=r.get("confidence", 0.7),
                    source_conversations=r.get("source_conversations", 1),
                    property_specific=r.get("property_specific", True)
                )
                rules.append(rule)
            
            print(f"   ✅ Generated {len(rules)} rules")
            
            # Save insights separately
            insights = result.get("insights", {})
            if insights:
                insights_path = OUTPUT_DIR / f"{property_id}_insights.json"
                with open(insights_path, 'w') as f:
                    json.dump(insights, f, indent=2)
                print(f"   📝 Saved insights to {insights_path.name}")
            
            return PropertyRules(
                property_id=property_id,
                property_name=property_name,
                rules=rules,
                total_conversations_analyzed=len(set(qa.thread_id for qa in qa_pairs)),
                total_messages_analyzed=len(qa_pairs) * 2,
                generated_at=datetime.utcnow().isoformat()
            )
            
        except Exception as e:
            print(f"   ❌ Error generating rules: {e}")
            return PropertyRules(
                property_id=property_id,
                property_name=property_name,
                rules=[],
                total_conversations_analyzed=0,
                total_messages_analyzed=0,
                generated_at=datetime.utcnow().isoformat()
            )
    
    async def generate_global_rules(self) -> GlobalRules:
        """Generate cross-property rules from all collected Q&A pairs."""
        
        print(f"\n{'='*60}")
        print("🌐 Generating GLOBAL rules (cross-property patterns)")
        print(f"{'='*60}")
        
        # Combine all Q&A pairs
        all_pairs = []
        properties_analyzed = []
        
        for property_id, pairs in self.all_qa_pairs.items():
            all_pairs.extend(pairs)
            properties_analyzed.append(property_id)
        
        if len(all_pairs) < 10:
            print("   ⚠️ Not enough total Q&A pairs for meaningful global rules")
            return GlobalRules(
                rules=[],
                properties_analyzed=properties_analyzed,
                common_question_categories={},
                generated_at=datetime.utcnow().isoformat()
            )
        
        print(f"   Total Q&A pairs across all properties: {len(all_pairs)}")
        
        # Sample pairs for analysis (avoid token limits)
        import random
        sample_pairs = random.sample(all_pairs, min(100, len(all_pairs)))
        
        prompt = self._generate_rules_prompt("ALL PROPERTIES", sample_pairs, is_global=True)
        
        # Add global-specific instructions
        global_instructions = """

ADDITIONAL GLOBAL ANALYSIS:
- Focus on questions that appear across MULTIPLE properties
- These rules should be general enough to apply to ANY vacation rental
- Look for universal patterns like "How do I..." questions
- Identify common troubleshooting steps that work everywhere
- Note questions that ALWAYS need human review (complaints, refunds, etc.)

For global rules, set "property_specific": false"""
        
        prompt += global_instructions
        
        try:
            response = self.openai.chat.completions.create(
                model="gpt-4o",
                response_format={"type": "json_object"},
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at analyzing vacation rental conversations across multiple properties to identify universal patterns. Output valid JSON only."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=4000
            )
            
            result = json.loads(response.choices[0].message.content)
            raw_rules = result.get("rules", [])
            
            # Convert to RAGRule objects
            rules = []
            for i, r in enumerate(raw_rules):
                rule = RAGRule(
                    rule_id=f"global_rule_{i+1}",
                    category=r.get("category", "other"),
                    title=r.get("title", "Untitled"),
                    trigger_questions=r.get("trigger_questions", []),
                    response_template=r.get("response_template", ""),
                    context_notes=r.get("context_notes", ""),
                    confidence=r.get("confidence", 0.7),
                    source_conversations=r.get("source_conversations", 1),
                    property_specific=False
                )
                rules.append(rule)
            
            # Count categories
            category_counts = defaultdict(int)
            for rule in rules:
                category_counts[rule.category] += 1
            
            print(f"   ✅ Generated {len(rules)} global rules")
            
            return GlobalRules(
                rules=rules,
                properties_analyzed=properties_analyzed,
                common_question_categories=dict(category_counts),
                generated_at=datetime.utcnow().isoformat()
            )
            
        except Exception as e:
            print(f"   ❌ Error generating global rules: {e}")
            return GlobalRules(
                rules=[],
                properties_analyzed=properties_analyzed,
                common_question_categories={},
                generated_at=datetime.utcnow().isoformat()
            )
    
    def save_property_rules(self, rules: PropertyRules):
        """Save property rules to a file."""
        # Create a clean filename
        safe_name = "".join(c if c.isalnum() or c in "._- " else "_" for c in rules.property_name)
        filename = f"property_{rules.property_id}_{safe_name[:30]}.json"
        filepath = OUTPUT_DIR / filename
        
        # Convert to dict for JSON serialization
        data = {
            "property_id": rules.property_id,
            "property_name": rules.property_name,
            "total_conversations_analyzed": rules.total_conversations_analyzed,
            "total_messages_analyzed": rules.total_messages_analyzed,
            "generated_at": rules.generated_at,
            "rules_count": len(rules.rules),
            "rules": [asdict(r) for r in rules.rules]
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"   💾 Saved to: {filepath.name}")
        
        # Also save a human-readable markdown version
        md_filename = f"property_{rules.property_id}_{safe_name[:30]}.md"
        md_filepath = OUTPUT_DIR / md_filename
        
        md_content = self._rules_to_markdown(rules)
        with open(md_filepath, 'w') as f:
            f.write(md_content)
        
        print(f"   📄 Saved readable version: {md_filename}")
    
    def save_global_rules(self, rules: GlobalRules):
        """Save global rules to a file."""
        filepath = OUTPUT_DIR / "GLOBAL_master_rules.json"
        
        data = {
            "properties_analyzed": rules.properties_analyzed,
            "common_question_categories": rules.common_question_categories,
            "generated_at": rules.generated_at,
            "rules_count": len(rules.rules),
            "rules": [asdict(r) for r in rules.rules]
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"   💾 Saved to: {filepath.name}")
        
        # Also save markdown version
        md_filepath = OUTPUT_DIR / "GLOBAL_master_rules.md"
        md_content = self._global_rules_to_markdown(rules)
        with open(md_filepath, 'w') as f:
            f.write(md_content)
        
        print(f"   📄 Saved readable version: GLOBAL_master_rules.md")
    
    def _rules_to_markdown(self, rules: PropertyRules) -> str:
        """Convert property rules to readable markdown."""
        lines = [
            f"# RAG Rules: {rules.property_name}",
            f"",
            f"**Property ID:** {rules.property_id}",
            f"**Generated:** {rules.generated_at}",
            f"**Conversations Analyzed:** {rules.total_conversations_analyzed}",
            f"**Messages Analyzed:** {rules.total_messages_analyzed}",
            f"**Rules Generated:** {len(rules.rules)}",
            f"",
            "---",
            ""
        ]
        
        # Group by category
        by_category = defaultdict(list)
        for rule in rules.rules:
            by_category[rule.category].append(rule)
        
        for category in sorted(by_category.keys()):
            category_rules = by_category[category]
            lines.append(f"## {category.upper()}")
            lines.append("")
            
            for rule in category_rules:
                lines.append(f"### {rule.title}")
                lines.append(f"**Confidence:** {rule.confidence:.0%}")
                lines.append(f"**Based on:** {rule.source_conversations} conversation(s)")
                lines.append("")
                
                lines.append("**Trigger Questions:**")
                for q in rule.trigger_questions:
                    lines.append(f"- {q}")
                lines.append("")
                
                lines.append("**Response Template:**")
                lines.append(f"> {rule.response_template}")
                lines.append("")
                
                if rule.context_notes:
                    lines.append(f"**Notes:** {rule.context_notes}")
                    lines.append("")
                
                lines.append("---")
                lines.append("")
        
        return "\n".join(lines)
    
    def _global_rules_to_markdown(self, rules: GlobalRules) -> str:
        """Convert global rules to readable markdown."""
        lines = [
            "# GLOBAL Master Rules (Cross-Property)",
            "",
            f"**Generated:** {rules.generated_at}",
            f"**Properties Analyzed:** {len(rules.properties_analyzed)}",
            f"**Rules Generated:** {len(rules.rules)}",
            "",
            "## Question Category Distribution",
            ""
        ]
        
        for cat, count in sorted(rules.common_question_categories.items(), key=lambda x: -x[1]):
            lines.append(f"- **{cat}:** {count} rules")
        
        lines.extend(["", "---", ""])
        
        # Group by category
        by_category = defaultdict(list)
        for rule in rules.rules:
            by_category[rule.category].append(rule)
        
        for category in sorted(by_category.keys()):
            category_rules = by_category[category]
            lines.append(f"## {category.upper()}")
            lines.append("")
            
            for rule in category_rules:
                lines.append(f"### {rule.title}")
                lines.append(f"**Confidence:** {rule.confidence:.0%}")
                lines.append(f"**Universal Rule:** {'Yes' if not rule.property_specific else 'No'}")
                lines.append("")
                
                lines.append("**Trigger Questions:**")
                for q in rule.trigger_questions:
                    lines.append(f"- {q}")
                lines.append("")
                
                lines.append("**Response Template:**")
                lines.append(f"> {rule.response_template}")
                lines.append("")
                
                if rule.context_notes:
                    lines.append(f"**Notes:** {rule.context_notes}")
                    lines.append("")
                
                lines.append("---")
                lines.append("")
        
        return "\n".join(lines)


def get_properties_from_db(limit: int = 5) -> List[Dict[str, Any]]:
    """
    Get unique properties from the database.
    Checks both GuestIndex cache and Conversations table.
    Fallback when Hostify API is unavailable.
    """
    from models import Conversation
    db = SessionLocal()
    try:
        seen = set()
        properties = []
        
        # First try GuestIndex
        guest_results = db.query(
            GuestIndex.listing_id,
            GuestIndex.listing_name
        ).filter(
            GuestIndex.listing_id.isnot(None),
            GuestIndex.listing_id != ""
        ).distinct().all()
        
        for listing_id, listing_name in guest_results:
            if listing_id and listing_id not in seen:
                seen.add(listing_id)
                properties.append({
                    "id": listing_id,
                    "name": listing_name or f"Property {listing_id}"
                })
        
        # Also check Conversations table (may have listings not in GuestIndex)
        conv_results = db.query(
            Conversation.listing_id,
            Conversation.listing_name
        ).filter(
            Conversation.listing_id.isnot(None),
            Conversation.listing_id != ""
        ).distinct().all()
        
        for listing_id, listing_name in conv_results:
            if listing_id and listing_id not in seen:
                seen.add(listing_id)
                properties.append({
                    "id": listing_id,
                    "name": listing_name or f"Property {listing_id}"
                })
        
        # Sort by name and return limited set
        properties.sort(key=lambda x: x.get("name", ""))
        return properties[:limit]
    finally:
        db.close()


async def main(num_properties: int = 5, use_db_fallback: bool = True, prioritize_with_messages: bool = True):
    """Main entry point."""
    print("\n" + "="*70)
    print("🚀 RAG RULES GENERATOR")
    print(f"   Processing {num_properties} properties")
    print(f"   Output directory: {OUTPUT_DIR.absolute()}")
    print("="*70)
    
    generator = RAGRulesGenerator()
    
    # Fetch all listings
    print("\n📋 Fetching property listings...")
    listings = await generator.hostify.get_listings()
    
    if not listings and use_db_fallback:
        print("   ⚠️ Hostify API unavailable, using cached properties from database...")
        listings = get_properties_from_db(num_properties * 5)  # Get more to filter
    
    if not listings:
        print("❌ No listings found. Check your Hostify API key or database.")
        return
    
    print(f"   Found {len(listings)} total properties")
    
    # Prioritize properties that have message threads
    if prioritize_with_messages:
        print("\n🔍 Scanning for properties with conversation history...")
        all_threads = []
        page = 1
        max_pages = 50  # Fetch up to 5000 threads
        while page <= max_pages:
            threads = await generator.hostify.get_inbox_threads(limit=100, page=page)
            if not threads:
                break
            all_threads.extend(threads)
            if page % 5 == 0 or len(threads) < 100:
                print(f"   Fetched page {page}: {len(all_threads)} total threads so far")
            if len(threads) < 100:
                break
            page += 1
            await asyncio.sleep(0.1)
        print(f"   ✅ Found {len(all_threads)} total conversation threads")
        
        # Count threads per listing
        thread_counts = defaultdict(int)
        for thread in all_threads:
            lid = str(thread.get("listing_id", ""))
            if lid:
                thread_counts[lid] += 1
        
        # Sort listings by thread count (descending)
        def get_thread_count(listing):
            return thread_counts.get(str(listing.get("id", "")), 0)
        
        listings_with_threads = [l for l in listings if get_thread_count(l) > 0]
        listings_with_threads.sort(key=get_thread_count, reverse=True)
        
        print(f"   Found {len(listings_with_threads)} properties with conversations")
        
        if listings_with_threads:
            properties_to_process = listings_with_threads[:num_properties]
        else:
            properties_to_process = listings[:num_properties]
    else:
        properties_to_process = listings[:num_properties]
    
    print(f"\n📦 Will process these {len(properties_to_process)} properties:")
    for i, listing in enumerate(properties_to_process, 1):
        thread_count = thread_counts.get(str(listing.get("id", "")), 0) if prioritize_with_messages else "?"
        print(f"   {i}. {listing.get('name', 'Unknown')} (ID: {listing.get('id')}) - {thread_count} threads")
    
    # Process each property
    all_property_rules = []
    
    for listing in properties_to_process:
        property_id = str(listing.get("id"))
        property_name = listing.get("name", "Unknown Property")
        
        # Fetch messages
        qa_pairs = await generator.fetch_property_messages(
            property_id, 
            property_name,
            max_threads=50
        )
        
        # Store for global analysis
        generator.all_qa_pairs[property_id] = qa_pairs
        
        # Generate rules
        property_rules = await generator.generate_property_rules(
            property_id,
            property_name,
            qa_pairs
        )
        
        # Save to file
        if property_rules.rules:
            generator.save_property_rules(property_rules)
            all_property_rules.append(property_rules)
    
    # Generate global rules
    global_rules = await generator.generate_global_rules()
    if global_rules.rules:
        generator.save_global_rules(global_rules)
    
    # Print summary
    print("\n" + "="*70)
    print("✅ GENERATION COMPLETE")
    print("="*70)
    print(f"\n📊 Summary:")
    print(f"   Properties processed: {len(properties_to_process)}")
    print(f"   Properties with rules: {len(all_property_rules)}")
    print(f"   Total property rules: {sum(len(pr.rules) for pr in all_property_rules)}")
    print(f"   Global rules: {len(global_rules.rules)}")
    print(f"\n📁 Output files saved to: {OUTPUT_DIR.absolute()}")
    print("\n   Files generated:")
    for f in sorted(OUTPUT_DIR.glob("*")):
        print(f"   - {f.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate RAG rules from conversation history")
    parser.add_argument(
        "--properties", "-p",
        type=int,
        default=5,
        help="Number of properties to process (default: 5)"
    )
    parser.add_argument(
        "--threads-per-property", "-t",
        type=int,
        default=50,
        help="Max conversation threads per property (default: 50)"
    )
    
    args = parser.parse_args()
    
    asyncio.run(main(num_properties=args.properties))
