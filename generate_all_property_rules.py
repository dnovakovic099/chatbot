#!/usr/bin/env python3
"""
Generate RAG rules for all properties in Guest Health Settings.
Scans all Hostify inbox threads, saves them to database, extracts Q&A pairs,
and generates rules using GPT-4o.
"""

import asyncio
import json
import random
import os
from datetime import datetime
from typing import List, Dict, Any, Optional
from collections import defaultdict

import httpx
from openai import OpenAI

# Import from local modules
from models import (
    GuestHealthSettings, HostifyThread, HostifyMessage, 
    SessionLocal, Base, engine
)
from cache import HostifyClient


def get_enabled_properties() -> List[Dict[str, str]]:
    """Get all properties enabled in Guest Health Settings."""
    db = SessionLocal()
    try:
        settings = db.query(GuestHealthSettings).filter(
            GuestHealthSettings.is_enabled == True
        ).all()
        
        return [
            {"id": s.listing_id, "name": s.listing_name or f"Property {s.listing_id}"}
            for s in settings
        ]
    finally:
        db.close()


def save_threads_batch_to_db(threads_data: List[Dict]) -> int:
    """Save a batch of threads to the database. Returns count saved."""
    if not threads_data:
        return 0
    
    db = SessionLocal()
    saved = 0
    try:
        for thread_data in threads_data:
            thread_id = thread_data.get("id")
            if not thread_id:
                continue
            
            # Check if exists
            existing = db.query(HostifyThread).filter(
                HostifyThread.thread_id == thread_id
            ).first()
            
            # Parse dates
            checkin = None
            checkout = None
            try:
                if thread_data.get("checkin"):
                    checkin = datetime.fromisoformat(thread_data["checkin"].replace("Z", "+00:00"))
                if thread_data.get("checkout"):
                    checkout = datetime.fromisoformat(thread_data["checkout"].replace("Z", "+00:00"))
            except:
                pass
            
            if existing:
                # Update
                existing.listing_id = str(thread_data.get("listing_id", "")) or existing.listing_id
                existing.listing_name = thread_data.get("listing_name") or existing.listing_name
                existing.guest_name = thread_data.get("guest_name") or existing.guest_name
                existing.guest_email = thread_data.get("guest_email") or existing.guest_email
                existing.reservation_id = thread_data.get("reservation_id") or existing.reservation_id
                existing.checkin = checkin or existing.checkin
                existing.checkout = checkout or existing.checkout
                existing.last_message = thread_data.get("last_message") or existing.last_message
                existing.synced_at = datetime.utcnow()
            else:
                # Create new
                new_thread = HostifyThread(
                    thread_id=thread_id,
                    listing_id=str(thread_data.get("listing_id", "")),
                    listing_name=thread_data.get("listing_name"),
                    guest_name=thread_data.get("guest_name"),
                    guest_email=thread_data.get("guest_email"),
                    reservation_id=thread_data.get("reservation_id"),
                    checkin=checkin,
                    checkout=checkout,
                    last_message=thread_data.get("last_message"),
                    synced_at=datetime.utcnow()
                )
                db.add(new_thread)
            saved += 1
        
        db.commit()
        return saved
    except Exception as e:
        db.rollback()
        print(f"    Error saving batch: {e}")
        return 0
    finally:
        db.close()


def save_messages_to_db(thread_id: int, messages: List[Dict]) -> int:
    """Save messages for a thread to the database. Returns count saved."""
    db = SessionLocal()
    saved = 0
    try:
        for msg in messages:
            msg_id = msg.get("id")
            if not msg_id:
                continue
            
            # Check if exists
            existing = db.query(HostifyMessage).filter(
                HostifyMessage.hostify_message_id == msg_id
            ).first()
            
            if existing:
                continue  # Already saved
            
            # Parse sent_at
            sent_at = None
            try:
                if msg.get("created"):
                    sent_at = datetime.fromisoformat(msg["created"].replace("Z", "+00:00"))
            except:
                sent_at = datetime.utcnow()
            
            # Determine direction
            sender_type = str(msg.get("from") or "")
            direction = "outbound" if sender_type in ("host", "automatic") else "inbound"
            
            new_msg = HostifyMessage(
                hostify_message_id=msg_id,
                inbox_id=thread_id,
                reservation_id=msg.get("reservation_id"),
                direction=direction,
                content=str(msg.get("message") or ""),
                sender_name=msg.get("sender_name"),
                sender_type=sender_type,
                guest_name=msg.get("guest_name"),
                sent_at=sent_at,
                synced_at=datetime.utcnow()
            )
            db.add(new_msg)
            saved += 1
        
        db.commit()
        
        # Update thread's messages_synced_at
        thread = db.query(HostifyThread).filter(HostifyThread.thread_id == thread_id).first()
        if thread:
            thread.messages_synced_at = datetime.utcnow()
            thread.message_count = db.query(HostifyMessage).filter(
                HostifyMessage.inbox_id == thread_id
            ).count()
            db.commit()
        
        return saved
    except Exception as e:
        db.rollback()
        print(f"    Error saving messages for thread {thread_id}: {e}")
        return 0
    finally:
        db.close()


async def scan_and_save_all_threads(target_property_ids: List[str] = None) -> Dict[str, int]:
    """
    Scan all Hostify inbox threads and save to database.
    Returns: {listing_id: thread_count}
    """
    client = HostifyClient()
    thread_counts = defaultdict(int)
    
    page = 1
    total_scanned = 0
    total_saved = 0
    batch_to_save = []
    BATCH_SIZE = 100  # Save in batches to reduce DB locks
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Scanning all Hostify inbox threads and saving to database...")
    
    async with httpx.AsyncClient(timeout=60.0) as http_client:
        while True:
            try:
                response = await http_client.get(
                    f"{client.base_url}/inbox",
                    headers=client.headers,
                    params={"limit": 100, "page": page}
                )
                data = response.json()
                threads = data.get("threads", [])
                
                if not threads:
                    break
                
                total_scanned += len(threads)
                
                # Collect threads to save
                for t in threads:
                    listing_id = str(t.get("listing_id", ""))
                    thread_counts[listing_id] += 1
                    
                    # Add to batch (all threads, or just target properties if specified)
                    if target_property_ids is None or listing_id in target_property_ids:
                        batch_to_save.append(t)
                
                # Save batch when it reaches BATCH_SIZE
                if len(batch_to_save) >= BATCH_SIZE:
                    saved = save_threads_batch_to_db(batch_to_save)
                    total_saved += saved
                    batch_to_save = []
                    await asyncio.sleep(0.1)  # Brief pause after DB write
                
                if page % 50 == 0:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] Page {page}: {total_scanned:,} scanned, {total_saved:,} saved to DB")
                
                if not data.get("next_page"):
                    break
                
                page += 1
                await asyncio.sleep(0.02)
                
            except Exception as e:
                print(f"  Error on page {page}: {e}")
                await asyncio.sleep(1)
                continue
    
    # Save any remaining threads
    if batch_to_save:
        saved = save_threads_batch_to_db(batch_to_save)
        total_saved += saved
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Scan complete: {total_scanned:,} threads scanned, {total_saved:,} saved to DB")
    return dict(thread_counts)


async def sync_messages_for_property(property_id: str, property_name: str) -> int:
    """Sync messages for all threads of a property. Returns total Q&A pairs extracted."""
    db = SessionLocal()
    client = HostifyClient()
    
    try:
        # Get all threads for this property from DB
        threads = db.query(HostifyThread).filter(
            HostifyThread.listing_id == property_id
        ).all()
        
        print(f"  Found {len(threads)} threads in database for {property_name}")
        
        total_messages = 0
        qa_pairs = []
        
        for i, thread in enumerate(threads):
            try:
                # Fetch messages from API
                messages = await client.get_inbox_messages(thread.thread_id, limit=100)
                if not messages:
                    continue
                
                messages.sort(key=lambda m: m.get("created", ""))
                
                # Save messages to DB
                saved = save_messages_to_db(thread.thread_id, messages)
                total_messages += len(messages)
                
                # Extract Q&A pairs
                guest_name = thread.guest_name or ""
                
                for j, msg in enumerate(messages):
                    sender_type = str(msg.get("from") or "")
                    msg_guest_name = str(msg.get("guest_name") or "")
                    
                    # Determine if guest message (Hostify bug workaround)
                    is_guest = False
                    if sender_type == "guest":
                        is_guest = True
                    elif sender_type == "host" and msg_guest_name and guest_name:
                        if msg_guest_name.strip().lower() == guest_name.strip().lower():
                            is_guest = True
                    
                    if is_guest:
                        question = str(msg.get("message") or "").strip()
                        if not question or len(question) < 10:
                            continue
                        
                        # Find host response
                        for k in range(j + 1, min(j + 5, len(messages))):
                            next_sender = str(messages[k].get("from") or "")
                            next_guest = str(messages[k].get("guest_name") or "")
                            
                            is_host = False
                            if next_sender == "automatic":
                                is_host = True
                            elif next_sender == "host":
                                if not next_guest or not guest_name:
                                    is_host = True
                                elif next_guest.strip().lower() != guest_name.strip().lower():
                                    is_host = True
                            
                            if is_host:
                                answer = str(messages[k].get("message") or "").strip()
                                if answer and len(answer) > 10:
                                    qa_pairs.append({
                                        "thread_id": thread.thread_id,
                                        "guest_name": guest_name,
                                        "question": question,
                                        "answer": answer,
                                        "timestamp": msg.get("created", "")
                                    })
                                break
                
                if (i + 1) % 50 == 0:
                    print(f"    [{property_name}] Processed {i+1}/{len(threads)} threads, {len(qa_pairs)} Q&A pairs, {total_messages} messages")
                
            except Exception as e:
                print(f"    Error on thread {thread.thread_id}: {e}")
                continue
            
            await asyncio.sleep(0.03)
        
        print(f"  Synced {total_messages:,} messages, extracted {len(qa_pairs):,} Q&A pairs")
        return qa_pairs
        
    finally:
        db.close()


def generate_rules_for_property(qa_pairs: List[Dict], property_name: str) -> Dict:
    """Generate RAG rules using GPT-4o."""
    if len(qa_pairs) < 5:
        return {"rules": [], "insights": {"note": "Too few Q&A pairs"}}
    
    # Sample for rule generation (GPT has token limits)
    sample_size = min(150, len(qa_pairs))
    sample = random.sample(qa_pairs, sample_size)
    
    # Build prompt
    pairs_text = "\n\n---\n\n".join([
        f"Guest: {p['question'][:500]}\nHost: {p['answer'][:500]}"
        for p in sample
    ])
    
    prompt = f'''Analyze these vacation rental conversations for "{property_name}" and extract reusable RAG rules.

CONVERSATIONS:
{pairs_text}

Create comprehensive rules covering ALL common topics. For each rule include:
- category: wifi, check_in, check_out, amenities, parking, appliances, local_tips, policies, troubleshooting, emergency, booking, other
- title: Brief descriptive title
- trigger_questions: 3-5 example questions that should trigger this rule
- response_template: A complete response template with placeholders like [TIME], [CODE], etc.
- context_notes: Important notes for the AI
- confidence: 0.0-1.0
- source_conversations: estimated count

OUTPUT FORMAT (JSON):
{{
    "rules": [...],
    "insights": {{
        "most_common_topics": [...],
        "questions_needing_human": [...],
        "suggested_improvements": "..."
    }}
}}

Generate 15-25 comprehensive rules covering all patterns you see.'''
    
    client = OpenAI()
    
    response = client.chat.completions.create(
        model="gpt-4o",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "You are an expert at analyzing vacation rental conversations and extracting reusable knowledge. Output valid JSON only."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=8000
    )
    
    return json.loads(response.choices[0].message.content)


def save_property_results(property_id: str, property_name: str, qa_pairs: List[Dict], rules_data: Dict, threads_count: int):
    """Save Q&A pairs and rules for a property."""
    os.makedirs("generated_rag_rules", exist_ok=True)
    
    # Save Q&A pairs
    qa_output = {
        "property_id": property_id,
        "property_name": property_name,
        "threads_processed": threads_count,
        "qa_pairs_count": len(qa_pairs),
        "extracted_at": datetime.now().isoformat(),
        "pairs": qa_pairs
    }
    
    with open(f"generated_rag_rules/{property_id}_qa_pairs.json", "w") as f:
        json.dump(qa_output, f, indent=2)
    
    # Save rules
    rules = rules_data.get("rules", [])
    insights = rules_data.get("insights", {})
    
    rules_output = {
        "property_id": property_id,
        "property_name": property_name,
        "threads_analyzed": threads_count,
        "qa_pairs_total": len(qa_pairs),
        "generated_at": datetime.now().isoformat(),
        "rules_count": len(rules),
        "rules": rules,
        "insights": insights
    }
    
    with open(f"generated_rag_rules/{property_id}_rules.json", "w") as f:
        json.dump(rules_output, f, indent=2)
    
    # Generate markdown
    md_lines = [
        f"# RAG Rules: {property_name}",
        "",
        f"**Property ID:** {property_id}",
        f"**Generated:** {datetime.now().isoformat()}",
        f"**Threads Analyzed:** {threads_count}",
        f"**Q&A Pairs:** {len(qa_pairs):,}",
        f"**Rules Generated:** {len(rules)}",
        "",
        "---",
        ""
    ]
    
    # Group by category
    by_category = defaultdict(list)
    for rule in rules:
        by_category[rule.get("category", "other")].append(rule)
    
    for category in sorted(by_category.keys()):
        md_lines.append(f"## {category.upper()}")
        md_lines.append("")
        
        for rule in by_category[category]:
            md_lines.append(f"### {rule.get('title', 'Untitled')}")
            md_lines.append(f"**Confidence:** {rule.get('confidence', 0.8):.0%}")
            md_lines.append(f"**Based on:** {rule.get('source_conversations', 1)} conversation(s)")
            md_lines.append("")
            
            md_lines.append("**Trigger Questions:**")
            for q in rule.get("trigger_questions", []):
                md_lines.append(f"- {q}")
            md_lines.append("")
            
            md_lines.append("**Response Template:**")
            md_lines.append(f"> {rule.get('response_template', '')}")
            md_lines.append("")
            
            if rule.get("context_notes"):
                md_lines.append(f"**Notes:** {rule.get('context_notes')}")
                md_lines.append("")
            
            md_lines.append("---")
            md_lines.append("")
    
    # Add insights
    if insights:
        md_lines.append("## INSIGHTS")
        md_lines.append("")
        if insights.get("most_common_topics"):
            md_lines.append("**Most Common Topics:** " + ", ".join(insights["most_common_topics"]))
        if insights.get("questions_needing_human"):
            md_lines.append("")
            md_lines.append("**Questions Needing Human Review:**")
            for q in insights["questions_needing_human"]:
                md_lines.append(f"- {q}")
        if insights.get("suggested_improvements"):
            md_lines.append("")
            md_lines.append(f"**Suggested Improvements:** {insights['suggested_improvements']}")
    
    with open(f"generated_rag_rules/{property_id}_rules.md", "w") as f:
        f.write("\n".join(md_lines))
    
    print(f"    Saved: {property_id}_qa_pairs.json, {property_id}_rules.json, {property_id}_rules.md")


async def main():
    """Main entry point."""
    print("=" * 60)
    print("RAG RULES GENERATOR - All Guest Health Properties")
    print("Threads and messages will be saved to database")
    print("=" * 60)
    print()
    
    # Ensure tables exist
    Base.metadata.create_all(bind=engine)
    print("Database tables ready.")
    print()
    
    # Get enabled properties
    properties = get_enabled_properties()
    print(f"Found {len(properties)} properties enabled in Guest Health Settings:")
    for p in properties:
        print(f"  - {p['id']}: {p['name']}")
    print()
    
    target_ids = [p["id"] for p in properties]
    
    # Step 1: Scan all threads and save ALL to DB
    thread_counts = await scan_and_save_all_threads(target_property_ids=None)  # Save ALL threads
    
    # Show thread counts for our properties
    print()
    print("Thread counts for target properties:")
    for p in properties:
        count = thread_counts.get(p["id"], 0)
        print(f"  - {p['name']}: {count:,} threads")
    print()
    
    # Step 2: Process each property - sync messages and generate rules
    results_summary = []
    
    for i, prop in enumerate(properties):
        property_id = prop["id"]
        property_name = prop["name"]
        threads_count = thread_counts.get(property_id, 0)
        
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(properties)}] Processing: {property_name}")
        print(f"{'='*60}")
        
        if threads_count == 0:
            print(f"  No threads found, skipping...")
            results_summary.append({
                "property_id": property_id,
                "property_name": property_name,
                "threads": 0,
                "qa_pairs": 0,
                "rules": 0,
                "status": "skipped"
            })
            continue
        
        # Sync messages and extract Q&A pairs
        qa_pairs = await sync_messages_for_property(property_id, property_name)
        
        if len(qa_pairs) < 5:
            print(f"  Not enough Q&A pairs for rule generation ({len(qa_pairs)} found)")
            save_property_results(property_id, property_name, qa_pairs, {"rules": [], "insights": {}}, threads_count)
            results_summary.append({
                "property_id": property_id,
                "property_name": property_name,
                "threads": threads_count,
                "qa_pairs": len(qa_pairs),
                "rules": 0,
                "status": "insufficient_data"
            })
            continue
        
        # Generate rules
        print(f"  Generating rules with GPT-4o...")
        rules_data = generate_rules_for_property(qa_pairs, property_name)
        rules_count = len(rules_data.get("rules", []))
        print(f"  Generated {rules_count} rules")
        
        # Save results
        save_property_results(property_id, property_name, qa_pairs, rules_data, threads_count)
        
        results_summary.append({
            "property_id": property_id,
            "property_name": property_name,
            "threads": threads_count,
            "qa_pairs": len(qa_pairs),
            "rules": rules_count,
            "status": "completed"
        })
    
    # Final summary
    print()
    print("=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print()
    
    total_threads = sum(r["threads"] for r in results_summary)
    total_qa = sum(r["qa_pairs"] for r in results_summary)
    total_rules = sum(r["rules"] for r in results_summary)
    
    print(f"{'Property':<50} {'Threads':>10} {'Q&A':>10} {'Rules':>8}")
    print("-" * 80)
    for r in results_summary:
        status = "" if r["status"] == "completed" else ""
        print(f"{status} {r['property_name'][:48]:<48} {r['threads']:>10,} {r['qa_pairs']:>10,} {r['rules']:>8}")
    print("-" * 80)
    print(f"{'TOTAL':<50} {total_threads:>10,} {total_qa:>10,} {total_rules:>8}")
    
    # Database stats
    db = SessionLocal()
    try:
        total_db_threads = db.query(HostifyThread).count()
        total_db_messages = db.query(HostifyMessage).count()
        print()
        print(f"Database: {total_db_threads:,} threads, {total_db_messages:,} messages saved")
    finally:
        db.close()
    
    print()
    print(f"All results saved to generated_rag_rules/")
    
    # Save summary
    with open("generated_rag_rules/summary.json", "w") as f:
        json.dump({
            "generated_at": datetime.now().isoformat(),
            "properties_processed": len(results_summary),
            "total_threads": total_threads,
            "total_qa_pairs": total_qa,
            "total_rules": total_rules,
            "results": results_summary
        }, f, indent=2)


if __name__ == "__main__":
    asyncio.run(main())
