"""
Scan ALL Hostify inbox threads to find mansion threads.
Run with: python3 scan_mansion_threads.py
"""
import asyncio
import httpx
import json
from datetime import datetime

HOSTIFY_API_KEY = "aOGSVrcPGOvvSsGD4idPKvxKaD0HGaAW"
MANSION_ID = "300017822"

async def scan_all_threads():
    base_url = 'https://api-rms.hostify.com'
    headers = {'x-api-key': HOSTIFY_API_KEY}
    
    mansion_threads = []
    page = 1
    total_scanned = 0
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting scan for mansion ID {MANSION_ID}...")
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        while True:
            try:
                r = await client.get(
                    f'{base_url}/inbox', 
                    headers=headers, 
                    params={'limit': 100, 'page': page}
                )
                data = r.json()
                threads = data.get('threads', [])
                
                if not threads:
                    break
                
                total_scanned += len(threads)
                
                for t in threads:
                    if str(t.get('listing_id', '')) == MANSION_ID:
                        mansion_threads.append({
                            'id': t.get('id'),
                            'guest_name': t.get('guest_name'),
                            'checkin': t.get('checkin'),
                            'checkout': t.get('checkout'),
                            'last_message': t.get('last_message')
                        })
                
                if page % 100 == 0:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] Page {page}: {total_scanned:,} scanned, {len(mansion_threads)} mansion threads found")
                
                if not data.get('next_page'):
                    break
                    
                page += 1
                
            except Exception as e:
                print(f"Error on page {page}: {e}")
                await asyncio.sleep(1)
                continue
    
    print(f"\n{'='*60}")
    print(f"SCAN COMPLETE")
    print(f"{'='*60}")
    print(f"Total threads scanned: {total_scanned:,}")
    print(f"Mansion threads found: {len(mansion_threads)}")
    
    # Save results
    output = {
        'mansion_id': MANSION_ID,
        'total_scanned': total_scanned,
        'mansion_thread_count': len(mansion_threads),
        'scanned_at': datetime.now().isoformat(),
        'threads': mansion_threads
    }
    
    with open('mansion_threads.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nSaved to mansion_threads.json")
    
    # Show sample
    print(f"\nSample threads (first 10):")
    for t in mansion_threads[:10]:
        print(f"  {t['id']}: {t['guest_name']} ({t['checkin']})")

if __name__ == "__main__":
    asyncio.run(scan_all_threads())
