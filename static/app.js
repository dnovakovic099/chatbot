/**
 * AI Property Manager - The Brain Dashboard
 * Frontend JavaScript
 */

// ============ STATE ============

let currentConversationId = null;
let refreshInterval = null;

// ============ INITIALIZATION ============

document.addEventListener('DOMContentLoaded', () => {
    initNavigation();
    checkSystemStatus();
    loadDashboard();
    
    // Auto-refresh every 30 seconds
    refreshInterval = setInterval(() => {
        const activeView = document.querySelector('.view.active');
        if (activeView) {
            const viewId = activeView.id.replace('view-', '');
            refreshView(viewId);
        }
    }, 30000);
});

function initNavigation() {
    document.querySelectorAll('.nav-item').forEach(item => {
        item.addEventListener('click', (e) => {
            e.preventDefault();
            const view = item.dataset.view;
            switchView(view);
        });
    });
}

function switchView(viewName) {
    // Update nav
    document.querySelectorAll('.nav-item').forEach(item => {
        item.classList.toggle('active', item.dataset.view === viewName);
    });
    
    // Update views
    document.querySelectorAll('.view').forEach(view => {
        view.classList.toggle('active', view.id === `view-${viewName}`);
    });
    
    // Load view data
    refreshView(viewName);
}

function refreshView(viewName) {
    switch (viewName) {
        case 'dashboard':
            loadDashboard();
            break;
        case 'conversations':
            loadConversations();
            break;
        case 'guests':
            loadGuests();
            break;
        case 'logs':
            loadLogs();
            break;
        case 'config':
            loadConfig();
            break;
    }
}

// ============ API HELPERS ============

async function apiGet(endpoint) {
    try {
        const response = await fetch(`/api/admin${endpoint}`);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        return await response.json();
    } catch (error) {
        console.error(`API Error (${endpoint}):`, error);
        throw error;
    }
}

async function apiPost(endpoint, data = {}) {
    try {
        const response = await fetch(`/api/admin${endpoint}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        return await response.json();
    } catch (error) {
        console.error(`API Error (${endpoint}):`, error);
        throw error;
    }
}

// ============ SYSTEM STATUS ============

async function checkSystemStatus() {
    const indicator = document.getElementById('status-indicator');
    try {
        const health = await fetch('/health').then(r => r.json());
        indicator.classList.remove('offline', 'dry-run', 'online');
        
        if (health.dry_run_mode) {
            indicator.classList.add('dry-run');
            indicator.querySelector('.status-text').textContent = 'Dry Run Mode';
        } else {
            indicator.classList.add('online');
            indicator.querySelector('.status-text').textContent = 'System Online';
        }
    } catch (error) {
        indicator.classList.add('offline');
        indicator.querySelector('.status-text').textContent = 'Offline';
    }
}

// ============ DASHBOARD ============

async function loadDashboard() {
    await refreshStats();
    await loadRecentActivity();
}

async function refreshStats() {
    try {
        const stats = await apiGet('/stats');
        
        document.getElementById('stat-total').textContent = stats.total_conversations;
        document.getElementById('stat-active').textContent = stats.active_conversations;
        document.getElementById('stat-escalated').textContent = stats.escalated_conversations;
        document.getElementById('stat-resolved').textContent = stats.resolved_today;
        document.getElementById('stat-messages').textContent = stats.messages_today;
        document.getElementById('stat-auto').textContent = stats.auto_sent_today;
    } catch (error) {
        console.error('Failed to load stats:', error);
    }
}

async function loadRecentActivity() {
    const feed = document.getElementById('activity-feed');
    try {
        const logs = await apiGet('/logs?limit=10');
        
        if (logs.length === 0) {
            feed.innerHTML = '<div class="empty-state"><span class="empty-icon">📭</span><p>No recent activity</p></div>';
            return;
        }
        
        feed.innerHTML = logs.map(log => `
            <div class="activity-item">
                <div class="activity-icon">${getEventIcon(log.event_type)}</div>
                <div class="activity-content">
                    <div class="activity-title">${formatEventType(log.event_type)}</div>
                    <div class="activity-subtitle">${formatLogPayload(log)}</div>
                </div>
                <div class="activity-time">${formatTime(log.timestamp)}</div>
            </div>
        `).join('');
    } catch (error) {
        feed.innerHTML = '<div class="loading">Failed to load activity</div>';
    }
}

function getEventIcon(eventType) {
    const icons = {
        'message_received': '📨',
        'ai_response_generated': '🤖',
        'auto_sent': '✅',
        'escalated_l1': '⚠️',
        'escalated_l2': '🚨',
        'human_approved': '👍',
        'human_edited': '✏️',
        'human_resolved': '✔️',
        'api_error': '❌',
        'cache_miss': '🔍',
        'test_message_simulated': '🧪',
        'webhook_received': '📥'
    };
    return icons[eventType] || '📋';
}

function formatEventType(eventType) {
    return eventType.split('_').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ');
}

function formatLogPayload(log) {
    if (log.guest_phone) return `Phone: ${log.guest_phone}`;
    if (log.conversation_id) return `Conversation #${log.conversation_id}`;
    if (log.payload) {
        try {
            const data = JSON.parse(log.payload);
            if (data.message) return data.message.substring(0, 60) + '...';
            if (data.error) return `Error: ${data.error.substring(0, 50)}`;
            if (data.confidence) return `Confidence: ${(data.confidence * 100).toFixed(0)}%`;
        } catch (e) {}
    }
    return '';
}

function formatDate(timestamp) {
    if (!timestamp) return '-';
    const date = new Date(timestamp);
    if (isNaN(date.getTime())) return '-';
    
    // Format as "Jan 21" or "Jan 21, 2025" if different year
    const now = new Date();
    const options = { month: 'short', day: 'numeric' };
    if (date.getFullYear() !== now.getFullYear()) {
        options.year = 'numeric';
    }
    return date.toLocaleDateString('en-US', options);
}

function formatTime(timestamp) {
    if (!timestamp) return '-';
    const date = new Date(timestamp);
    if (isNaN(date.getTime())) return '-';
    
    const now = new Date();
    const diff = now - date;
    
    // If within last hour, show minutes ago
    if (diff < 3600000 && diff >= 0) {
        const mins = Math.floor(diff / 60000);
        return mins < 1 ? 'Just now' : `${mins}m ago`;
    }
    
    // If today, show time
    if (date.toDateString() === now.toDateString()) {
        return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    }
    
    // If this year, show month/day + time
    if (date.getFullYear() === now.getFullYear()) {
        return date.toLocaleDateString([], { month: 'short', day: 'numeric' }) + 
               ' ' + date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    }
    
    // Otherwise show full date
    return date.toLocaleDateString([], { year: 'numeric', month: 'short', day: 'numeric' });
}

// ============ CONVERSATIONS ============

async function loadConversations() {
    const list = document.getElementById('conversations-list');
    const filter = document.getElementById('status-filter').value;
    
    try {
        const endpoint = filter ? `/conversations?status=${filter}` : '/conversations';
        const conversations = await apiGet(endpoint);
        
        if (conversations.length === 0) {
            list.innerHTML = '<div class="empty-state"><span class="empty-icon">💬</span><p>No conversations yet</p></div>';
            return;
        }
        
        list.innerHTML = conversations.map(conv => `
            <div class="conversation-item ${conv.id === currentConversationId ? 'active' : ''}" 
                 onclick="selectConversation(${conv.id})">
                <div class="conversation-header">
                    <span class="conversation-name">${conv.guest_name || conv.guest_phone}</span>
                    <span class="conversation-status ${conv.status}">${conv.status.replace('_', ' ')}</span>
                </div>
                <div class="conversation-preview">${conv.last_message_preview || 'No messages'}</div>
                <div class="conversation-meta">
                    <span>${conv.listing_name || 'Unknown property'}</span>
                    <span>•</span>
                    <span>${conv.message_count} messages</span>
                </div>
                ${conv.check_in_date ? `
                    <div class="conversation-dates">
                        📅 ${formatDate(conv.check_in_date)}${conv.check_out_date ? ` → ${formatDate(conv.check_out_date)}` : ''}
                        ${conv.booking_source ? ` • ${conv.booking_source}` : ''}
                    </div>
                ` : ''}
            </div>
        `).join('');
    } catch (error) {
        list.innerHTML = '<div class="loading">Failed to load conversations</div>';
    }
}

function filterConversations() {
    loadConversations();
}

async function selectConversation(id) {
    currentConversationId = id;
    
    // Update active state
    document.querySelectorAll('.conversation-item').forEach(item => {
        item.classList.toggle('active', item.onclick.toString().includes(id));
    });
    
    const detail = document.getElementById('conversation-detail');
    
    try {
        const conv = await apiGet(`/conversations/${id}`);
        
        detail.innerHTML = `
            <div class="detail-header">
                <div class="detail-info">
                    <h2>${conv.guest_name || 'Unknown Guest'}</h2>
                    <p class="detail-property">${conv.listing_name || 'Unknown property'}</p>
                    <div class="detail-meta">
                        ${conv.booking_source ? `<span class="source-badge ${(conv.booking_source || '').toLowerCase().replace(/\s+/g, '-')}">${conv.booking_source}</span>` : ''}
                        ${conv.check_in_date || conv.check_out_date ? `
                            <span class="date-range">
                                📅 ${conv.check_in_date ? formatDate(conv.check_in_date) : '?'} → ${conv.check_out_date ? formatDate(conv.check_out_date) : '?'}
                            </span>
                        ` : ''}
                        <span class="guest-contact">📞 ${conv.guest_phone}</span>
                    </div>
                </div>
                <div class="detail-actions">
                    ${conv.status !== 'resolved' ? `
                        <button class="btn btn-danger" onclick="resolveConversation(${conv.id})">
                            Mark Resolved
                        </button>
                    ` : ''}
                </div>
            </div>
            <div class="messages-container">
                ${conv.messages.map((msg, idx) => {
                    const nextMsg = conv.messages[idx + 1];
                    const hasAiSuggestion = msg.direction === 'inbound' && msg.ai_suggested_reply;
                    const nextIsHostReply = nextMsg && nextMsg.direction === 'outbound';
                    
                    return `
                    <div class="message ${msg.direction}">
                        <div class="message-sender">${msg.direction === 'inbound' ? '👤 Guest' : '🏠 Host'}</div>
                        <div class="message-content">${escapeHtml(msg.content)}</div>
                        <div class="message-meta">
                            ${formatTime(msg.sent_at)}
                            ${msg.ai_confidence ? ` • AI ${(msg.ai_confidence * 100).toFixed(0)}%` : ''}
                            ${msg.was_auto_sent ? ' • Auto-sent' : ''}
                            ${msg.was_human_edited ? ' • Edited' : ''}
                        </div>
                    </div>
                    ${hasAiSuggestion && nextIsHostReply ? `
                        <div class="ai-suggestion-inline">
                            <div class="ai-suggestion-header">
                                <span class="ai-label">🤖 AI Would Have Said</span>
                                <span class="ai-confidence ${getConfidenceClass(msg.ai_suggestion_confidence)}">
                                    ${msg.ai_suggestion_confidence ? (msg.ai_suggestion_confidence * 100).toFixed(0) + '%' : ''}
                                </span>
                            </div>
                            <div class="ai-suggestion-text">${escapeHtml(msg.ai_suggested_reply)}</div>
                            ${msg.ai_suggestion_reasoning ? `
                                <div class="ai-suggestion-reasoning">${escapeHtml(msg.ai_suggestion_reasoning)}</div>
                            ` : ''}
                        </div>
                    ` : ''}
                    ${hasAiSuggestion && !nextIsHostReply && idx === conv.messages.length - 1 ? `
                        <div class="ai-suggestion-inline pending">
                            <div class="ai-suggestion-header">
                                <span class="ai-label">🤖 AI Suggestion Ready</span>
                                <span class="ai-confidence ${getConfidenceClass(msg.ai_suggestion_confidence)}">
                                    ${msg.ai_suggestion_confidence ? (msg.ai_suggestion_confidence * 100).toFixed(0) + '%' : ''}
                                </span>
                            </div>
                            <div class="ai-suggestion-text">${escapeHtml(msg.ai_suggested_reply)}</div>
                            <div class="ai-suggestion-actions">
                                <button class="btn btn-primary btn-sm" onclick="sendSuggestion(${conv.id}, '${escapeForJs(msg.ai_suggested_reply)}')">
                                    ✅ Send
                                </button>
                                <button class="btn btn-secondary btn-sm" onclick="editSuggestion(${conv.id}, '${escapeForJs(msg.ai_suggested_reply)}')">
                                    ✏️ Edit
                                </button>
                            </div>
                        </div>
                    ` : ''}
                `}).join('')}
            </div>
            
            ${(() => {
                // Check if last guest message has an AI suggestion
                const lastGuestMsg = [...conv.messages].reverse().find(m => m.direction === 'inbound');
                const lastMsgIsGuest = conv.messages.length > 0 && conv.messages[conv.messages.length - 1].direction === 'inbound';
                const hasStoredSuggestion = lastGuestMsg && lastGuestMsg.ai_suggested_reply;
                
                // Only show panel if we need a response AND there's no stored suggestion
                if (lastMsgIsGuest && !hasStoredSuggestion) {
                    return `
                        <div class="suggestion-panel" id="suggestion-panel-${conv.id}">
                            <div class="suggestion-header">
                                <span class="suggestion-icon">🤖</span>
                                <span class="suggestion-title">AI Suggested Response</span>
                            </div>
                            <div class="suggestion-empty">
                                <p>💬 Guest is waiting for a response</p>
                                <button class="btn btn-primary" onclick="generateSuggestionAndSave(${conv.id})">
                                    🤖 Generate AI Suggestion
                                </button>
                            </div>
                        </div>
                    `;
                }
                return '';
            })()}
        `;
        
        // Auto-scroll to bottom of messages
        const messagesContainer = document.querySelector('.messages-container');
        if (messagesContainer) {
            messagesContainer.scrollTop = messagesContainer.scrollHeight;
        }
    } catch (error) {
        detail.innerHTML = '<div class="empty-state"><span class="empty-icon">❌</span><p>Failed to load conversation</p></div>';
    }
}

async function generateSuggestionAndSave(conversationId) {
    // Generate and save AI suggestion to the message, then reload
    const panel = document.querySelector('.suggestion-panel');
    if (panel) {
        panel.innerHTML = `
            <div class="suggestion-header">
                <span class="suggestion-icon">🤖</span>
                <span class="suggestion-title">Generating...</span>
            </div>
            <div class="suggestion-loading">
                <div class="spinner"></div>
                <p>Generating and saving AI suggestion...</p>
            </div>
        `;
    }
    
    try {
        const result = await apiPost(`/conversations/${conversationId}/generate-and-save-suggestion`);
        
        if (result.success) {
            // Reload the conversation to show the saved suggestion inline
            await selectConversation(conversationId);
        } else {
            if (panel) {
                panel.innerHTML = `
                    <div class="suggestion-header">
                        <span class="suggestion-icon">❌</span>
                        <span class="suggestion-title">Error</span>
                    </div>
                    <div class="suggestion-error">
                        <p>${result.error || 'Failed to generate suggestion'}</p>
                        <button class="btn btn-ghost" onclick="generateSuggestionAndSave(${conversationId})">
                            🔄 Try Again
                        </button>
                    </div>
                `;
            }
        }
    } catch (error) {
        console.error("Error generating suggestion:", error);
        if (panel) {
            panel.innerHTML = `
                <div class="suggestion-header">
                    <span class="suggestion-icon">❌</span>
                    <span class="suggestion-title">Error</span>
                </div>
                <div class="suggestion-error">
                    <p>${error.message || 'Network error'}</p>
                    <button class="btn btn-ghost" onclick="generateSuggestionAndSave(${conversationId})">
                        🔄 Try Again
                    </button>
                </div>
            `;
        }
    }
}

async function generateSuggestion(conversationId, forLearning = false) {
    const panel = document.querySelector('.suggestion-panel');
    if (panel) {
        panel.innerHTML = `
            <div class="suggestion-header">
                <span class="suggestion-icon">🤖</span>
                <span class="suggestion-title">${forLearning ? 'AI Learning Mode' : 'AI Suggested Response'}</span>
            </div>
            <div class="suggestion-loading">
                <div class="spinner"></div>
                <p>Generating AI suggestion...</p>
            </div>
        `;
    }
    
    try {
        const url = forLearning 
            ? `/conversations/${conversationId}/generate-suggestion?for_learning=true`
            : `/conversations/${conversationId}/generate-suggestion`;
        const result = await apiPost(url);
        
        if (result.suggested_reply) {
            const sr = result.suggested_reply;
            const isLearning = result.for_learning || forLearning;
            const actualReply = result.actual_host_reply;
            
            panel.innerHTML = `
                <div class="suggestion-header">
                    <span class="suggestion-icon">🤖</span>
                    <span class="suggestion-title">${isLearning ? 'AI Learning Comparison' : 'AI Suggested Response'}</span>
                    <span class="suggestion-confidence ${getConfidenceClass(sr.confidence)}">
                        ${(sr.confidence * 100).toFixed(0)}% confident
                    </span>
                </div>
                
                ${isLearning && actualReply ? `
                    <div class="learning-comparison">
                        <div class="comparison-section ai-section">
                            <div class="comparison-label">🤖 AI Would Have Said:</div>
                            <div class="comparison-text">${escapeHtml(sr.text)}</div>
                        </div>
                        <div class="comparison-section host-section">
                            <div class="comparison-label">👤 Host Actually Said:</div>
                            <div class="comparison-text">${escapeHtml(actualReply)}</div>
                        </div>
                    </div>
                ` : `
                    <div class="suggestion-text">${escapeHtml(sr.text)}</div>
                `}
                
                ${sr.reasoning ? `
                    <div class="suggestion-reasoning">
                        <strong>Reasoning:</strong> ${escapeHtml(sr.reasoning)}
                    </div>
                ` : ''}
                ${sr.needs_human_review ? `
                    <div class="suggestion-warning">⚠️ Needs human review before sending</div>
                ` : ''}
                
                ${!isLearning ? `
                    <div class="suggestion-actions">
                        <button class="btn btn-primary" onclick="sendSuggestion(${conversationId}, '${escapeForJs(sr.text)}')">
                            ✅ Send This Response
                        </button>
                        <button class="btn btn-secondary" onclick="editSuggestion(${conversationId}, '${escapeForJs(sr.text)}')">
                            ✏️ Edit First
                        </button>
                        <button class="btn btn-ghost" onclick="generateSuggestion(${conversationId})">
                            🔄 Regenerate
                        </button>
                    </div>
                ` : `
                    <div class="suggestion-actions">
                        <button class="btn btn-ghost" onclick="generateSuggestion(${conversationId}, true)">
                            🔄 Regenerate Comparison
                        </button>
                    </div>
                `}
            `;
        } else {
            panel.innerHTML = `
                <div class="suggestion-header">
                    <span class="suggestion-icon">❌</span>
                    <span class="suggestion-title">AI Suggestion Failed</span>
                </div>
                <div class="suggestion-error">
                    <p>${result.error || 'Unable to generate suggestion'}</p>
                    <button class="btn btn-ghost" onclick="generateSuggestion(${conversationId}, ${forLearning})">
                        🔄 Try Again
                    </button>
                </div>
            `;
        }
    } catch (error) {
        panel.innerHTML = `
            <div class="suggestion-header">
                <span class="suggestion-icon">❌</span>
                <span class="suggestion-title">AI Suggestion Failed</span>
            </div>
            <div class="suggestion-error">
                <p>Error: ${error.message}</p>
                <button class="btn btn-ghost" onclick="generateSuggestion(${conversationId})">
                    🔄 Try Again
                </button>
            </div>
        `;
    }
}

async function resolveConversation(id) {
    try {
        await apiPost(`/conversations/${id}/resolve`);
        loadConversations();
        document.getElementById('conversation-detail').innerHTML = `
            <div class="empty-state">
                <span class="empty-icon">✅</span>
                <p>Conversation resolved</p>
            </div>
        `;
        currentConversationId = null;
    } catch (error) {
        alert('Failed to resolve conversation');
    }
}

// ============ TEST CONSOLE ============

function setTestMessage(message) {
    document.getElementById('test-message').value = message;
}

async function sendTestMessage() {
    const phone = document.getElementById('test-phone').value;
    const message = document.getElementById('test-message').value;
    const result = document.getElementById('test-result');
    
    if (!phone || !message) {
        result.className = 'test-result error';
        result.textContent = 'Please enter both phone number and message';
        return;
    }
    
    try {
        const response = await apiPost('/simulate', { phone, message });
        result.className = 'test-result success';
        result.textContent = `✓ Message queued for ${response.phone}\n${response.note}`;
        
        // Refresh after burst delay
        setTimeout(() => {
            loadConversations();
            loadLogs();
        }, 16000);
    } catch (error) {
        result.className = 'test-result error';
        result.textContent = `✗ Failed to send: ${error.message}`;
    }
}

async function addTestGuest() {
    try {
        const response = await apiPost('/test-guest');
        alert(`Test guest ${response.status}!\nName: ${response.guest}\nPhone: ${response.phone}`);
        loadGuests();
    } catch (error) {
        alert('Failed to add test guest');
    }
}

async function triggerSync() {
    try {
        await apiPost('/sync');
        alert('Reservation sync completed!');
        loadGuests();
    } catch (error) {
        alert('Sync failed');
    }
}

// ============ GUESTS ============

let guestSearchTimeout = null;

async function loadGuests() {
    const tbody = document.getElementById('guests-tbody');
    const count = document.getElementById('cache-count');
    const searchInput = document.getElementById('guest-search');
    const filterSelect = document.getElementById('guest-filter');
    
    const search = searchInput ? searchInput.value : '';
    const hasPhone = filterSelect ? filterSelect.value : '';
    
    try {
        // Get total count from stats
        const stats = await apiGet('/stats');
        count.textContent = `${stats.guests_in_cache.toLocaleString()} guests cached`;
        
        // Build query string
        let query = '?limit=100';
        if (search) query += `&search=${encodeURIComponent(search)}`;
        if (hasPhone === 'yes') query += '&has_phone=true';
        if (hasPhone === 'no') query += '&has_phone=false';
        
        const guests = await apiGet(`/guests${query}`);
        
        if (guests.length === 0) {
            tbody.innerHTML = '<tr><td colspan="7" class="loading">No guests found</td></tr>';
            return;
        }
        
        tbody.innerHTML = guests.map(guest => `
            <tr class="${!guest.guest_phone ? 'no-phone' : ''}">
                <td>
                    <strong>${escapeHtml(guest.guest_name)}</strong>
                    ${guest.source ? `<br><span class="source-badge ${guest.source.toLowerCase()}">${guest.source}</span>` : ''}
                </td>
                <td>
                    ${guest.guest_phone 
                        ? `<code class="phone-linked">${guest.guest_phone}</code>` 
                        : `<button class="btn-link-phone" onclick="showLinkPhoneModal('${guest.reservation_id}', '${escapeHtml(guest.guest_name)}')">
                             + Link Phone
                           </button>`
                    }
                </td>
                <td>${escapeHtml(guest.listing_name || '')}</td>
                <td>${guest.check_in_date ? new Date(guest.check_in_date).toLocaleDateString() : '-'}</td>
                <td>${guest.check_out_date ? new Date(guest.check_out_date).toLocaleDateString() : '-'}</td>
                <td class="reservation-id">${guest.reservation_id}</td>
                <td>${formatTime(guest.synced_at)}</td>
            </tr>
        `).join('');
    } catch (error) {
        tbody.innerHTML = '<tr><td colspan="7" class="loading">Failed to load guests</td></tr>';
    }
}

function searchGuests() {
    clearTimeout(guestSearchTimeout);
    guestSearchTimeout = setTimeout(loadGuests, 300);
}

function filterGuests() {
    loadGuests();
}

function showLinkPhoneModal(reservationId, guestName) {
    const modal = document.getElementById('link-phone-modal');
    document.getElementById('link-reservation-id').value = reservationId;
    document.getElementById('link-guest-name').textContent = guestName;
    document.getElementById('link-phone-input').value = '';
    document.getElementById('link-result').textContent = '';
    modal.classList.add('active');
}

function closeLinkPhoneModal() {
    document.getElementById('link-phone-modal').classList.remove('active');
}

async function submitLinkPhone() {
    const reservationId = document.getElementById('link-reservation-id').value;
    const phone = document.getElementById('link-phone-input').value;
    const result = document.getElementById('link-result');
    
    if (!phone) {
        result.className = 'link-result error';
        result.textContent = 'Please enter a phone number';
        return;
    }
    
    try {
        const response = await apiPost('/guests/link-phone', {
            reservation_id: reservationId,
            phone: phone
        });
        
        result.className = 'link-result success';
        result.textContent = `✓ Linked ${response.phone} to ${response.guest_name}`;
        
        setTimeout(() => {
            closeLinkPhoneModal();
            loadGuests();
        }, 1500);
    } catch (error) {
        result.className = 'link-result error';
        result.textContent = `✗ Failed: ${error.message}`;
    }
}

async function triggerSyncBackground() {
    const btn = document.querySelector('.sync-btn');
    btn.disabled = true;
    btn.textContent = '⏳ Syncing...';
    
    try {
        // Trigger sync in background
        fetch('/api/admin/sync', { method: 'POST' });
        
        // Poll for updates
        let lastCount = 0;
        const pollInterval = setInterval(async () => {
            const stats = await apiGet('/stats');
            document.getElementById('cache-count').textContent = 
                `${stats.guests_in_cache.toLocaleString()} guests cached`;
            
            if (stats.guests_in_cache === lastCount) {
                clearInterval(pollInterval);
                btn.disabled = false;
                btn.textContent = '🔄 Sync Now';
                loadGuests();
            }
            lastCount = stats.guests_in_cache;
        }, 3000);
        
    } catch (error) {
        btn.disabled = false;
        btn.textContent = '🔄 Sync Now';
        alert('Sync failed to start');
    }
}

// ============ LOGS ============

async function loadLogs() {
    const container = document.getElementById('logs-container');
    const filter = document.getElementById('log-filter').value;
    
    try {
        const endpoint = filter ? `/logs?event_type=${filter}` : '/logs';
        const logs = await apiGet(endpoint);
        
        if (logs.length === 0) {
            container.innerHTML = '<div class="empty-state"><span class="empty-icon">📋</span><p>No logs yet</p></div>';
            return;
        }
        
        container.innerHTML = logs.map(log => `
            <div class="log-entry">
                <div class="log-time">${new Date(log.timestamp).toLocaleString()}</div>
                <div class="log-type ${log.event_type}">${log.event_type}</div>
                <div class="log-content">
                    ${log.conversation_id ? `Conv #${log.conversation_id}` : ''}
                    ${log.guest_phone ? `• ${log.guest_phone}` : ''}
                    ${log.payload ? `<br><code>${truncate(log.payload, 100)}</code>` : ''}
                </div>
            </div>
        `).join('');
    } catch (error) {
        container.innerHTML = '<div class="loading">Failed to load logs</div>';
    }
}

function filterLogs() {
    loadLogs();
}

// ============ CONFIG ============

async function loadConfig() {
    const container = document.getElementById('config-container');
    
    try {
        const config = await apiGet('/config');
        
        container.innerHTML = `
            <div class="config-card">
                <h3>Mode</h3>
                <span class="config-badge ${config.dry_run_mode ? 'enabled' : 'disabled'}">
                    ${config.dry_run_mode ? '🧪 Dry Run Mode' : '🚀 Live Mode'}
                </span>
            </div>
            
            <div class="config-card">
                <h3>Auto-Send Threshold</h3>
                <div class="config-value">${(config.confidence_auto_send * 100).toFixed(0)}%</div>
            </div>
            
            <div class="config-card">
                <h3>Soft Escalate Threshold</h3>
                <div class="config-value">${(config.confidence_soft_escalate * 100).toFixed(0)}%</div>
            </div>
            
            <div class="config-card">
                <h3>Message Burst Delay</h3>
                <div class="config-value">${config.message_burst_delay_seconds}s</div>
            </div>
            
            <div class="config-card">
                <h3>L1 Escalation Timeout</h3>
                <div class="config-value">${config.escalation_l1_timeout_mins} min</div>
            </div>
            
            <div class="config-card">
                <h3>L2 Escalation Timeout</h3>
                <div class="config-value">${config.escalation_l2_timeout_mins} min</div>
            </div>
            
            <div class="config-card">
                <h3>Max Outbound/Hour</h3>
                <div class="config-value">${config.max_outbound_per_hour}</div>
            </div>
            
            <div class="config-card">
                <h3>Business Hours</h3>
                <div class="config-value mono">${config.business_hours}</div>
                <div style="color: var(--text-secondary); font-size: 0.85rem; margin-top: 4px;">${config.timezone}</div>
            </div>
            
            <div class="config-card" style="grid-column: span 2;">
                <h3>Escalation Keywords</h3>
                <div class="keywords-list">
                    ${config.escalation_keywords.map(kw => `<span class="keyword">${kw}</span>`).join('')}
                </div>
            </div>
        `;
    } catch (error) {
        container.innerHTML = '<div class="loading">Failed to load configuration</div>';
    }
}

// ============ UTILITIES ============

function escapeHtml(text) {
    if (!text) return '';
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function escapeForJs(text) {
    if (!text) return '';
    return text.replace(/'/g, "\\'").replace(/\n/g, "\\n").replace(/\r/g, "");
}

function getConfidenceClass(confidence) {
    if (confidence >= 0.8) return 'high';
    if (confidence >= 0.5) return 'medium';
    return 'low';
}

async function sendSuggestion(conversationId, text) {
    if (!confirm('Send this AI-generated response?')) return;
    try {
        // TODO: Implement actual send via Hostify API
        alert('Send functionality coming soon! For now, copy the text and send manually.');
    } catch (error) {
        alert('Failed to send: ' + error.message);
    }
}

function editSuggestion(conversationId, text) {
    const newText = prompt('Edit the response:', text);
    if (newText && newText !== text) {
        // TODO: Implement send edited response
        alert('Edit functionality coming soon! For now, copy and send manually.');
    }
}

async function regenerateSuggestion(conversationId) {
    await generateSuggestion(conversationId);
}

function truncate(str, len) {
    if (!str) return '';
    if (str.length <= len) return str;
    return str.substring(0, len) + '...';
}
