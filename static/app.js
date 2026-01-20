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
        case 'knowledge':
            loadKnowledgeListings();
            break;
        case 'logs':
            loadLogs();
            break;
        case 'config':
            loadConfig();
            break;
        case 'guest-health':
            loadGuestHealth();
            break;
        case 'health-settings':
            loadHealthSettings();
            break;
        case 'inquiry-analysis':
            loadInquiryAnalysis();
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
    
    // Server sends UTC timestamps without 'Z' suffix - add it so JS parses correctly
    let ts = timestamp;
    if (ts && !ts.endsWith('Z') && !ts.includes('+')) {
        ts = ts + 'Z';
    }
    
    const date = new Date(ts);
    if (isNaN(date.getTime())) return '-';
    
    const now = new Date();
    const diff = now - date;
    
    // If within last hour, show minutes ago
    if (diff < 3600000 && diff >= 0) {
        const mins = Math.floor(diff / 60000);
        return mins < 1 ? 'Just now' : `${mins}m ago`;
    }
    
    // If today, show time in local timezone
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
                ${(() => {
                    // System messages that don't need responses
                    const SYSTEM_MESSAGES = ['INQUIRY_CREATED', 'BOOKING_CONFIRMED', 'BOOKING_CANCELLED', 
                                             'CHECK_IN_REMINDER', 'CHECK_OUT_REMINDER', 'REVIEW_REMINDER'];
                    const isSystemMsg = (content) => SYSTEM_MESSAGES.includes((content || '').trim().toUpperCase());
                    
                    return conv.messages.map((msg, idx) => {
                        const nextMsg = conv.messages[idx + 1];
                        const hasAiSuggestion = msg.direction === 'inbound' && msg.ai_suggested_reply;
                        const nextIsHostReply = nextMsg && nextMsg.direction === 'outbound';
                        const nextIsSystemMsg = nextMsg && nextMsg.direction === 'inbound' && isSystemMsg(nextMsg.content);
                        const isLastRealMsg = idx === conv.messages.length - 1 || 
                                              (nextMsg && isSystemMsg(nextMsg.content) && !conv.messages.slice(idx + 1).some(m => m.direction === 'outbound'));
                        
                        return `
                        <div class="message ${msg.direction}">
                            <div class="message-sender">${msg.direction === 'inbound' ? '👤 Guest' : '🏠 Host'}</div>
                            <div class="message-content">
                                ${msg.attachment_url ? `<img src="${msg.attachment_url}" class="message-image" onclick="window.open('${msg.attachment_url}', '_blank')" alt="Attachment" style="max-width: 300px; max-height: 200px; border-radius: 8px; margin-bottom: 8px; cursor: pointer;" />` : ''}
                                ${escapeHtml(msg.content || '')}
                            </div>
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
                        ${hasAiSuggestion && !nextIsHostReply && isLastRealMsg ? `
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
                    `}).join('');
                })()}
            </div>
            
            ${(() => {
                // System messages that don't need responses
                const SYSTEM_MESSAGES = ['INQUIRY_CREATED', 'BOOKING_CONFIRMED', 'BOOKING_CANCELLED', 
                                         'CHECK_IN_REMINDER', 'CHECK_OUT_REMINDER', 'REVIEW_REMINDER'];
                
                // Filter out system messages to find real guest messages
                const realGuestMsgs = conv.messages.filter(m => 
                    m.direction === 'inbound' && 
                    !SYSTEM_MESSAGES.includes((m.content || '').trim().toUpperCase())
                );
                
                // Check if the last real guest message needs a response
                const lastRealGuestMsg = [...realGuestMsgs].reverse()[0];
                const lastMsgIsGuest = conv.messages.length > 0 && conv.messages[conv.messages.length - 1].direction === 'inbound';
                const lastMsgIsSystemMsg = lastMsgIsGuest && SYSTEM_MESSAGES.includes((conv.messages[conv.messages.length - 1].content || '').trim().toUpperCase());
                
                // Has AI suggestion for any unanswered guest message
                const hasStoredSuggestion = lastRealGuestMsg && lastRealGuestMsg.ai_suggested_reply;
                
                // Show panel only if real guest is waiting AND no AI suggestion exists yet
                // (Don't show for system messages like INQUIRY_CREATED)
                if (lastRealGuestMsg && !hasStoredSuggestion && !lastMsgIsSystemMsg) {
                    return `
                        <div class="suggestion-panel" id="suggestion-panel-${conv.id}">
                            <div class="suggestion-header">
                                <span class="suggestion-icon">🤖</span>
                                <span class="suggestion-title">AI Suggested Response</span>
                            </div>
                            <div class="suggestion-empty">
                                <p>⏳ Waiting for AI suggestion...</p>
                                <p class="help-text">AI suggestions are generated automatically when messages arrive via webhook.</p>
                                <button class="btn btn-secondary" onclick="generateSuggestionAndSave(${conv.id})">
                                    Generate Now
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


// ============ PROPERTY KNOWLEDGE BASE ============

let selectedKnowledgeListing = null;

async function loadKnowledgeListings() {
    const list = document.getElementById('knowledge-list');
    const countEl = document.getElementById('knowledge-listing-count');
    
    list.innerHTML = '<div class="loading">Loading properties...</div>';
    
    try {
        const data = await apiGet('/knowledge/listings');
        const listings = data.listings || [];
        
        countEl.textContent = `${listings.length} properties`;
        
        if (listings.length === 0) {
            list.innerHTML = `
                <div class="empty-state small">
                    <p>No properties found</p>
                    <p class="help-text">Conversations will appear here as they come in</p>
                </div>
            `;
            return;
        }
        
        // Sort: properties with knowledge first, then by name
        listings.sort((a, b) => {
            if (a.knowledge_count !== b.knowledge_count) {
                return b.knowledge_count - a.knowledge_count;
            }
            return (a.listing_name || '').localeCompare(b.listing_name || '');
        });
        
        list.innerHTML = listings.map(l => `
            <div class="knowledge-listing-item ${selectedKnowledgeListing === l.listing_id ? 'selected' : ''}" 
                 onclick="selectKnowledgeListing('${l.listing_id}', '${escapeForJs(l.listing_name || '')}')">
                <div class="listing-name">${escapeHtml(l.listing_name || 'Unknown Property')}</div>
                <div class="listing-stats">
                    ${l.knowledge_count > 0 ? `<span class="badge knowledge">${l.knowledge_count} entries</span>` : ''}
                    ${l.file_count > 0 ? `<span class="badge files">${l.file_count} files</span>` : ''}
                    ${l.knowledge_count === 0 && l.file_count === 0 ? '<span class="badge empty">No data</span>' : ''}
                </div>
            </div>
        `).join('');
        
    } catch (error) {
        list.innerHTML = `<div class="error">Failed to load: ${error.message}</div>`;
    }
}

async function selectKnowledgeListing(listingId, listingName) {
    selectedKnowledgeListing = listingId;
    
    // Update selection in list
    document.querySelectorAll('.knowledge-listing-item').forEach(el => {
        el.classList.remove('selected');
    });
    event.currentTarget.classList.add('selected');
    
    const detail = document.getElementById('knowledge-detail');
    detail.innerHTML = '<div class="loading">Loading knowledge...</div>';
    
    try {
        const data = await apiGet(`/knowledge/${listingId}`);
        
        const byType = data.by_type || {};
        const entries = data.entries || [];
        const files = data.files || [];
        
        detail.innerHTML = `
            <div class="knowledge-property-header">
                <h2>${escapeHtml(listingName || data.listing_name || 'Property')}</h2>
                <div class="knowledge-summary">
                    <span>${data.total_entries || 0} knowledge entries</span>
                    <span>•</span>
                    <span>${files.length} files uploaded</span>
                </div>
            </div>
            
            <div class="knowledge-actions">
                <div class="upload-section">
                    <h3>📤 Upload Property Documents</h3>
                    <p class="help-text">Upload PDFs, Word docs, or text files with property information</p>
                    <form id="upload-form" onsubmit="uploadPropertyFile(event, '${listingId}', '${escapeForJs(listingName)}')">
                        <input type="file" id="file-input" accept=".pdf,.docx,.doc,.txt,.md,.json" required>
                        <button type="submit" class="btn btn-primary">Upload & Process</button>
                    </form>
                    <div id="upload-status"></div>
                </div>
                
                <div class="learn-section">
                    <h3>🧠 AI Learning</h3>
                    <p class="help-text">Let AI analyze past conversations to extract knowledge</p>
                    <button class="btn btn-secondary" onclick="triggerLearningForProperty('${listingId}')">
                        Learn from This Property's Messages
                    </button>
                </div>
            </div>
            
            ${files.length > 0 ? `
                <div class="knowledge-section">
                    <h3>📁 Uploaded Files</h3>
                    <div class="files-list">
                        ${files.map(f => `
                            <div class="file-item ${f.processed ? 'processed' : 'pending'}">
                                <span class="file-icon">${getFileIcon(f.file_type)}</span>
                                <span class="file-name">${escapeHtml(f.filename)}</span>
                                <span class="file-status">
                                    ${f.processed 
                                        ? `✅ ${f.entries_created} entries extracted` 
                                        : '⏳ Processing...'}
                                </span>
                                ${!f.processed ? `
                                    <button class="btn btn-small" onclick="processFile('${listingId}', ${f.id})">
                                        Process Now
                                    </button>
                                ` : ''}
                            </div>
                        `).join('')}
                    </div>
                </div>
            ` : ''}
            
            <div class="knowledge-section">
                <div class="section-header">
                    <h3>📖 Knowledge Entries</h3>
                    <button class="btn btn-small btn-primary" onclick="showAddKnowledgeForm('${listingId}', '${escapeForJs(listingName)}')">
                        + Add Entry
                    </button>
                </div>
                
                ${entries.length === 0 ? `
                    <div class="empty-state small">
                        <p>No knowledge entries yet</p>
                        <p class="help-text">Upload documents or use AI learning to add knowledge</p>
                    </div>
                ` : `
                    <div class="knowledge-type-filter">
                        <button class="chip active" onclick="filterKnowledgeEntries('all')">All (${entries.length})</button>
                        ${Object.entries(byType).map(([type, count]) => `
                            <button class="chip" onclick="filterKnowledgeEntries('${type}')">${formatKnowledgeType(type)} (${count})</button>
                        `).join('')}
                    </div>
                    
                    <div class="knowledge-entries" id="knowledge-entries">
                        ${entries.map(e => `
                            <div class="knowledge-entry" data-type="${e.type}">
                                <div class="entry-header">
                                    <span class="entry-type ${e.type}">${formatKnowledgeType(e.type)}</span>
                                    <span class="entry-title">${escapeHtml(e.title)}</span>
                                    <span class="entry-source">${e.source}</span>
                                    <button class="btn-icon" onclick="deleteKnowledgeEntry(${e.id})" title="Delete">🗑️</button>
                                </div>
                                <div class="entry-content">${escapeHtml(e.content)}</div>
                                ${e.times_used > 0 ? `<div class="entry-usage">Used ${e.times_used} times</div>` : ''}
                            </div>
                        `).join('')}
                    </div>
                `}
            </div>
        `;
        
    } catch (error) {
        detail.innerHTML = `<div class="error">Failed to load: ${error.message}</div>`;
    }
}

function getFileIcon(fileType) {
    const icons = {
        'pdf': '📕',
        'docx': '📘',
        'doc': '📘',
        'txt': '📄',
        'md': '📝',
        'json': '📋'
    };
    return icons[fileType] || '📄';
}

function formatKnowledgeType(type) {
    const names = {
        'amenity': '🏊 Amenity',
        'house_rule': '📋 House Rule',
        'local_recommendation': '📍 Local',
        'appliance_guide': '🔧 Appliance',
        'common_issue': '⚠️ Issue',
        'faq': '❓ FAQ',
        'general': '📝 General'
    };
    return names[type] || type;
}

function filterKnowledgeEntries(type) {
    // Update active chip
    document.querySelectorAll('.knowledge-type-filter .chip').forEach(chip => {
        chip.classList.remove('active');
    });
    event.currentTarget.classList.add('active');
    
    // Filter entries
    document.querySelectorAll('.knowledge-entry').forEach(entry => {
        if (type === 'all' || entry.dataset.type === type) {
            entry.style.display = 'block';
        } else {
            entry.style.display = 'none';
        }
    });
}

async function uploadPropertyFile(event, listingId, listingName) {
    event.preventDefault();
    
    const fileInput = document.getElementById('file-input');
    const statusEl = document.getElementById('upload-status');
    
    if (!fileInput.files.length) {
        statusEl.innerHTML = '<div class="error">Please select a file</div>';
        return;
    }
    
    const file = fileInput.files[0];
    statusEl.innerHTML = '<div class="info">Uploading...</div>';
    
    const formData = new FormData();
    formData.append('file', file);
    formData.append('listing_name', listingName);
    
    try {
        const response = await fetch(`/api/admin/knowledge/${listingId}/upload`, {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        
        if (result.status === 'uploaded') {
            statusEl.innerHTML = `<div class="success">✅ ${result.message}</div>`;
            // Reload the property view
            setTimeout(() => selectKnowledgeListing(listingId, listingName), 2000);
        } else if (result.status === 'duplicate') {
            statusEl.innerHTML = `<div class="warning">⚠️ ${result.message}</div>`;
        } else {
            statusEl.innerHTML = `<div class="error">❌ ${result.error || 'Upload failed'}</div>`;
        }
    } catch (error) {
        statusEl.innerHTML = `<div class="error">❌ ${error.message}</div>`;
    }
}

async function processFile(listingId, fileId) {
    try {
        const result = await apiPost(`/knowledge/${listingId}/process/${fileId}`);
        
        if (result.success) {
            alert(`✅ Extracted ${result.entries_created} knowledge entries!`);
            // Reload
            loadKnowledgeListings();
            if (selectedKnowledgeListing) {
                selectKnowledgeListing(selectedKnowledgeListing, '');
            }
        } else {
            alert(`❌ Error: ${result.error}`);
        }
    } catch (error) {
        alert(`❌ Error: ${error.message}`);
    }
}

async function triggerLearning() {
    if (!confirm('This will analyze all conversations to extract knowledge. This may take a few minutes and use API credits. Continue?')) {
        return;
    }
    
    try {
        const result = await apiPost('/knowledge/learn');
        
        if (result.status === 'started') {
            alert('✅ Learning started in background. Refresh in a few minutes to see results.');
        } else if (result.status === 'already_running') {
            alert('⚠️ A learning session is already running. Please wait.');
        } else if (result.success) {
            alert(`✅ Learning complete!\n\nConversations analyzed: ${result.conversations_analyzed}\nNew entries: ${result.entries_created}\nUpdated entries: ${result.entries_updated}`);
            loadKnowledgeListings();
        } else {
            alert(`❌ Error: ${result.error}`);
        }
    } catch (error) {
        alert(`❌ Error: ${error.message}`);
    }
}

async function triggerLearningForProperty(listingId) {
    if (!confirm('Analyze this property\'s conversations to extract knowledge?')) {
        return;
    }
    
    try {
        const result = await apiPost(`/knowledge/learn?listing_id=${listingId}`);
        
        if (result.status === 'started') {
            alert('✅ Learning started. Refresh in a moment to see results.');
        } else if (result.success) {
            alert(`✅ Learning complete!\n\nMessages analyzed: ${result.messages_analyzed}\nNew entries: ${result.entries_created}`);
            selectKnowledgeListing(listingId, '');
        } else {
            alert(`❌ Error: ${result.error}`);
        }
    } catch (error) {
        alert(`❌ Error: ${error.message}`);
    }
}

async function deleteKnowledgeEntry(entryId) {
    if (!confirm('Delete this knowledge entry?')) {
        return;
    }
    
    try {
        await apiDelete(`/knowledge/entry/${entryId}`);
        // Reload current property
        if (selectedKnowledgeListing) {
            selectKnowledgeListing(selectedKnowledgeListing, '');
        }
    } catch (error) {
        alert(`❌ Error: ${error.message}`);
    }
}

function showAddKnowledgeForm(listingId, listingName) {
    const detail = document.getElementById('knowledge-detail');
    
    const formHtml = `
        <div class="add-knowledge-form">
            <h2>Add Knowledge Entry</h2>
            <p class="help-text">Manually add knowledge about ${escapeHtml(listingName)}</p>
            
            <form onsubmit="submitKnowledgeEntry(event, '${listingId}', '${escapeForJs(listingName)}')">
                <div class="form-group">
                    <label>Type</label>
                    <select id="new-knowledge-type" required>
                        <option value="amenity">🏊 Amenity</option>
                        <option value="house_rule">📋 House Rule</option>
                        <option value="local_recommendation">📍 Local Recommendation</option>
                        <option value="appliance_guide">🔧 Appliance Guide</option>
                        <option value="common_issue">⚠️ Common Issue</option>
                        <option value="faq" selected>❓ FAQ</option>
                        <option value="general">📝 General</option>
                    </select>
                </div>
                
                <div class="form-group">
                    <label>Title</label>
                    <input type="text" id="new-knowledge-title" placeholder="e.g., Pool Hours" required>
                </div>
                
                <div class="form-group">
                    <label>Content</label>
                    <textarea id="new-knowledge-content" rows="3" placeholder="Detailed information..." required></textarea>
                </div>
                
                <div class="form-group">
                    <label>Common Question (optional)</label>
                    <input type="text" id="new-knowledge-question" placeholder="e.g., What are the pool hours?">
                </div>
                
                <div class="form-group">
                    <label>Answer (optional)</label>
                    <textarea id="new-knowledge-answer" rows="2" placeholder="The answer to give guests..."></textarea>
                </div>
                
                <div class="form-actions">
                    <button type="button" class="btn btn-secondary" onclick="selectKnowledgeListing('${listingId}', '${escapeForJs(listingName)}')">Cancel</button>
                    <button type="submit" class="btn btn-primary">Save Entry</button>
                </div>
            </form>
        </div>
    `;
    
    detail.innerHTML = formHtml;
}

async function submitKnowledgeEntry(event, listingId, listingName) {
    event.preventDefault();
    
    const entry = {
        listing_id: listingId,
        listing_name: listingName,
        knowledge_type: document.getElementById('new-knowledge-type').value,
        title: document.getElementById('new-knowledge-title').value,
        content: document.getElementById('new-knowledge-content').value,
        question: document.getElementById('new-knowledge-question').value || null,
        answer: document.getElementById('new-knowledge-answer').value || null
    };
    
    try {
        const result = await apiPost('/knowledge/entry', entry);
        
        if (result.status === 'created') {
            alert('✅ Knowledge entry added!');
            selectKnowledgeListing(listingId, listingName);
        } else {
            alert(`❌ Error: ${result.error}`);
        }
    } catch (error) {
        alert(`❌ Error: ${error.message}`);
    }
}

async function apiDelete(endpoint) {
    const response = await fetch(`/api/admin${endpoint}`, {
        method: 'DELETE',
        headers: { 'Content-Type': 'application/json' }
    });
    return response.json();
}


// ============ GUEST HEALTH MONITOR ============

let guestHealthData = [];

async function loadGuestHealth() {
    await loadGuestHealthStats();
    await loadGuestHealthList();
}

async function loadGuestHealthStats() {
    try {
        const stats = await apiGet('/guest-health/stats');
        
        document.getElementById('health-stat-total').textContent = stats.total_checked_in_guests || 0;
        document.getElementById('health-stat-attention').textContent = stats.needs_attention || 0;
        document.getElementById('health-stat-risk').textContent = stats.at_risk_count || 0;
        document.getElementById('health-stat-issues').textContent = stats.total_unresolved_issues || 0;
    } catch (error) {
        console.error('Failed to load guest health stats:', error);
    }
}

async function loadGuestHealthList() {
    const container = document.getElementById('health-guests-container');
    
    try {
        const data = await apiGet('/guest-health/guests');
        guestHealthData = data.guests || [];
        
        if (guestHealthData.length === 0) {
            container.innerHTML = `
                <div class="empty-state">
                    <span class="empty-icon">💚</span>
                    <p>No guest health data available</p>
                    <p class="help-text">Make sure you have properties configured in Health Settings and guests currently checked in.</p>
                    <button class="btn btn-primary" onclick="switchView('health-settings')">Configure Properties</button>
                </div>
            `;
            return;
        }
        
        renderGuestHealthList(guestHealthData);
    } catch (error) {
        container.innerHTML = `<div class="error">Failed to load guest health data: ${error.message}</div>`;
    }
}

function renderGuestHealthList(guests) {
    const container = document.getElementById('health-guests-container');
    
    container.innerHTML = guests.map(g => `
        <div class="health-guest-card ${g.needs_attention ? 'attention' : ''} ${g.risk_level}" onclick="showGuestHealthDetail('${g.reservation_id}')">
            <div class="health-guest-header">
                <div class="health-guest-info">
                    <span class="health-guest-name">${escapeHtml(g.guest_name || 'Guest')}</span>
                    <span class="health-guest-property">${escapeHtml(g.listing_name || '')}</span>
                </div>
                <div class="health-badges">
                    ${g.needs_attention ? '<span class="badge attention-badge">⚠️ Needs Attention</span>' : ''}
                    <span class="badge risk-badge ${g.risk_level}">${g.risk_level.toUpperCase()}</span>
                </div>
            </div>
            
            <div class="health-sentiment-row">
                <div class="sentiment-indicator ${g.sentiment}">
                    ${getSentimentIcon(g.sentiment)}
                    <span class="sentiment-label">${formatSentiment(g.sentiment)}</span>
                </div>
                <div class="sentiment-score" style="color: ${getSentimentColor(g.sentiment_score)}">
                    ${g.sentiment_score !== null ? (g.sentiment_score > 0 ? '+' : '') + g.sentiment_score.toFixed(2) : 'N/A'}
                </div>
            </div>
            
            <div class="health-guest-details">
                <div class="detail-row">
                    <span class="detail-label">📅 Stay:</span>
                    <span class="detail-value">${formatDate(g.check_in_date)} → ${formatDate(g.check_out_date)}</span>
                </div>
                <div class="detail-row">
                    <span class="detail-label">🌙 Progress:</span>
                    <span class="detail-value">${g.nights_stayed || 0} nights stayed, ${g.nights_remaining || 0} remaining</span>
                </div>
                ${g.booking_source ? `
                    <div class="detail-row">
                        <span class="detail-label">📱 Source:</span>
                        <span class="detail-value source-badge ${(g.booking_source || '').toLowerCase()}">${g.booking_source}</span>
                    </div>
                ` : ''}
                <div class="detail-row">
                    <span class="detail-label">💬 Messages:</span>
                    <span class="detail-value">${g.total_messages || 0} total (${g.guest_messages || 0} from guest)</span>
                </div>
            </div>
            
            ${g.complaints && g.complaints.length > 0 ? `
                <div class="health-complaints">
                    <span class="complaints-label">Issues (${g.complaints.length}):</span>
                    <div class="complaints-list">
                        ${g.complaints.slice(0, 3).map(c => `
                            <span class="complaint-tag ${c.severity}">
                                ${getDepartmentIcon(c.department)} ${escapeHtml(c.issue.substring(0, 40))}${c.issue.length > 40 ? '...' : ''}
                            </span>
                        `).join('')}
                        ${g.complaints.length > 3 ? `<span class="more-tag">+${g.complaints.length - 3} more</span>` : ''}
                    </div>
                </div>
            ` : ''}
            
            ${g.attention_reason ? `
                <div class="health-attention-reason">
                    <strong>⚠️</strong> ${escapeHtml(g.attention_reason)}
                </div>
            ` : ''}
            
            <div class="health-guest-footer">
                <span class="last-analyzed">Last analyzed: ${formatTime(g.last_analyzed_at)}</span>
                <button class="btn btn-small btn-secondary" onclick="event.stopPropagation(); refreshSingleGuest('${g.reservation_id}')">🔄 Refresh</button>
            </div>
        </div>
    `).join('');
}

function getSentimentIcon(sentiment) {
    const icons = {
        'very_unhappy': '😡',
        'unhappy': '😞',
        'neutral': '😐',
        'happy': '🙂',
        'very_happy': '😊'
    };
    return icons[sentiment] || '😐';
}

function formatSentiment(sentiment) {
    const labels = {
        'very_unhappy': 'Very Unhappy',
        'unhappy': 'Unhappy',
        'neutral': 'Neutral',
        'happy': 'Happy',
        'very_happy': 'Very Happy'
    };
    return labels[sentiment] || 'Neutral';
}

function getSentimentColor(score) {
    if (score === null) return 'var(--text-tertiary)';
    if (score <= -0.5) return '#ef4444';
    if (score < 0) return '#f97316';
    if (score === 0) return 'var(--text-secondary)';
    if (score < 0.5) return '#84cc16';
    return '#22c55e';
}

function getDepartmentIcon(department) {
    const icons = {
        'housekeeping': '🧹',
        'maintenance': '🔧',
        'amenities': '🏊',
        'communication': '📞',
        'billing': '💳',
        'noise': '🔊',
        'safety': '🔒',
        'check_in': '🔑',
        'property_condition': '🏠'
    };
    return icons[department] || '📋';
}

function filterGuestHealth() {
    const sentimentFilter = document.getElementById('health-filter-sentiment').value;
    const riskFilter = document.getElementById('health-filter-risk').value;
    const attentionFilter = document.getElementById('health-filter-attention').checked;
    
    let filtered = guestHealthData;
    
    if (sentimentFilter) {
        filtered = filtered.filter(g => g.sentiment === sentimentFilter);
    }
    
    if (riskFilter) {
        filtered = filtered.filter(g => g.risk_level === riskFilter);
    }
    
    if (attentionFilter) {
        filtered = filtered.filter(g => g.needs_attention);
    }
    
    renderGuestHealthList(filtered);
}

async function refreshGuestHealth() {
    const container = document.getElementById('health-guests-container');
    container.innerHTML = `
        <div class="loading">
            <div class="spinner"></div>
            <p>Analyzing guests... This may take a moment.</p>
        </div>
    `;
    
    try {
        const result = await apiPost('/guest-health/refresh');
        
        if (result.status === 'no_properties') {
            container.innerHTML = `
                <div class="empty-state">
                    <span class="empty-icon">⚙️</span>
                    <p>${result.message}</p>
                    <button class="btn btn-primary" onclick="switchView('health-settings')">Configure Properties</button>
                </div>
            `;
            return;
        }
        
        await loadGuestHealth();
    } catch (error) {
        container.innerHTML = `<div class="error">Failed to refresh: ${error.message}</div>`;
    }
}

async function refreshSingleGuest(reservationId) {
    try {
        await apiPost(`/guest-health/guests/${reservationId}/refresh`);
        await loadGuestHealth();
    } catch (error) {
        alert('Failed to refresh guest: ' + error.message);
    }
}

async function showGuestHealthDetail(reservationId) {
    const panel = document.getElementById('health-detail-panel');
    const content = document.getElementById('health-detail-content');
    
    panel.style.display = 'block';
    content.innerHTML = '<div class="loading">Loading guest details...</div>';
    
    try {
        const data = await apiGet(`/guest-health/guests/${reservationId}`);
        const a = data.analysis;
        const messages = data.messages || [];
        
        content.innerHTML = `
            <div class="health-detail-header">
                <div class="detail-close" onclick="closeGuestHealthDetail()">×</div>
                <h2>${escapeHtml(a.guest_name || 'Guest')}</h2>
                <p class="detail-property">${escapeHtml(a.listing_name || '')}</p>
            </div>
            
            <div class="health-detail-summary">
                <div class="summary-card sentiment ${a.sentiment}">
                    <span class="summary-icon">${getSentimentIcon(a.sentiment)}</span>
                    <div class="summary-info">
                        <span class="summary-label">Sentiment</span>
                        <span class="summary-value">${formatSentiment(a.sentiment)}</span>
                    </div>
                </div>
                <div class="summary-card risk ${a.risk_level}">
                    <span class="summary-icon">⚠️</span>
                    <div class="summary-info">
                        <span class="summary-label">Risk Level</span>
                        <span class="summary-value">${a.risk_level.toUpperCase()}</span>
                    </div>
                </div>
            </div>
            
            ${a.sentiment_reasoning ? `
                <div class="detail-section">
                    <h3>AI Assessment</h3>
                    <p class="reasoning-text">${escapeHtml(a.sentiment_reasoning)}</p>
                </div>
            ` : ''}
            
            <div class="detail-section">
                <h3>Stay Details</h3>
                <div class="detail-grid">
                    <div class="detail-item">
                        <span class="label">Check-in</span>
                        <span class="value">${formatDate(a.check_in_date)}</span>
                    </div>
                    <div class="detail-item">
                        <span class="label">Check-out</span>
                        <span class="value">${formatDate(a.check_out_date)}</span>
                    </div>
                    <div class="detail-item">
                        <span class="label">Nights Stayed</span>
                        <span class="value">${a.nights_stayed || 0}</span>
                    </div>
                    <div class="detail-item">
                        <span class="label">Nights Remaining</span>
                        <span class="value">${a.nights_remaining || 0}</span>
                    </div>
                    ${a.booking_source ? `
                        <div class="detail-item">
                            <span class="label">Booking Source</span>
                            <span class="value">${a.booking_source}</span>
                        </div>
                    ` : ''}
                    ${a.guest_phone ? `
                        <div class="detail-item">
                            <span class="label">Phone</span>
                            <span class="value">${a.guest_phone}</span>
                        </div>
                    ` : ''}
                </div>
            </div>
            
            ${a.complaints && a.complaints.length > 0 ? `
                <div class="detail-section">
                    <h3>Complaints & Issues (${a.complaints.length})</h3>
                    <div class="complaints-detail-list">
                        ${a.complaints.map(c => `
                            <div class="complaint-detail-item ${c.severity}">
                                <div class="complaint-header">
                                    <span class="complaint-dept">${getDepartmentIcon(c.department)} ${c.department}</span>
                                    <span class="complaint-severity ${c.severity}">${c.severity}</span>
                                    <span class="complaint-status ${c.status}">${c.status}</span>
                                </div>
                                <p class="complaint-text">${escapeHtml(c.issue)}</p>
                            </div>
                        `).join('')}
                    </div>
                </div>
            ` : ''}
            
            ${a.recommended_actions && a.recommended_actions.length > 0 ? `
                <div class="detail-section">
                    <h3>Recommended Actions</h3>
                    <div class="actions-list">
                        ${a.recommended_actions.map(act => `
                            <div class="action-item ${act.priority}">
                                <span class="action-priority">${act.priority.toUpperCase()}</span>
                                <div class="action-content">
                                    <p class="action-text">${escapeHtml(act.action)}</p>
                                    ${act.reason ? `<p class="action-reason">${escapeHtml(act.reason)}</p>` : ''}
                                </div>
                            </div>
                        `).join('')}
                    </div>
                </div>
            ` : ''}
            
            <div class="detail-section">
                <h3>Message History (${messages.length})</h3>
                <div class="detail-messages">
                    ${messages.length === 0 ? '<p class="no-messages">No messages exchanged yet.</p>' : ''}
                    ${messages.slice(-20).map(m => `
                        <div class="detail-message ${m.direction === 'inbound' ? 'guest' : 'host'}">
                            <span class="message-role">${m.direction === 'inbound' ? '👤 Guest' : '🏠 Host'}</span>
                            <p class="message-text">${escapeHtml(m.content)}</p>
                            <span class="message-time">${formatTime(m.sent_at)}</span>
                        </div>
                    `).join('')}
                </div>
            </div>
            
            <div class="detail-footer">
                <span>Last analyzed: ${formatTime(a.last_analyzed_at)}</span>
                <button class="btn btn-primary" onclick="refreshSingleGuest('${a.reservation_id}'); showGuestHealthDetail('${a.reservation_id}');">
                    🔄 Re-analyze
                </button>
            </div>
        `;
    } catch (error) {
        content.innerHTML = `<div class="error">Failed to load guest details: ${error.message}</div>`;
    }
}

function closeGuestHealthDetail() {
    document.getElementById('health-detail-panel').style.display = 'none';
}


// ============ HEALTH SETTINGS ============

let availableProperties = [];
let selectedPropertyIds = new Set();

async function loadHealthSettings() {
    await loadAvailableProperties();
}

async function loadAvailableProperties() {
    const container = document.getElementById('property-list-container');
    
    try {
        // Get available properties
        const availableData = await apiGet('/guest-health/available-properties');
        availableProperties = availableData.properties || [];
        
        // Get current settings
        const settingsData = await apiGet('/guest-health/settings');
        const currentSettings = settingsData.settings || [];
        selectedPropertyIds = new Set(currentSettings.map(s => s.listing_id));
        
        if (availableProperties.length === 0) {
            container.innerHTML = `
                <div class="empty-state small">
                    <span class="empty-icon">🏠</span>
                    <p>No properties found</p>
                    <p class="help-text">Properties will appear here once you sync reservations.</p>
                </div>
            `;
            return;
        }
        
        renderPropertyList();
        updateSelectedCount();
    } catch (error) {
        container.innerHTML = `<div class="error">Failed to load properties: ${error.message}</div>`;
    }
}

function renderPropertyList() {
    const container = document.getElementById('property-list-container');
    
    container.innerHTML = `
        <div class="property-selection-list">
            ${availableProperties.map(p => `
                <label class="property-checkbox-item ${selectedPropertyIds.has(p.listing_id) ? 'selected' : ''}">
                    <input type="checkbox" 
                           value="${p.listing_id}" 
                           ${selectedPropertyIds.has(p.listing_id) ? 'checked' : ''}
                           onchange="toggleProperty('${p.listing_id}', this.checked)">
                    <div class="property-info">
                        <span class="property-name">${escapeHtml(p.listing_name || p.listing_id)}</span>
                        <span class="property-id">${p.listing_id}</span>
                    </div>
                    <span class="checkmark">✓</span>
                </label>
            `).join('')}
        </div>
    `;
}

function toggleProperty(listingId, isChecked) {
    if (isChecked) {
        selectedPropertyIds.add(listingId);
    } else {
        selectedPropertyIds.delete(listingId);
    }
    
    // Update visual state
    const item = document.querySelector(`input[value="${listingId}"]`).closest('.property-checkbox-item');
    item.classList.toggle('selected', isChecked);
    
    updateSelectedCount();
}

function selectAllProperties() {
    availableProperties.forEach(p => selectedPropertyIds.add(p.listing_id));
    renderPropertyList();
    updateSelectedCount();
}

function deselectAllProperties() {
    selectedPropertyIds.clear();
    renderPropertyList();
    updateSelectedCount();
}

function updateSelectedCount() {
    document.getElementById('selected-count').textContent = `${selectedPropertyIds.size} properties selected`;
}

async function saveHealthSettings() {
    const listingIds = Array.from(selectedPropertyIds);
    
    try {
        const result = await apiPost('/guest-health/settings/bulk', listingIds);
        
        alert(`✅ Settings saved!\n\n${result.added} properties added\n${result.removed} properties removed\n${result.total_monitored} total monitored`);
        
        // Offer to refresh guest health data
        if (listingIds.length > 0 && confirm('Would you like to refresh guest health data now?')) {
            switchView('guest-health');
            setTimeout(() => refreshGuestHealth(), 500);
        }
    } catch (error) {
        alert('Failed to save settings: ' + error.message);
    }
}


// ============ INQUIRY ANALYSIS ============

let inquiriesData = [];

async function loadInquiryAnalysis() {
    try {
        // Load summary and inquiries in parallel
        const [summaryData, inquiriesResult] = await Promise.all([
            apiGet('/inquiries/summary'),
            apiGet('/inquiries?limit=100')
        ]);
        
        inquiriesData = inquiriesResult.inquiries || [];
        
        renderInquirySummary(summaryData);
        renderInquiryList(inquiriesData);
    } catch (error) {
        console.error('Error loading inquiry analysis:', error);
        document.getElementById('inquiry-list-container').innerHTML = 
            '<div class="error-message">Failed to load inquiry data. Click "Analyze Inquiries" to start analysis.</div>';
    }
}

function renderInquirySummary(summary) {
    // Stats
    document.getElementById('stat-total-inquiries').textContent = summary.total_inquiries || 0;
    document.getElementById('stat-lost-revenue').textContent = formatCurrency(summary.total_lost_revenue || 0);
    document.getElementById('stat-avg-response').textContent = formatResponseTime(summary.avg_response_time_minutes || 0);
    document.getElementById('stat-avg-conversion').textContent = 
        Math.round((summary.avg_conversion_likelihood || 0) * 100) + '%';
    
    // Common mistakes
    const mistakesList = document.getElementById('common-mistakes-list');
    if (summary.common_mistakes && summary.common_mistakes.length > 0) {
        mistakesList.innerHTML = summary.common_mistakes.slice(0, 5).map(m => 
            `<li><span class="mistake-text">${escapeHtml(m.mistake)}</span><span class="count">${m.count}</span></li>`
        ).join('');
    } else {
        mistakesList.innerHTML = '<li class="empty">No data yet</li>';
    }
    
    // Training needs
    const trainingList = document.getElementById('training-needs-list');
    if (summary.top_training_needs && summary.top_training_needs.length > 0) {
        trainingList.innerHTML = summary.top_training_needs.slice(0, 5).map(t => 
            `<li><span class="training-text">${escapeHtml(t.topic)}</span><span class="count">${t.count}</span></li>`
        ).join('');
    } else {
        trainingList.innerHTML = '<li class="empty">No data yet</li>';
    }
    
    // Outcome breakdown
    const outcomeDiv = document.getElementById('outcome-breakdown');
    const outcomes = summary.outcomes || {};
    const outcomeLabels = {
        'booked_elsewhere': '📍 Booked Elsewhere',
        'price_objection': '💰 Price Objection',
        'dates_unavailable': '📅 Dates Unavailable',
        'slow_response': '⏰ Slow Response',
        'poor_communication': '💬 Poor Communication',
        'ghost': '👻 Guest Ghosted',
        'no_response': '❌ No Team Response',
        'requirements_not_met': '❓ Requirements Not Met',
        'still_deciding': '🤔 Still Deciding',
        'unknown': '❓ Unknown'
    };
    
    const sortedOutcomes = Object.entries(outcomes)
        .sort((a, b) => b[1] - a[1])
        .slice(0, 6);
    
    if (sortedOutcomes.length > 0) {
        outcomeDiv.innerHTML = sortedOutcomes.map(([outcome, count]) => 
            `<div class="outcome-item">
                <span class="outcome-label">${outcomeLabels[outcome] || outcome}</span>
                <span class="outcome-count">${count}</span>
            </div>`
        ).join('');
    } else {
        outcomeDiv.innerHTML = '<div class="empty">No data yet</div>';
    }
}

function renderInquiryList(inquiries) {
    const container = document.getElementById('inquiry-list-container');
    
    if (!inquiries || inquiries.length === 0) {
        container.innerHTML = `
            <div class="empty-state">
                <div class="empty-icon">📉</div>
                <h3>No Inquiries Analyzed</h3>
                <p>Click "Analyze Inquiries" to analyze recent inquiries that didn't convert to bookings.</p>
            </div>
        `;
        return;
    }
    
    container.innerHTML = inquiries.map(inquiry => renderInquiryCard(inquiry)).join('');
}

function renderInquiryCard(inquiry) {
    const responseTimeClass = inquiry.first_response_minutes > 60 ? 'bad' : 
                              inquiry.first_response_minutes > 30 ? '' : 'good';
    
    const conversionClass = inquiry.conversion_likelihood > 0.7 ? 'bad' : 
                            inquiry.conversion_likelihood > 0.4 ? '' : 'good';
    
    const mistakes = inquiry.team_mistakes || [];
    const mistakeTags = mistakes.slice(0, 3).map(m => 
        `<span class="mistake-tag severity-${m.severity}">${escapeHtml(m.mistake)}</span>`
    ).join('');
    
    const dates = [];
    if (inquiry.requested_checkin) {
        const checkin = new Date(inquiry.requested_checkin);
        const checkout = inquiry.requested_checkout ? new Date(inquiry.requested_checkout) : null;
        dates.push(`${checkin.toLocaleDateString()}`);
        if (checkout) {
            const nights = Math.round((checkout - checkin) / (1000 * 60 * 60 * 24));
            dates.push(`→ ${checkout.toLocaleDateString()} (${nights} nights)`);
        }
    }
    
    return `
        <div class="inquiry-card" onclick="showInquiryDetail(${inquiry.thread_id})">
            <div class="inquiry-card-header">
                <div>
                    <h3>${escapeHtml(inquiry.guest_name || 'Unknown Guest')}</h3>
                    <div class="inquiry-card-property">${escapeHtml(inquiry.listing_name || 'Unknown Property')}</div>
                </div>
                <span class="inquiry-outcome outcome-${inquiry.outcome}">${formatOutcome(inquiry.outcome)}</span>
            </div>
            
            ${dates.length > 0 ? `<div class="inquiry-card-dates">📅 ${dates.join(' ')}</div>` : ''}
            
            <div class="inquiry-card-metrics">
                <div class="inquiry-metric">
                    <span class="inquiry-metric-label">Response Time:</span>
                    <span class="inquiry-metric-value ${responseTimeClass}">${formatResponseTime(inquiry.first_response_minutes)}</span>
                </div>
                <div class="inquiry-metric">
                    <span class="inquiry-metric-label">Messages:</span>
                    <span class="inquiry-metric-value">${inquiry.total_messages || 0}</span>
                </div>
                <div class="inquiry-metric">
                    <span class="inquiry-metric-label">Could Have Converted:</span>
                    <span class="inquiry-metric-value ${conversionClass}">${Math.round((inquiry.conversion_likelihood || 0) * 100)}%</span>
                </div>
                <div class="inquiry-metric">
                    <span class="inquiry-metric-label">Est. Lost Revenue:</span>
                    <span class="inquiry-metric-value bad">${formatCurrency(inquiry.lost_revenue_estimate || 0)}</span>
                </div>
            </div>
            
            <div class="inquiry-card-reasoning">
                ${escapeHtml(inquiry.outcome_reasoning || 'No analysis available')}
            </div>
            
            ${mistakes.length > 0 ? `<div class="inquiry-card-mistakes">${mistakeTags}</div>` : ''}
        </div>
    `;
}

function formatOutcome(outcome) {
    const labels = {
        'booked_elsewhere': 'Booked Elsewhere',
        'price_objection': 'Price Objection',
        'dates_unavailable': 'Dates Unavailable',
        'slow_response': 'Slow Response',
        'poor_communication': 'Poor Communication',
        'ghost': 'Ghosted',
        'no_response': 'No Response',
        'requirements_not_met': 'Requirements Not Met',
        'still_deciding': 'Still Deciding',
        'unknown': 'Unknown'
    };
    return labels[outcome] || outcome;
}

function formatResponseTime(minutes) {
    if (!minutes && minutes !== 0) return 'N/A';
    if (minutes < 60) return `${Math.round(minutes)}m`;
    const hours = Math.floor(minutes / 60);
    const mins = Math.round(minutes % 60);
    if (hours < 24) return `${hours}h ${mins}m`;
    const days = Math.floor(hours / 24);
    return `${days}d ${hours % 24}h`;
}

function formatCurrency(amount) {
    if (!amount) return '$0';
    return '$' + Math.round(amount).toLocaleString();
}

async function showInquiryDetail(threadId) {
    try {
        const data = await apiGet(`/inquiries/${threadId}`);
        renderInquiryDetail(data);
        document.getElementById('inquiry-detail-modal').classList.add('active');
    } catch (error) {
        console.error('Error loading inquiry detail:', error);
        alert('Failed to load inquiry details');
    }
}

function renderInquiryDetail(data) {
    const analysis = data.analysis;
    const messages = data.messages || [];
    
    const content = document.getElementById('inquiry-detail-content');
    
    content.innerHTML = `
        <div class="inquiry-detail-header">
            <div>
                <h2>${escapeHtml(analysis.guest_name || 'Unknown Guest')}</h2>
                <div class="inquiry-detail-property">${escapeHtml(analysis.listing_name || 'Unknown Property')}</div>
            </div>
            <button class="inquiry-detail-close" onclick="closeInquiryDetail()">×</button>
        </div>
        
        <div class="inquiry-detail-section">
            <h3>📊 Analysis Summary</h3>
            <div class="inquiry-card-metrics" style="margin-bottom: 12px;">
                <div class="inquiry-metric">
                    <span class="inquiry-metric-label">Outcome:</span>
                    <span class="inquiry-outcome outcome-${analysis.outcome}">${formatOutcome(analysis.outcome)}</span>
                </div>
                <div class="inquiry-metric">
                    <span class="inquiry-metric-label">Response Time:</span>
                    <span class="inquiry-metric-value">${formatResponseTime(analysis.first_response_minutes)}</span>
                </div>
                <div class="inquiry-metric">
                    <span class="inquiry-metric-label">Quality Score:</span>
                    <span class="inquiry-metric-value">${Math.round((analysis.response_quality_score || 0) * 100)}%</span>
                </div>
                <div class="inquiry-metric">
                    <span class="inquiry-metric-label">Est. Lost:</span>
                    <span class="inquiry-metric-value bad">${formatCurrency(analysis.lost_revenue_estimate)}</span>
                </div>
            </div>
            <div class="inquiry-card-reasoning">
                ${escapeHtml(analysis.outcome_reasoning || 'No analysis')}
            </div>
        </div>
        
        ${analysis.team_mistakes && analysis.team_mistakes.length > 0 ? `
        <div class="inquiry-detail-section">
            <h3>🚨 Team Mistakes</h3>
            <div class="recommendations-list">
                ${analysis.team_mistakes.map(m => `
                    <div class="recommendation-item">
                        <span class="recommendation-priority ${m.severity}">${m.severity}</span>
                        <div class="recommendation-content">
                            <div class="recommendation-action">${escapeHtml(m.mistake)}</div>
                            <div class="recommendation-impact">${escapeHtml(m.impact)}</div>
                        </div>
                    </div>
                `).join('')}
            </div>
        </div>
        ` : ''}
        
        ${analysis.team_strengths && analysis.team_strengths.length > 0 ? `
        <div class="inquiry-detail-section">
            <h3>✅ What Went Well</h3>
            <ul class="insight-list">
                ${analysis.team_strengths.map(s => `<li>${escapeHtml(s)}</li>`).join('')}
            </ul>
        </div>
        ` : ''}
        
        ${analysis.recommendations && analysis.recommendations.length > 0 ? `
        <div class="inquiry-detail-section">
            <h3>💡 Recommendations</h3>
            <div class="recommendations-list">
                ${analysis.recommendations.map(r => `
                    <div class="recommendation-item">
                        <span class="recommendation-priority ${r.priority}">${r.priority}</span>
                        <div class="recommendation-content">
                            <div class="recommendation-action">${escapeHtml(r.action)}</div>
                            <div class="recommendation-impact">${escapeHtml(r.expected_impact)}</div>
                        </div>
                    </div>
                `).join('')}
            </div>
        </div>
        ` : ''}
        
        ${analysis.guest_questions && analysis.guest_questions.length > 0 ? `
        <div class="inquiry-detail-section">
            <h3>❓ Guest Questions</h3>
            <ul class="insight-list">
                ${analysis.guest_questions.map(q => `<li>${escapeHtml(q)}</li>`).join('')}
            </ul>
            ${analysis.unanswered_questions && analysis.unanswered_questions.length > 0 ? `
                <h4 style="margin-top: 12px; color: var(--danger);">⚠️ Unanswered:</h4>
                <ul class="insight-list">
                    ${analysis.unanswered_questions.map(q => `<li style="color: var(--danger);">${escapeHtml(q)}</li>`).join('')}
                </ul>
            ` : ''}
        </div>
        ` : ''}
        
        <div class="inquiry-detail-section">
            <h3>💬 Conversation (${messages.length} messages)</h3>
            <div class="inquiry-messages">
                ${messages.map(m => `
                    <div class="inquiry-message ${m.direction}">
                        <div class="inquiry-message-content">${escapeHtml(m.content || '')}</div>
                        <div class="inquiry-message-time">${m.sent_at ? new Date(m.sent_at).toLocaleString() : ''}</div>
                    </div>
                `).join('')}
            </div>
        </div>
    `;
}

function closeInquiryDetail() {
    document.getElementById('inquiry-detail-modal').classList.remove('active');
}

async function refreshInquiries() {
    const container = document.getElementById('inquiry-list-container');
    container.innerHTML = '<div class="loading">Analyzing inquiries... This may take a few minutes.</div>';
    
    try {
        const result = await apiPost('/inquiries/refresh?days_back=60&limit=30');
        
        alert(`✅ Analysis complete!\n\n${result.inquiries_found} inquiries found\n${result.inquiries_analyzed} analyzed\n${result.errors} errors`);
        
        // Reload the data
        await loadInquiryAnalysis();
    } catch (error) {
        console.error('Error refreshing inquiries:', error);
        alert('Failed to analyze inquiries: ' + error.message);
        container.innerHTML = '<div class="error-message">Analysis failed. Please try again.</div>';
    }
}

// Close modal on click outside
document.addEventListener('click', (e) => {
    const modal = document.getElementById('inquiry-detail-modal');
    if (e.target === modal) {
        closeInquiryDetail();
    }
});
