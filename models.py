"""
SQLAlchemy database models for AI Property Manager.
"""

from enum import Enum as PyEnum
from datetime import datetime
from sqlalchemy import (
    Column, Integer, String, Float, Text, DateTime, 
    ForeignKey, Enum, Boolean, create_engine
)
from sqlalchemy.orm import relationship, declarative_base, sessionmaker

from config import settings

# Create base class for models
Base = declarative_base()

# Create engine and session
engine = create_engine(
    settings.DATABASE_URL, 
    connect_args={"check_same_thread": False} if "sqlite" in settings.DATABASE_URL else {}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class ConversationStatus(PyEnum):
    """Status of a conversation with a guest."""
    active = "active"
    escalated_l1 = "escalated_l1"
    escalated_l2 = "escalated_l2"
    snoozed = "snoozed"
    resolved = "resolved"


class MessageDirection(PyEnum):
    """Direction of a message - inbound from guest or outbound from system."""
    inbound = "inbound"
    outbound = "outbound"


class Conversation(Base):
    """
    Represents an ongoing conversation with a guest.
    Tracks status, escalation state, and rate limiting.
    """
    __tablename__ = "conversations"
    
    id = Column(Integer, primary_key=True)
    guest_phone = Column(String(20), index=True, nullable=False)
    guest_name = Column(String(255), nullable=True)
    hostify_reservation_id = Column(String(100), nullable=True, index=True)
    listing_id = Column(String(100), nullable=True)
    listing_name = Column(String(255), nullable=True)
    
    # Reservation dates
    check_in_date = Column(DateTime, nullable=True)
    check_out_date = Column(DateTime, nullable=True)
    booking_source = Column(String(50), nullable=True)  # Airbnb, VRBO, Direct, etc.
    
    status = Column(Enum(ConversationStatus), default=ConversationStatus.active)
    
    # Slack tracking
    slack_thread_ts = Column(String(50), nullable=True)
    slack_channel_id = Column(String(50), nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    last_message_at = Column(DateTime, default=datetime.utcnow)
    last_human_action_at = Column(DateTime, nullable=True)
    
    # Rate limiting
    outbound_count_this_hour = Column(Integer, default=0)
    outbound_hour_started = Column(DateTime, nullable=True)
    
    # Relationship to messages
    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Conversation {self.id} - {self.guest_phone}>"


class Message(Base):
    """
    Individual message within a conversation.
    Tracks content, source, and AI metadata.
    """
    __tablename__ = "messages"
    
    id = Column(Integer, primary_key=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id"))
    
    external_id = Column(String(100), nullable=True)  # For deduplication
    direction = Column(Enum(MessageDirection))
    source = Column(String(20))  # "hostify", "openphone", "slack", "system"
    content = Column(Text)
    attachment_url = Column(String(500), nullable=True)  # Image/file attachments
    
    # AI metadata (for outbound messages)
    ai_confidence = Column(Float, nullable=True)
    ai_reasoning = Column(Text, nullable=True)
    was_auto_sent = Column(Boolean, default=False)
    was_human_edited = Column(Boolean, default=False)
    
    # AI suggestion (for inbound messages - what AI would have replied)
    # This is generated when the message comes in, so we can compare later
    ai_suggested_reply = Column(Text, nullable=True)
    ai_suggestion_confidence = Column(Float, nullable=True)
    ai_suggestion_reasoning = Column(Text, nullable=True)
    ai_suggestion_generated_at = Column(DateTime, nullable=True)
    
    sent_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationship to conversation
    conversation = relationship("Conversation", back_populates="messages")
    
    def __repr__(self):
        return f"<Message {self.id} - {self.direction.value}>"


class GuestIndex(Base):
    """
    Cache of guest reservation data from Hostify.
    Allows lookup by phone number without hitting external API.
    """
    __tablename__ = "guest_index"
    
    id = Column(Integer, primary_key=True)
    guest_phone = Column(String(20), index=True, nullable=True)  # Can be null - matched later
    guest_email = Column(String(255), nullable=True)
    guest_name = Column(String(255))
    reservation_id = Column(String(100), unique=True, index=True)
    listing_id = Column(String(100))
    listing_name = Column(String(255))
    listing_address = Column(Text)
    check_in_date = Column(DateTime)
    check_out_date = Column(DateTime)
    door_code = Column(String(50))
    wifi_name = Column(String(100))
    wifi_password = Column(String(100))
    special_instructions = Column(Text, nullable=True)
    synced_at = Column(DateTime, default=datetime.utcnow)
    
    # Hostify inbox for messaging
    inbox_id = Column(Integer, nullable=True)
    
    # Booking source (Airbnb, VRBO, Direct, etc.)
    source = Column(String(50), nullable=True)
    
    # Status for manual matching
    is_phone_verified = Column(Boolean, default=False)  # True if phone was from API or verified
    
    def __repr__(self):
        return f"<GuestIndex {self.reservation_id} - {self.guest_name}>"


class HostifyThread(Base):
    """
    Hostify inbox threads (conversations).
    Stores thread-level metadata for quick querying by property.
    """
    __tablename__ = "hostify_threads"
    
    id = Column(Integer, primary_key=True)
    thread_id = Column(Integer, unique=True, index=True, nullable=False)  # Hostify's inbox/thread ID
    
    # Property info
    listing_id = Column(String(100), index=True, nullable=True)
    listing_name = Column(String(255), nullable=True)
    
    # Guest info
    guest_name = Column(String(255), nullable=True)
    guest_email = Column(String(255), nullable=True)
    reservation_id = Column(String(100), index=True, nullable=True)
    
    # Dates
    checkin = Column(DateTime, nullable=True)
    checkout = Column(DateTime, nullable=True)
    
    # Thread status
    last_message = Column(Text, nullable=True)
    last_message_at = Column(DateTime, nullable=True)
    message_count = Column(Integer, default=0)
    
    # Sync tracking
    synced_at = Column(DateTime, default=datetime.utcnow)
    messages_synced_at = Column(DateTime, nullable=True)  # When we last synced messages for this thread
    
    # Relationship to messages
    messages = relationship("HostifyMessage", back_populates="thread", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<HostifyThread {self.thread_id} - {self.guest_name} @ {self.listing_name}>"


class HostifyMessage(Base):
    """
    Messages from Hostify inbox.
    Synced from Hostify to provide conversation history.
    """
    __tablename__ = "hostify_messages"
    
    id = Column(Integer, primary_key=True)
    hostify_message_id = Column(Integer, unique=True, index=True)
    inbox_id = Column(Integer, ForeignKey("hostify_threads.thread_id"), index=True)
    reservation_id = Column(String(100), index=True, nullable=True)
    
    direction = Column(String(20))  # "inbound" or "outbound"
    content = Column(Text)
    sender_name = Column(String(255), nullable=True)
    
    # Additional fields for Q&A extraction
    sender_type = Column(String(20), nullable=True)  # "guest", "host", "automatic"
    guest_name = Column(String(255), nullable=True)  # For direction detection
    
    sent_at = Column(DateTime)
    synced_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationship to thread
    thread = relationship("HostifyThread", back_populates="messages")
    
    def __repr__(self):
        return f"<HostifyMessage {self.hostify_message_id}>"


class SystemLog(Base):
    """
    System event log for auditing and debugging.
    Tracks all significant events in the application.
    """
    __tablename__ = "system_logs"
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    event_type = Column(String(50), index=True)
    conversation_id = Column(Integer, nullable=True)
    guest_phone = Column(String(20), nullable=True)
    payload = Column(Text)  # JSON string
    
    def __repr__(self):
        return f"<SystemLog {self.id} - {self.event_type}>"


# Event types for SystemLog:
# message_received, ai_response_generated, auto_sent, draft_created
# escalated_l1, escalated_l2, escalation_timeout
# human_approved, human_edited, human_snoozed, human_resolved
# api_error, cache_miss, force_sync_triggered
# external_reply_detected, unknown_guest, rate_limit_hit
# webhook_received, webhook_verification_failed


class GuestProfile(Base):
    """
    Long-term guest profile built from conversation history.
    Tracks preferences, communication style, and past interactions.
    """
    __tablename__ = "guest_profiles"
    
    id = Column(Integer, primary_key=True)
    guest_phone = Column(String(20), unique=True, index=True, nullable=False)
    guest_email = Column(String(255), nullable=True)
    guest_name = Column(String(255), nullable=True)
    
    # Aggregated stats
    total_stays = Column(Integer, default=0)
    total_conversations = Column(Integer, default=0)
    first_interaction_at = Column(DateTime, nullable=True)
    last_interaction_at = Column(DateTime, nullable=True)
    
    # Inferred preferences (JSON)
    preferences = Column(Text, nullable=True)  # {"early_checkin": true, "quiet_property": true}
    
    # Communication style analysis
    communication_style = Column(String(50), nullable=True)  # "brief", "detailed", "friendly", "formal"
    avg_message_length = Column(Float, nullable=True)
    response_time_preference = Column(String(20), nullable=True)  # "immediate", "patient"
    
    # Sentiment tracking
    overall_sentiment = Column(String(20), default="neutral")  # "positive", "neutral", "negative"
    sentiment_history = Column(Text, nullable=True)  # JSON array of sentiment scores
    
    # Issues and notes
    past_issues = Column(Text, nullable=True)  # JSON array of past issues
    internal_notes = Column(Text, nullable=True)  # Staff notes about this guest
    
    # VIP/special handling
    is_vip = Column(Boolean, default=False)
    special_handling_notes = Column(Text, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f"<GuestProfile {self.guest_phone} - {self.guest_name}>"


class PropertyKnowledge(Base):
    """
    Property-specific knowledge base entries.
    Stores house rules, FAQs, local recommendations, etc.
    """
    __tablename__ = "property_knowledge"
    
    id = Column(Integer, primary_key=True)
    property_id = Column(String(100), index=True, nullable=False)
    property_name = Column(String(255), nullable=True)
    
    # Document classification
    doc_type = Column(String(50), nullable=False)  # house_rules, faq, local_tips, appliance_guide, known_issues
    title = Column(String(255), nullable=False)
    content = Column(Text, nullable=False)
    
    # Metadata
    tags = Column(Text, nullable=True)  # Comma-separated tags
    priority = Column(Integer, default=0)  # Higher = more important
    
    # Tracking
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    created_by = Column(String(100), nullable=True)
    
    # Vector embedding status
    is_embedded = Column(Boolean, default=False)
    embedded_at = Column(DateTime, nullable=True)
    
    def __repr__(self):
        return f"<PropertyKnowledge {self.property_id} - {self.doc_type}: {self.title}>"


class ConversationSummary(Base):
    """
    Summarized version of long conversations.
    Used to provide context without overwhelming the LLM.
    """
    __tablename__ = "conversation_summaries"
    
    id = Column(Integer, primary_key=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id"), index=True)
    
    # Summary content
    summary = Column(Text, nullable=False)
    key_topics = Column(Text, nullable=True)  # JSON array of topics discussed
    resolved_issues = Column(Text, nullable=True)  # JSON array
    pending_issues = Column(Text, nullable=True)  # JSON array
    
    # Guest sentiment in this conversation
    sentiment = Column(String(20), nullable=True)
    
    # Coverage
    messages_summarized = Column(Integer, default=0)
    summary_start_at = Column(DateTime, nullable=True)
    summary_end_at = Column(DateTime, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<ConversationSummary conv={self.conversation_id}>"


class ConversationIntent(Base):
    """
    Detected intents within a conversation.
    Used for routing and analytics.
    """
    __tablename__ = "conversation_intents"
    
    id = Column(Integer, primary_key=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id"), index=True)
    message_id = Column(Integer, ForeignKey("messages.id"), nullable=True)
    
    # Intent classification
    intent = Column(String(50), nullable=False)  # wifi_help, early_checkin, complaint, etc.
    confidence = Column(Float, nullable=True)
    
    # Status
    status = Column(String(20), default="open")  # open, resolved, escalated
    resolved_at = Column(DateTime, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<ConversationIntent {self.intent} - {self.status}>"


class ResponseFeedback(Base):
    """
    Feedback on AI responses for learning.
    Tracks human corrections and their effectiveness.
    """
    __tablename__ = "response_feedback"
    
    id = Column(Integer, primary_key=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id"), index=True)
    message_id = Column(Integer, ForeignKey("messages.id"), nullable=True)
    
    # Original AI response
    original_response = Column(Text, nullable=False)
    original_confidence = Column(Float, nullable=True)
    
    # Correction
    corrected_response = Column(Text, nullable=True)
    correction_type = Column(String(50), nullable=True)  # tone, factual, policy, style
    
    # Outcome
    was_approved = Column(Boolean, default=False)
    was_edited = Column(Boolean, default=False)
    was_rejected = Column(Boolean, default=False)
    
    # Attribution
    human_reviewer = Column(String(100), nullable=True)
    review_time_seconds = Column(Integer, nullable=True)
    
    # Learning status
    is_indexed = Column(Boolean, default=False)  # True if added to vector DB
    indexed_at = Column(DateTime, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<ResponseFeedback conv={self.conversation_id} edited={self.was_edited}>"


class GuestHealthSettings(Base):
    """
    Settings for the Guest Health monitoring feature.
    Stores which properties should be monitored for guest sentiment analysis.
    """
    __tablename__ = "guest_health_settings"
    
    id = Column(Integer, primary_key=True)
    listing_id = Column(String(100), unique=True, index=True, nullable=False)
    listing_name = Column(String(255), nullable=True)
    is_enabled = Column(Boolean, default=True)  # Whether to monitor this property
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f"<GuestHealthSettings {self.listing_id} enabled={self.is_enabled}>"


class SentimentLevel(PyEnum):
    """Guest sentiment/happiness level."""
    very_unhappy = "very_unhappy"  # Likely to leave negative review
    unhappy = "unhappy"  # Has issues, needs attention
    neutral = "neutral"  # No strong signals
    happy = "happy"  # Positive interactions
    very_happy = "very_happy"  # Great experience, likely good review


class GuestHealthAnalysis(Base):
    """
    AI-analyzed guest health data for checked-in guests.
    Stores sentiment analysis, complaints, issues, and recommendations.
    """
    __tablename__ = "guest_health_analysis"
    
    id = Column(Integer, primary_key=True)
    
    # Guest/Reservation identification
    reservation_id = Column(String(100), unique=True, index=True, nullable=False)
    guest_phone = Column(String(20), index=True, nullable=True)
    guest_name = Column(String(255), nullable=True)
    guest_email = Column(String(255), nullable=True)
    
    # Property info
    listing_id = Column(String(100), index=True, nullable=False)
    listing_name = Column(String(255), nullable=True)
    
    # Reservation details
    check_in_date = Column(DateTime, nullable=True)
    check_out_date = Column(DateTime, nullable=True)
    nights_stayed = Column(Integer, nullable=True)
    nights_remaining = Column(Integer, nullable=True)
    reservation_total = Column(Float, nullable=True)
    booking_source = Column(String(50), nullable=True)  # Airbnb, VRBO, Direct, etc.
    
    # AI-analyzed sentiment
    sentiment = Column(Enum(SentimentLevel), default=SentimentLevel.neutral)
    sentiment_score = Column(Float, nullable=True)  # -1.0 (very unhappy) to 1.0 (very happy)
    sentiment_reasoning = Column(Text, nullable=True)
    
    # Complaints/Issues (JSON arrays)
    complaints = Column(Text, nullable=True)  # JSON: [{issue, department, severity, status}]
    unresolved_issues = Column(Text, nullable=True)  # JSON: [{issue, department, urgency}]
    resolved_issues = Column(Text, nullable=True)  # JSON: [{issue, resolution}]
    
    # Response metrics
    total_messages = Column(Integer, default=0)
    guest_messages = Column(Integer, default=0)
    avg_response_time_mins = Column(Float, nullable=True)
    last_message_at = Column(DateTime, nullable=True)
    last_message_from = Column(String(20), nullable=True)  # "guest" or "host"
    
    # Risk assessment
    risk_level = Column(String(20), default="low")  # low, medium, high, critical
    needs_attention = Column(Boolean, default=False)
    attention_reason = Column(Text, nullable=True)
    
    # AI recommendations
    recommended_actions = Column(Text, nullable=True)  # JSON: [{action, priority, reason}]
    
    # Tracking
    last_analyzed_at = Column(DateTime, default=datetime.utcnow)
    conversation_id = Column(Integer, ForeignKey("conversations.id"), nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f"<GuestHealthAnalysis {self.guest_name} @ {self.listing_name} sentiment={self.sentiment.value if self.sentiment else 'unknown'}>"


def init_db():
    """Initialize the database, creating all tables."""
    # Import knowledge_base models so they get created
    try:
        from knowledge_base import PropertyKnowledge, UploadedFile, LearningSession
    except ImportError:
        pass  # knowledge_base not available yet
    
    Base.metadata.create_all(bind=engine)


def get_db():
    """Get a database session. Use as a context manager or dependency."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
