"""
Configuration settings for AI Property Manager.
Loads from environment variables with sensible defaults.
"""

from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Hostify
    HOSTIFY_API_KEY: str = ""
    HOSTIFY_BASE_URL: str = "https://api-rms.hostify.com"
    
    # OpenPhone
    OPENPHONE_API_KEY: str = ""
    OPENPHONE_WEBHOOK_SECRET: str = ""
    OPENPHONE_NUMBER: str = ""  # Your OpenPhone number
    
    # Slack
    SLACK_BOT_TOKEN: str = ""
    SLACK_SIGNING_SECRET: str = ""
    SLACK_CHANNEL_L1: str = "#guest-alerts"
    SLACK_CHANNEL_L2: str = "#guest-urgent"
    SLACK_CHANNEL_ERRORS: str = "#system-alerts"
    
    # OpenAI
    OPENAI_API_KEY: str = ""
    OPENAI_MODEL: str = "gpt-4o"
    
    # Confidence Thresholds
    CONFIDENCE_AUTO_SEND: float = 0.85
    CONFIDENCE_SOFT_ESCALATE: float = 0.70  # Send but flag for review
    
    # Escalation Keywords (comma-separated)
    ESCALATION_KEYWORDS: str = "police,refund,emergency,lawyer,sue,fire,flood,locked out,blood,hurt,cancel,damage,broken"
    
    # Timers (minutes)
    MESSAGE_BURST_DELAY_SECONDS: int = 15
    ESCALATION_L1_TIMEOUT_MINS: int = 5
    ESCALATION_L1_TIMEOUT_AFTER_HOURS_MINS: int = 15
    ESCALATION_L2_TIMEOUT_MINS: int = 10
    
    # Business Hours
    BUSINESS_HOURS_START: int = 9   # 9 AM
    BUSINESS_HOURS_END: int = 21    # 9 PM
    TIMEZONE: str = "America/Chicago"
    
    # Safety
    MAX_OUTBOUND_PER_HOUR: int = 5
    DRY_RUN_MODE: bool = False
    SKIP_WEBHOOK_VERIFICATION: bool = False
    
    # Database
    DATABASE_URL: str = "sqlite:///./property_manager.db"
    
    # RAG / Advanced AI
    ENABLE_RAG: bool = True  # Enable full RAG + agents pipeline
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    INTENT_CLASSIFICATION_MODEL: str = "gpt-4o-mini"  # Fast model for intent
    SUMMARIZATION_MODEL: str = "gpt-4o-mini"  # Fast model for summaries
    CONVERSATION_SUMMARY_THRESHOLD: int = 15  # Messages before summarizing
    
    @property
    def escalation_keywords_list(self) -> List[str]:
        """Parse comma-separated keywords into a list."""
        return [kw.strip().lower() for kw in self.ESCALATION_KEYWORDS.split(",")]
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
