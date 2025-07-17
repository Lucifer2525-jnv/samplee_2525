from sqlalchemy import Column, Integer, String, Text, DateTime, Boolean, ForeignKey, create_engine, JSON
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from datetime import datetime
from sqlalchemy.pool import StaticPool
import os
import sys
sys.path.append(os.path.dirname(__file__))

Base = declarative_base()

class User(Base):
    __tablename__ = "users_gsc_chatbot"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    email = Column(String(255), unique=True, index=True, nullable=True)  # Added email field for SSO
    microsoft_id = Column(String(255), unique=True, index=True, nullable=True)  # Microsoft SSO ID
    hashed_password = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_active = Column(DateTime, default=datetime.utcnow)  # Added for tracking
    is_admin = Column(Boolean, default=False, nullable=False)  # Admin role flag
    # Relationships
    chat_histories = relationship("ChatHistory", back_populates="user")
    feedbacks = relationship("Feedback", back_populates="user")

class ChatHistory(Base):
    __tablename__ = "chat_history_gsc_chatbot"
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(100), index=True, nullable=False)
    user_id = Column(Integer, ForeignKey("users_gsc_chatbot.id"), index=True, nullable=False)
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    response_id = Column(String(36), unique=True, index=True, nullable=True)  #UUID for feedback linking
    # Relationships
    user = relationship("User", back_populates="chat_histories")
    feedbacks = relationship("Feedback", back_populates="chat_history")

class Feedback(Base):
    __tablename__ = "feedback_gsc_chatbot"
    id = Column(Integer, primary_key=True, index=True)
    chat_history_id = Column(Integer, ForeignKey("chat_history_gsc_chatbot.id"), nullable=False)
    user_id = Column(Integer, ForeignKey("users_gsc_chatbot.id"), nullable=False)
    session_id = Column(String(100), index=True, nullable=False)
    # Feedback types
    rating = Column(Integer, nullable=True)  #1-5 star rating
    is_helpful = Column(Boolean, nullable=True)  #thumbs up/down
    # Detailed feedback
    feedback_text = Column(Text, nullable=True)
    feedback_category = Column(String(50), nullable=True)  #accuracy, helpfulness, clarity, etc.
    # Specific issues
    is_accurate = Column(Boolean, nullable=True)
    is_relevant = Column(Boolean, nullable=True)
    is_clear = Column(Boolean, nullable=True)
    is_complete = Column(Boolean, nullable=True)
    # Metadata
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    ip_address = Column(String(45), nullable=True)
    user_agent = Column(String(500), nullable=True)
    # Relationships
    user = relationship("User", back_populates="feedbacks")
    chat_history = relationship("ChatHistory", back_populates="feedbacks")

class DocumentTracking(Base):
    __tablename__ = "documents_tracking"
    id = Column(String, primary_key=True)
    page_url = Column(String, unique=True, nullable=False)
    title = Column(String)
    source = Column(String, default="unknown")
    processed_at = Column(DateTime)
    last_updated_at = Column(DateTime)
    indexing_status = Column(String)
    error_message = Column(String)

class SafetyLog(Base):
    __tablename__ = "safety_logs"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users_gsc_chatbot.id"), nullable=True)
    session_id = Column(String(100), index=True, nullable=True)
    chat_history_id = Column(Integer, ForeignKey("chat_history_gsc_chatbot.id"), nullable=True)
    
    # Safety event details
    event_type = Column(String(50), nullable=False)  # 'content_blocked', 'pii_detected', 'response_filtered'
    original_text = Column(Text, nullable=True)
    processed_text = Column(Text, nullable=True)
    
    # Content Safety results
    content_safety_result = Column(JSON, nullable=True)
    content_blocked = Column(Boolean, default=False)
    blocked_categories = Column(JSON, nullable=True)
    
    # PII Detection results
    pii_detection_result = Column(JSON, nullable=True)
    pii_detected = Column(Boolean, default=False)
    pii_entities = Column(JSON, nullable=True)
    
    # Metadata
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    ip_address = Column(String(45), nullable=True)
    user_agent = Column(String(500), nullable=True)
    
    # Relationships
    user = relationship("User")
    chat_history = relationship("ChatHistory")

# Database configuration
# DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./chatbot.db")

# # Use environment-based database configuration
# engine = create_engine(
#    DATABASE_URL,
#    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {},
#    poolclass=StaticPool if "sqlite" in DATABASE_URL else None,
#    echo=False,  # Set to True for SQL debugging
# )

# SessionLocal = sessionmaker(
#    bind=engine,
#    autoflush=False,
#    autocommit=False
# )

# def get_db():
#    db = SessionLocal()
#    try:
#        yield db
#    except Exception as e:
#        db.rollback()
#        raise e
#    finally:
#        db.close()

from database_utils import AzureSQLConnector
from sqlalchemy.orm import sessionmaker

connector = AzureSQLConnector()
engine = connector.create_sqlalchemy_engine()
if not engine:
    raise ConnectionError("Could not create SQLAlchemy engine. Check connection details.")

# engine = create_engine(
#    DATABASE_URL,
#    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {},
#    poolclass=StaticPool if "sqlite" in DATABASE_URL else None,
#    echo=False,  # Set to True for SQL debugging
# )

SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False
)

def get_db():
    db = SessionLocal()
    try:
        yield db
    except Exception as e:
        db.rollback()
        raise e
    finally:
        db.close()