"""
Admin utility functions for database operations
"""
import os
from typing import List, Dict, Optional, Any
from sqlalchemy.orm import Session
from sqlalchemy import func, desc, and_, or_, Integer, cast, Date
from datetime import datetime, timedelta
from .models import User, ChatHistory, Feedback, DocumentTracking

def get_admin_emails() -> List[str]:
    """Get admin emails from environment variable"""
    admin_emails_str = os.getenv("ADMIN_EMAILS", "")
    if not admin_emails_str:
        return []
    return [email.strip() for email in admin_emails_str.split(",") if email.strip()]

def is_admin_user(email: str) -> bool:
    """Check if user email is in admin list"""
    admin_emails = get_admin_emails()
    return email.lower() in [admin_email.lower() for admin_email in admin_emails]

def update_user_admin_status(db: Session, user_id: int, is_admin: bool) -> bool:
    """Update user admin status"""
    try:
        user = db.query(User).filter(User.id == user_id).first()
        if user:
            user.is_admin = is_admin
            db.commit()
            return True
        return False
    except Exception as e:
        db.rollback()
        print(f"Error updating user admin status: {e}")
        return False

# Document Tracking Functions
def get_all_documents(db: Session, skip: int = 0, limit: int = 100, 
                     status_filter: Optional[str] = None,
                     source_filter: Optional[str] = None) -> List[DocumentTracking]:
    """Get all documents with optional filtering"""
    try:
        query = db.query(DocumentTracking)
        
        if status_filter:
            query = query.filter(DocumentTracking.indexing_status == status_filter)
        
        if source_filter:
            query = query.filter(DocumentTracking.source == source_filter)
        
        return query.order_by(desc(DocumentTracking.last_updated_at)).offset(skip).limit(limit).all()
    except Exception as e:
        print(f"Error getting documents: {e}")
        return []

def get_document_by_id(db: Session, doc_id: str) -> Optional[DocumentTracking]:
    """Get document by ID"""
    try:
        return db.query(DocumentTracking).filter(DocumentTracking.id == doc_id).first()
    except Exception as e:
        print(f"Error getting document by ID: {e}")
        return None

def get_document_stats(db: Session) -> Dict[str, Any]:
    """Get document statistics"""
    try:
        total_docs = db.query(DocumentTracking).count()
        
        # Count by status
        status_counts = db.query(
            DocumentTracking.indexing_status,
            func.count(DocumentTracking.id)
        ).group_by(DocumentTracking.indexing_status).all()
        
        # Count by source
        source_counts = db.query(
            DocumentTracking.source,
            func.count(DocumentTracking.id)
        ).group_by(DocumentTracking.source).all()
        
        # Recent processing activity
        recent_processed = db.query(DocumentTracking).filter(
            DocumentTracking.processed_at >= datetime.utcnow() - timedelta(days=7)
        ).count()
        
        # Error count
        error_count = db.query(DocumentTracking).filter(
            DocumentTracking.error_message.isnot(None)
        ).count()
        
        return {
            "total_documents": total_docs,
            "status_breakdown": dict(status_counts),
            "source_breakdown": dict(source_counts),
            "recent_processed": recent_processed,
            "error_count": error_count
        }
    except Exception as e:
        print(f"Error getting document stats: {e}")
        return {}

def update_document_status(db: Session, doc_id: str, status: str, error_message: Optional[str] = None) -> bool:
    """Update document indexing status"""
    try:
        document = db.query(DocumentTracking).filter(DocumentTracking.id == doc_id).first()
        if document:
            document.indexing_status = status
            document.last_updated_at = datetime.utcnow()
            if error_message:
                document.error_message = error_message
            db.commit()
            return True
        return False
    except Exception as e:
        db.rollback()
        print(f"Error updating document status: {e}")
        return False

# User Management Functions
def get_all_users_admin(db: Session, skip: int = 0, limit: int = 100) -> List[Dict[str, Any]]:
    """Get all users with admin information"""
    try:
        users = db.query(User).order_by(desc(User.created_at)).offset(skip).limit(limit).all()
        
        user_data = []
        for user in users:
            # Get user stats
            chat_count = db.query(ChatHistory).filter(ChatHistory.user_id == user.id).count()
            feedback_count = db.query(Feedback).filter(Feedback.user_id == user.id).count()
            
            # Get last activity
            last_chat = db.query(ChatHistory).filter(
                ChatHistory.user_id == user.id
            ).order_by(desc(ChatHistory.timestamp)).first()
            
            user_data.append({
                "id": user.id,
                "username": user.username,
                "email": user.email,
                "is_admin": user.is_admin,
                "created_at": user.created_at,
                "last_active": user.last_active,
                "chat_count": chat_count,
                "feedback_count": feedback_count,
                "last_chat_time": last_chat.timestamp if last_chat else None
            })
        
        return user_data
    except Exception as e:
        print(f"Error getting users admin data: {e}")
        return []

def get_user_activity_stats(db: Session, days: int = 30) -> Dict[str, Any]:
    """Get user activity statistics"""
    try:
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        # Active users (users who chatted in the period)
        active_users = db.query(func.count(func.distinct(ChatHistory.user_id))).filter(
            ChatHistory.timestamp >= cutoff_date
        ).scalar()
        
        # Total messages in period
        total_messages = db.query(ChatHistory).filter(
            ChatHistory.timestamp >= cutoff_date
        ).count()
        
        # New users in period
        new_users = db.query(User).filter(User.created_at >= cutoff_date).count()
        
        # Daily activity - SQL Server compatible
        daily_activity = db.query(
            cast(ChatHistory.timestamp, Date).label('date'),
            func.count(ChatHistory.id).label('message_count'),
            func.count(func.distinct(ChatHistory.user_id)).label('active_users')
        ).filter(
            ChatHistory.timestamp >= cutoff_date
        ).group_by(cast(ChatHistory.timestamp, Date)).order_by('date').all()
        
        return {
            "active_users": active_users or 0,
            "total_messages": total_messages,
            "new_users": new_users,
            "daily_activity": [
                {
                    "date": str(day.date),
                    "message_count": day.message_count,
                    "active_users": day.active_users
                }
                for day in daily_activity
            ]
        }
    except Exception as e:
        print(f"Error getting user activity stats: {e}")
        return {}

def get_feedback_analytics(db: Session, days: int = 30) -> Dict[str, Any]:
    """Get detailed feedback analytics"""
    try:
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        # Basic feedback stats
        total_feedback = db.query(Feedback).filter(Feedback.timestamp >= cutoff_date).count()
        helpful_feedback = db.query(Feedback).filter(
            and_(Feedback.timestamp >= cutoff_date, Feedback.is_helpful == True)
        ).count()
        
        # Rating distribution
        rating_dist = db.query(
            Feedback.rating,
            func.count(Feedback.id)
        ).filter(
            and_(Feedback.timestamp >= cutoff_date, Feedback.rating.isnot(None))
        ).group_by(Feedback.rating).all()
        
        # Category breakdown
        category_breakdown = db.query(
            Feedback.feedback_category,
            func.count(Feedback.id)
        ).filter(
            and_(Feedback.timestamp >= cutoff_date, Feedback.feedback_category.isnot(None))
        ).group_by(Feedback.feedback_category).all()
        
        # Quality metrics - SQL Server compatible
        quality_metrics = db.query(
            func.avg(cast(Feedback.is_accurate, Integer)).label('accuracy_rate'),
            func.avg(cast(Feedback.is_relevant, Integer)).label('relevance_rate'),
            func.avg(cast(Feedback.is_clear, Integer)).label('clarity_rate'),
            func.avg(cast(Feedback.is_complete, Integer)).label('completeness_rate')
        ).filter(Feedback.timestamp >= cutoff_date).first()
        
        return {
            "total_feedback": total_feedback,
            "helpful_feedback": helpful_feedback,
            "helpfulness_rate": (helpful_feedback / total_feedback * 100) if total_feedback > 0 else 0,
            "rating_distribution": dict(rating_dist),
            "category_breakdown": dict(category_breakdown),
            "quality_metrics": {
                "accuracy_rate": float(quality_metrics.accuracy_rate or 0) * 100,
                "relevance_rate": float(quality_metrics.relevance_rate or 0) * 100,
                "clarity_rate": float(quality_metrics.clarity_rate or 0) * 100,
                "completeness_rate": float(quality_metrics.completeness_rate or 0) * 100
            } if quality_metrics else {}
        }
    except Exception as e:
        print(f"Error getting feedback analytics: {e}")
        return {}

def get_system_metrics(db: Session) -> Dict[str, Any]:
    """Get system-wide metrics"""
    try:
        # Total counts
        total_users = db.query(User).count()
        total_chats = db.query(ChatHistory).count()
        total_feedback = db.query(Feedback).count()
        total_documents = db.query(DocumentTracking).count()
        
        # Recent activity (last 24 hours)
        yesterday = datetime.utcnow() - timedelta(days=1)
        recent_chats = db.query(ChatHistory).filter(ChatHistory.timestamp >= yesterday).count()
        recent_users = db.query(func.count(func.distinct(ChatHistory.user_id))).filter(
            ChatHistory.timestamp >= yesterday
        ).scalar()
        
        # Average session length (approximate) - SQL Server compatible
        # First get message counts per session, then calculate average
        session_message_counts = db.query(
            func.count(ChatHistory.id).label('message_count')
        ).group_by(ChatHistory.session_id).subquery()
        
        avg_messages_per_session = db.query(
            func.avg(session_message_counts.c.message_count)
        ).scalar()
        
        return {
            "total_users": total_users,
            "total_chats": total_chats,
            "total_feedback": total_feedback,
            "total_documents": total_documents,
            "recent_chats_24h": recent_chats,
            "recent_active_users_24h": recent_users or 0,
            "avg_messages_per_session": float(avg_messages_per_session or 0)
        }
    except Exception as e:
        print(f"Error getting system metrics: {e}")
        return {}