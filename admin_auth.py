"""
Admin authentication helpers
"""
from fastapi import HTTPException, Depends
from typing import Dict, Any
from .admin_utils import is_admin_user

async def get_admin_user(current_user: Dict[str, Any]) -> Dict[str, Any]:
    """
    Verify that the current user has admin privileges
    Args:
        current_user: Current authenticated user from SSO
    Returns:
        Dict: User data if admin, raises HTTPException if not
    Raises:
        HTTPException: If user is not an admin
    """
    if not current_user:
        raise HTTPException(
            status_code=401,
            detail="Authentication required"
        )
    
    user_email = current_user.get("email_id")
    if not user_email:
        raise HTTPException(
            status_code=401,
            detail="User email not found in authentication data"
        )
    
    if not is_admin_user(user_email):
        raise HTTPException(
            status_code=403,
            detail="Admin access required. Contact system administrator."
        )
    
    # Add admin flag to user data
    current_user["is_admin"] = True
    return current_user

def check_admin_access(user_email: str) -> bool:
    """
    Simple check if user has admin access
    Args:
        user_email: User's email address
    Returns:
        bool: True if user is admin, False otherwise
    """
    return is_admin_user(user_email)

def get_admin_permissions(user_email: str) -> Dict[str, bool]:
    """
    Get admin permissions for a user
    Args:
        user_email: User's email address
    Returns:
        Dict: Permission flags
    """
    is_admin = is_admin_user(user_email)
    
    return {
        "can_view_users": is_admin,
        "can_manage_users": is_admin,
        "can_view_documents": is_admin,
        "can_manage_documents": is_admin,
        "can_view_analytics": is_admin,
        "can_manage_system": is_admin,
        "can_view_feedback": is_admin,
        "can_export_data": is_admin
    }