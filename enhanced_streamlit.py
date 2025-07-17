import streamlit as st
import requests
import uuid
import json
from datetime import datetime
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from database_utils import *
from db_utils import get_top_questions
import functools
from dotenv import load_dotenv
from database_utils import AzureSQLConnector
from sqlalchemy.orm import sessionmaker

load_dotenv()

connector = AzureSQLConnector()
engine = connector.create_sqlalchemy_engine()
if not engine:
    raise ConnectionError("Could not create SQLAlchemy engine. Check connection details.")

# Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
db = SessionLocal()

# Streamlit Database Optimization
# Thread pool for database operations to prevent UI blocking
db_thread_pool = ThreadPoolExecutor(max_workers=5, thread_name_prefix="streamlit_db")

# Persistent requests session for better performance or connection pooling
@functools.lru_cache(maxsize=1)
def get_requests_session():
    session = requests.Session()
    session.headers.update({
        'Connection': 'keep-alive',
        'Accept-Encoding': 'gzip, deflate',
        'User-Agent': 'GSC-ARB-Chatbot-Frontend/1.0'
    })
    adapter = requests.adapters.HTTPAdapter(
        pool_connections=10,
        pool_maxsize=20,
        max_retries=3
    )
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session

@st.cache_data(ttl=300)
def get_top_questions_cached(limit=5):
    def _get_questions():
        try:
            thread_db = SessionLocal()
            try:
                return get_top_questions(thread_db, limit=limit)
            finally:
                thread_db.close()
        except Exception as e:
            st.error(f"Error loading FAQs: {e}")
            return []
    
    # Database operation in thread pool
    future = db_thread_pool.submit(_get_questions)
    try:
        # Wait for result with timeout to prevent hanging
        return future.result(timeout=10)
    except Exception as e:
        st.warning(f"Could not load FAQs: {e}")
        return []

@st.cache_data(ttl=60)
def test_api_connection_cached():
    try:
        session = get_requests_session()
        response = session.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200, response.json() if response.status_code == 200 else None
    except requests.exceptions.RequestException as e:
        return False, str(e)

@st.cache_data(ttl=120)
def get_feedback_stats_cached(headers):
    session = get_requests_session()
    for attempt in range(3):
        try:
            stats_response = session.get(f"{API_BASE_URL}/chatbot/feedback/stats", headers=headers, timeout=10)
            if stats_response.status_code == 200:
                return True, stats_response.json()
            else:
                return False, f"Status: {stats_response.status_code}"
        except requests.exceptions.RequestException as e:
            if attempt == 2:
                return False, str(e)
            time.sleep(0.5 * (attempt + 1))
    return False, "Max retries exceeded"

@st.cache_data(ttl=180)
def get_user_feedback_cached(headers):
    try:
        feedback_response = requests.get(f"{API_BASE_URL}/chatbot/feedback/my-feedback", headers=headers, timeout=10)
        if feedback_response.status_code == 200:
            return True, feedback_response.json()
        else:
            return False, f"Status: {feedback_response.status_code}"
    except Exception as e:
        return False, str(e)

@st.cache_data(ttl=30)
def get_system_status_cached(headers):
    try:
        status_response = requests.get(f"{API_BASE_URL}/chatbot/system/status", timeout=10, headers=headers)
        if status_response.status_code == 200:
            return True, status_response.json()
        else:
            return False, f"Status: {status_response.status_code}"
    except Exception as e:
        return False, str(e)

# Admin functionality
def check_admin_access(email):
    """Check if user has admin access"""
    admin_emails_str = os.getenv("ADMIN_EMAILS", "")
    if not admin_emails_str:
        return False
    admin_emails = [email.strip().lower() for email in admin_emails_str.split(",") if email.strip()]
    return email.lower() in admin_emails

@st.cache_data(ttl=60)
def get_admin_dashboard_cached(headers):
    """Get admin dashboard data with caching"""
    try:
        session = get_requests_session()
        response = session.get(f"{API_BASE_URL}/admin/dashboard", headers=headers, timeout=10)
        if response.status_code == 200:
            return True, response.json()
        else:
            return False, f"Status: {response.status_code}"
    except Exception as e:
        return False, str(e)

@st.cache_data(ttl=120)
def get_admin_documents_cached(headers, status_filter=None, source_filter=None):
    """Get admin documents with caching"""
    try:
        session = get_requests_session()
        params = {}
        if status_filter:
            params['status_filter'] = status_filter
        if source_filter:
            params['source_filter'] = source_filter
        
        response = session.get(f"{API_BASE_URL}/admin/documents", headers=headers, params=params, timeout=15)
        if response.status_code == 200:
            return True, response.json()
        else:
            return False, f"Status: {response.status_code}"
    except Exception as e:
        return False, str(e)

@st.cache_data(ttl=180)
def get_admin_users_cached(headers):
    """Get admin users data with caching"""
    try:
        session = get_requests_session()
        response = session.get(f"{API_BASE_URL}/admin/users", headers=headers, timeout=10)
        if response.status_code == 200:
            return True, response.json()
        else:
            return False, f"Status: {response.status_code}"
    except Exception as e:
        return False, str(e)

@st.cache_data(ttl=60)
def get_admin_analytics_cached(headers, days=30):
    """Get admin analytics with caching"""
    try:
        session = get_requests_session()
        response = session.get(f"{API_BASE_URL}/admin/analytics", headers=headers, params={"days": days}, timeout=15)
        if response.status_code == 200:
            return True, response.json()
        else:
            return False, f"Status: {response.status_code}"
    except Exception as e:
        return False, str(e)

# Page config
st.info("Note: Ask Questions related to ARB Process, Platform Provisioning")

# SSO Authentication
import requests
import streamlit as st
from msal import ConfidentialClientApplication
from dotenv import load_dotenv
import os
import extra_streamlit_components as stx

def get_manager():
    cm = stx.CookieManager()
    return cm

if 'cm' not in st.session_state:
    st.session_state.cm = get_manager()
    
CONTROLLER = st.session_state.cm

def initialize_client():
    load_dotenv()
    client_id = os.getenv("AZURE_AD_CLIENT_ID")
    tenant_id = os.getenv("AZURE_AD_TENANT_ID")
    secret = os.getenv("AZURE_AD_CLIENT_SECRET")
    url = f"https://login.microsoftonline.com/{tenant_id}"

    return ConfidentialClientApplication(client_id=client_id,authority=url,client_credential=secret)

def acquire_access_token(app:ConfidentialClientApplication, code, scopes, redirect_uri):
    return app.acquire_token_by_authorization_code(code, scopes=scopes, redirect_uri=redirect_uri)

def fetch_user_data(access_token):
    headers = {"Authorization": f"Bearer {access_token}"}
    graph_api_endpoint = "https://graph.microsoft.com/v1.0/me"
    response = requests.get(graph_api_endpoint, headers=headers)
    return response.json()

# Helper functions
def nav_to(url):
    nav_script = f"<meta http-equiv='refresh' content='0; url={url}'>"
    st.write(nav_script, unsafe_allow_html=True)

def authenticate(app:ConfidentialClientApplication):
    scopes = ["User.Read"]
    redirect_url = os.getenv("AZURE_AD_REDIRECT_URL")
    auth_url = app.get_authorization_request_url(scopes,redirect_uri=redirect_url)
    if len(list(st.query_params)) == 0:
        nav_to(auth_url)

    if st.query_params.get("code"):
        print(list(st.query_params))
        st.session_state["auth_code"] = st.query_params.get("code")
        token_result = acquire_access_token(app, st.session_state.auth_code, scopes, redirect_uri=redirect_url)
        if "access_token" in token_result:
            print(token_result)
            username = token_result['id_token_claims']['name']
            email_id = token_result['id_token_claims']['email']
            access_token = token_result['access_token']
            refresh_token = token_result['refresh_token']
            id_token = token_result['id_token']
            CONTROLLER.set('access_token',access_token)
            st.session_state['username'] =  username
            st.session_state['email_id'] =  email_id
            return True
        else:
            return False

def login():
    print("login")
    app = initialize_client()
    user_data = authenticate(app)
    if user_data:
        print(user_data)
        st.session_state["authenticated"] = True
        redirect_url = os.getenv("REDIRECT_URL")
        st.rerun()

if __name__ == "__main__":
    if 'authenticated' not in st.session_state:
        st.session_state['authenticated'] = False
    if st.session_state['authenticated'] == False:
        login()
    else:
        # Check if user has admin access
        user_email = st.session_state.get('email_id', '')
        is_admin = check_admin_access(user_email)
        
        # Admin Panel UI
        def render_admin_panel():
            """Render the admin panel interface"""
            st.title("🔧 Admin Panel")
            st.markdown("---")
            
            # Get admin token
            token = CONTROLLER.get("access_token")
            if not token:
                st.error("Authentication required for admin access.")
                return
            
            headers = {"Authorization": f"Bearer {token}"}
            
            # Admin navigation tabs
            admin_tab1, admin_tab2, admin_tab3, admin_tab4 = st.tabs([
                "📊 Dashboard", "📄 Documents", "👥 Users", "📈 Analytics"
            ])
            
            with admin_tab1:
                st.subheader("Admin Dashboard")
                
                # Dashboard metrics
                success, dashboard_data = get_admin_dashboard_cached(headers)
                if success:
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Total Users", dashboard_data.get("total_users", 0))
                    with col2:
                        st.metric("Active Sessions", dashboard_data.get("active_sessions", 0))
                    with col3:
                        st.metric("Total Documents", dashboard_data.get("total_documents", 0))
                    with col4:
                        st.metric("System Health", "✅ Healthy" if dashboard_data.get("system_healthy") else "❌ Issues")
                    
                    # Recent activity
                    if dashboard_data.get("recent_activity"):
                        st.subheader("Recent Activity")
                        for activity in dashboard_data["recent_activity"][:10]:
                            st.write(f"• {activity.get('description', 'Unknown activity')} - {activity.get('timestamp', 'Unknown time')}")
                else:
                    st.error(f"Failed to load dashboard: {dashboard_data}")
            
            with admin_tab2:
                st.subheader("Document Management")
                
                # Document filters
                col1, col2, col3 = st.columns([2, 2, 1])
                with col1:
                    status_filter = st.selectbox(
                        "Filter by Status",
                        ["All", "processed", "processing", "failed", "pending"],
                        key="doc_status_filter"
                    )
                with col2:
                    source_filter = st.text_input("Filter by Source", key="doc_source_filter")
                with col3:
                    if st.button("Refresh Documents"):
                        get_admin_documents_cached.clear()
                
                # Get documents
                status_param = None if status_filter == "All" else status_filter
                source_param = source_filter if source_filter else None
                
                success, docs_data = get_admin_documents_cached(headers, status_param, source_param)
                if success:
                    documents = docs_data.get("documents", [])
                    st.write(f"**Total Documents:** {len(documents)}")
                    
                    if documents:
                        # Document table
                        for doc in documents:
                            with st.expander(f"📄 {doc.get('title', 'Untitled')} - {doc.get('indexing_status', 'Unknown')}"):
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.write(f"**URL:** {doc.get('page_url', 'N/A')}")
                                    st.write(f"**Source:** {doc.get('source', 'N/A')}")
                                    st.write(f"**Status:** {doc.get('indexing_status', 'N/A')}")
                                with col2:
                                    st.write(f"**Processed:** {doc.get('processed_at', 'N/A')}")
                                    st.write(f"**Updated:** {doc.get('last_updated_at', 'N/A')}")
                                    if doc.get('error_message'):
                                        st.error(f"**Error:** {doc['error_message']}")
                                
                                # Document actions
                                doc_col1, doc_col2 = st.columns(2)
                                with doc_col1:
                                    if st.button(f"Reprocess", key=f"reprocess_{doc.get('id')}"):
                                        try:
                                            response = requests.post(
                                                f"{API_BASE_URL}/admin/documents/{doc['id']}/reprocess",
                                                headers=headers,
                                                timeout=30
                                            )
                                            if response.status_code == 200:
                                                st.success("Document reprocessing initiated")
                                                get_admin_documents_cached.clear()
                                                st.rerun()
                                            else:
                                                st.error(f"Failed to reprocess: {response.text}")
                                        except Exception as e:
                                            st.error(f"Error: {str(e)}")
                                
                                with doc_col2:
                                    if st.button(f"Delete", key=f"delete_{doc.get('id')}"):
                                        if st.session_state.get(f"confirm_delete_{doc.get('id')}"):
                                            try:
                                                response = requests.delete(
                                                    f"{API_BASE_URL}/admin/documents/{doc['id']}",
                                                    headers=headers,
                                                    timeout=30
                                                )
                                                if response.status_code == 200:
                                                    st.success("Document deleted")
                                                    get_admin_documents_cached.clear()
                                                    st.rerun()
                                                else:
                                                    st.error(f"Failed to delete: {response.text}")
                                            except Exception as e:
                                                st.error(f"Error: {str(e)}")
                                        else:
                                            st.session_state[f"confirm_delete_{doc.get('id')}"] = True
                                            st.warning("Click again to confirm deletion")
                    else:
                        st.info("No documents found matching the filters.")
                else:
                    st.error(f"Failed to load documents: {docs_data}")
            
            with admin_tab3:
                st.subheader("User Management")
                
                # User statistics
                success, users_data = get_admin_users_cached(headers)
                if success:
                    users = users_data.get("users", [])
                    user_stats = users_data.get("user_stats", {})
                    
                    # User metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Users", len(users))
                    with col2:
                        st.metric("Active Today", user_stats.get("active_today", 0))
                    with col3:
                        st.metric("Admin Users", sum(1 for u in users if u.get("is_admin")))
                    
                    # User list
                    st.subheader("User List")
                    for user in users:
                        with st.expander(f"👤 {user.get('display_name', 'Unknown')} {'(Admin)' if user.get('is_admin') else ''}"):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write(f"**Email:** {user.get('email', 'N/A')}")
                                st.write(f"**Created:** {user.get('created_at', 'N/A')}")
                                st.write(f"**Last Active:** {user.get('last_active_at', 'N/A')}")
                            with col2:
                                st.write(f"**Total Sessions:** {user.get('total_sessions', 0)}")
                                st.write(f"**Total Messages:** {user.get('total_messages', 0)}")
                                st.write(f"**Admin Status:** {'Yes' if user.get('is_admin') else 'No'}")
                else:
                    st.error(f"Failed to load users: {users_data}")
            
            with admin_tab4:
                st.subheader("System Analytics")
                
                # Analytics time range
                days = st.selectbox("Time Range", [7, 30, 90], index=1, key="analytics_days")
                
                success, analytics_data = get_admin_analytics_cached(headers, days)
                if success:
                    # Usage metrics
                    st.subheader("Usage Metrics")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Total Chats", analytics_data.get("total_chats", 0))
                    with col2:
                        st.metric("Unique Users", analytics_data.get("unique_users", 0))
                    with col3:
                        st.metric("Avg Session Length", f"{analytics_data.get('avg_session_length', 0):.1f} min")
                    
                    # Feedback analytics
                    feedback_stats = analytics_data.get("feedback_stats", {})
                    if feedback_stats:
                        st.subheader("Feedback Analytics")
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Total Feedback", feedback_stats.get("total_feedback", 0))
                        with col2:
                            st.metric("Helpfulness Rate", f"{feedback_stats.get('helpfulness_rate', 0):.1f}%")
                        with col3:
                            st.metric("Average Rating", f"{feedback_stats.get('average_rating', 0):.1f}/5")
                    
                    # Top questions
                    top_questions = analytics_data.get("top_questions", [])
                    if top_questions:
                        st.subheader("Most Asked Questions")
                        for i, (question, count) in enumerate(top_questions[:10], 1):
                            st.write(f"{i}. {question} ({count} times)")
                    
                    # Error analytics
                    error_stats = analytics_data.get("error_stats", {})
                    if error_stats:
                        st.subheader("Error Statistics")
                        st.write(f"**Total Errors:** {error_stats.get('total_errors', 0)}")
                        st.write(f"**Error Rate:** {error_stats.get('error_rate', 0):.2f}%")
                else:
                    st.error(f"Failed to load analytics: {analytics_data}")
        
        # Main app logic - show admin panel or regular chat
        if is_admin and st.sidebar.button("🔧 Admin Panel", use_container_width=True):
            st.session_state.show_admin = True
        
        if st.session_state.get('show_admin') and is_admin:
            # Show admin panel
            if st.sidebar.button("💬 Back to Chat", use_container_width=True):
                st.session_state.show_admin = False
                st.rerun()
            render_admin_panel()
        else:
            # Regular chat interface
            # FAQs
            st.sidebar.header("FAQs")
            try:
                # Cached version that runs in background thread
                top_questions = get_top_questions_cached(limit=5)
                for q, cnt in top_questions:
                    if st.sidebar.button(f"{q[:50]}... ({cnt}× asked)", key=f"faq_{hash(q)}"):
                        st.session_state.question = q
            except Exception as e:
                st.sidebar.warning("FAQs temporarily unavailable")

        # Backend-only session creation function
        def create_new_session(reset_chat_state=False):
            """
            Backend-only session creation - maintains backend as single source of truth
            Args:
                reset_chat_state (bool): Whether to reset chat messages and feedback
            Returns:
                str: The created session ID from backend, or None if failed
            """
            try:
                token = CONTROLLER.get("access_token")
                if not token:
                    st.error("Authentication required. Please refresh and log in again.")
                    st.stop()
                
                headers = {"Authorization": f"Bearer {token}"}
                session = get_requests_session()
                response = session.post(f"{API_BASE_URL}/sessions", headers=headers, timeout=10)
                
                if response.status_code == 200:
                    result = response.json()
                    new_session_id = result["session_id"]
                    print(f"Frontend: Created new session via backend API: {new_session_id}")
                    
                    # Update session state
                    st.session_state.session_id = new_session_id
                    if reset_chat_state:
                        st.session_state.messages = []
                        st.session_state.feedback_given = {}
                    
                    return new_session_id
                else:
                    # Backend session creation failed
                    error_msg = f"Backend session creation failed: HTTP {response.status_code}"
                    try:
                        error_detail = response.json().get('detail', response.text)
                        error_msg += f" - {error_detail}"
                    except:
                        error_msg += f" - {response.text}"
                    
                    print(f"Frontend: {error_msg}")
                    st.error(f"Unable to create session. {error_msg}")
                    st.error("Please check if the backend service is running and try again.")
                    st.stop()
                    
            except requests.exceptions.RequestException as e:
                print(f"Frontend: Network error creating session: {e}")
                st.error(f"Network error: Unable to connect to backend service.")
                st.error("Please check your connection and ensure the backend is running.")
                st.stop()
            except Exception as e:
                print(f"Frontend: Unexpected error creating session: {e}")
                st.error(f"Unexpected error creating session: {str(e)}")
                st.error("Please refresh the page and try again.")
                st.stop()

        # Initialize session state using consolidated function
        if "session_id" not in st.session_state:
            create_new_session(reset_chat_state=False)

        if "messages" not in st.session_state:
            st.session_state.messages = []

        if "feedback_given" not in st.session_state:
            st.session_state.feedback_given = {}
        
        print(f"Frontend: Using session_id: {st.session_state.session_id}")

        # Memory optimization by limiting message history to prevent memory issues
        def optimize_message_history():
            MAX_MESSAGES = 100
            if len(st.session_state.messages) > MAX_MESSAGES:
                st.session_state.messages = st.session_state.messages[-MAX_MESSAGES:]
                valid_message_ids = {f"msg_{i}" for i in range(len(st.session_state.messages))}
                st.session_state.feedback_given = {
                    k: v for k, v in st.session_state.feedback_given.items() if k in valid_message_ids
                }

        # Helper function to test API connection (uses cached version)
        def test_api_connection():
            # Wrapper that uses cached version for better performance
            return test_api_connection_cached()

        # Helper function to validate SSO token with backend
        @st.cache_data(ttl=300)
        def validate_sso_token_with_backend(token):
            try:
                session = get_requests_session()
                headers = {"Authorization": f"Bearer {token}"}
                response = session.get(f"{API_BASE_URL}/sso/validate-token", headers=headers, timeout=10)
                if response.status_code == 200:
                    return True, response.json()
                else:
                    return False, f"Status: {response.status_code}"
            except Exception as e:
                return False, str(e)

        # Enhanced Feedback UI
        def render_feedback_ui(message_id, message_index=None):
            # Render feedback UI for a specific message
            if message_id in st.session_state.feedback_given:
                st.success("Feedback submitted! Thank you..!!")
                return
            
            # Get the message data
            if message_index is not None and message_index < len(st.session_state.messages):
                message = st.session_state.messages[message_index]
                response_id = message.get("response_id")
                chat_history_id = message.get("chat_history_id")
            else:
                response_id = None
                chat_history_id = None
            
            st.markdown("---")
            st.markdown("**Was this response helpful?**")
            
            # Create unique keys for this message
            col1, col2, col3 = st.columns([1, 1, 2])
            
            with col1:
                if st.button("👍 Helpful", key=f"up_{message_id}"):
                    submit_feedback(
                        response_id=response_id,
                        chat_history_id=chat_history_id,
                        is_helpful=True,
                        message_id=message_id
                    )
            
            with col2:
                if st.button("👎 Not Helpful", key=f"down_{message_id}"):
                    submit_feedback(
                        response_id=response_id,
                        chat_history_id=chat_history_id,
                        is_helpful=False,
                        message_id=message_id
                    )
            
            # Detailed feedback form
            with st.expander("Provide detailed feedback (optional)"):
                rating = st.select_slider(
                    "Rate this response (1-5 stars)",
                    options=[1, 2, 3, 4, 5],
                    value=3,
                    key=f"rating_{message_id}"
                )
                
                feedback_category = st.selectbox(
                    "What aspect needs improvement?",
                    ["accuracy", "helpfulness", "clarity", "completeness", "relevance", "other"],
                    key=f"category_{message_id}"
                )
                
                feedback_text = st.text_area(
                    "Additional comments",
                    placeholder="Tell us how we can improve...",
                    key=f"text_{message_id}"
                )
                
                # Detailed ratings
                col_acc, col_rel, col_clear, col_comp = st.columns(4)
                with col_acc:
                    is_accurate = st.checkbox("Accurate", key=f"acc_{message_id}")
                with col_rel:
                    is_relevant = st.checkbox("Relevant", key=f"rel_{message_id}")
                with col_clear:
                    is_clear = st.checkbox("Clear", key=f"clear_{message_id}")
                with col_comp:
                    is_complete = st.checkbox("Complete", key=f"comp_{message_id}")
                
                if st.button("Submit Detailed Feedback", key=f"submit_{message_id}"):
                    submit_feedback(
                        response_id=response_id,
                        chat_history_id=chat_history_id,
                        rating=rating,
                        is_helpful=None,
                        feedback_text=feedback_text,
                        feedback_category=feedback_category,
                        is_accurate=is_accurate,
                        is_relevant=is_relevant,
                        is_clear=is_clear,
                        is_complete=is_complete,
                        message_id=message_id
                    )

        def submit_feedback(response_id=None, chat_history_id=None, message_id=None, **feedback_data):
            # Submit feedback to the API with enhanced error handling
            try:
                # Debug logging
                print(f"Submitting feedback - response_id: {response_id}, chat_history_id: {chat_history_id}")
                
                # Retrieve token from the cookie
                token = CONTROLLER.get("access_token")
                if not token:
                    st.error("Session expired. Please log in again.")
                    st.stop()

                # Add the  token to the headers
                headers = {"Authorization": f"Bearer {token}"}
                
                # Enhanced logic to find response identifiers
                if not response_id and not chat_history_id:
                    # Try to find from the most recent assistant message
                    for msg in reversed(st.session_state.messages):
                        if msg["role"] == "assistant":
                            response_id = msg.get("response_id")
                            chat_history_id = msg.get("chat_history_id")
                            if response_id or chat_history_id:
                                print(f"Found identifiers from recent message: response_id={response_id}, chat_history_id={chat_history_id}")
                                break
                
                feedback_payload = {
                    "response_id": response_id,
                    "chat_history_id": chat_history_id,
                    "session_id": st.session_state.session_id,  # Add session_id
                    **feedback_data
                }
                
                # Remove message_id from payload as it's only for UI state
                feedback_payload.pop("message_id", None)
                
                # Remove None values to avoid sending unnecessary data
                feedback_payload = {k: v for k, v in feedback_payload.items() if v is not None}
                
                # If no identifiers at all, we can add a flag for backend to use the latest chat
                if not response_id and not chat_history_id:
                    feedback_payload["use_latest_chat"] = True
                
                print(f"Final feedback payload: {feedback_payload}")
                
                response = requests.post(
                    f"{API_BASE_URL}/chatbot/feedback",
                    json=feedback_payload,
                    headers=headers,
                    timeout=30
                )
                
                if response.status_code == 200:
                    st.session_state.feedback_given[message_id] = True
                    st.success("Thank you for your feedback!")
                    st.rerun()
                else:
                    error_detail = "Unknown error"
                    try:
                        if response.headers.get('content-type', '').startswith('application/json'):
                            error_json = response.json()
                            error_detail = error_json.get('detail', response.text)
                        else:
                            error_detail = response.text
                    except:
                        error_detail = f"HTTP {response.status_code}: {response.reason}"
                    
                    st.error(f"Failed to submit feedback: {error_detail}")
                    print(f"Feedback submission failed: {response.status_code} - {error_detail}")
                    
            except requests.exceptions.RequestException as e:
                st.error(f"Network error submitting feedback: {str(e)}")
                print(f"Network error: {e}")
            except Exception as e:
                st.error(f"Error submitting feedback: {str(e)}")
                print(f"Unexpected error: {e}")

        # Main UI
        st.title("GSC ARB Chatbot")   

        # Validate SSO token with backend
        token = CONTROLLER.get("access_token")
        if token:
            token_valid, token_data = validate_sso_token_with_backend(token)
            if not token_valid:
                st.error("SSO token validation failed. Please refresh and login again.")
                st.stop()

        # Connection status
        col1, col2 = st.columns([3, 1])
        with col1:
            st.write(f"**Session ID:** `{st.session_state.session_id}`")
            if "username" in st.session_state:
                st.write(f"**User:** {st.session_state.username}")
            if "email_id" in st.session_state:
                st.write(f"**Email:** {st.session_state.email_id}")

        with col2:
            is_connected, health_data = test_api_connection()
            if is_connected:
                st.success("ARB Chatbot API Connected")
            else:
                st.error("API Disconnected")

        # Display chat messages
        for i, message in enumerate(st.session_state.messages):
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if "timestamp" in message:
                    st.caption(f"*{message['timestamp']}*")
                
                # Feedback UI for assistant messages
                if message["role"] == "assistant":
                    render_feedback_ui(
                        message_id=f"msg_{i}",
                        message_index=i
                    )

        # Chat input
        st.toast("Welcome..!!")
        if prompt := st.chat_input("What's on your mind?"):
            if not test_api_connection()[0]:
                st.error("Cannot connect to ARB Chatbot API.")
                st.stop()

            # Add user message to session state
            user_message = {
                "role": "user",
                "content": prompt,
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "session_id": st.session_state.session_id  # Store session_id for reference
            }
            st.session_state.messages.append(user_message)
            optimize_message_history()
            with st.chat_message("user"):
                st.markdown(prompt)
                st.caption(f"*{user_message['timestamp']}*")

            # Get response from API
            with st.chat_message("assistant"):
                message_placeholder = st.empty()

                with st.spinner("ARB Chatbot is Generating Response..."):
                    try:
                        # Retrieve token from the cookie
                        token = CONTROLLER.get("access_token")
                        if not token:
                            st.error("Session expired. Please log in again.")
                            st.stop()

                        # Add the token to the headers
                        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

                        # Prepare request data - CRITICAL: Always send session_id
                        request_data = {
                            "message": prompt,
                            "session_id": st.session_state.session_id
                        }
                        
                        print(f"Frontend: Sending chat request with session_id: {st.session_state.session_id}")

                        # Persistent session for better performance
                        session = get_requests_session()
                        response = session.post(
                            f"{API_BASE_URL}/chat",
                            json=request_data,
                            timeout=120,
                            headers=headers
                        )

                        if response.status_code == 200:
                            result = response.json()
                            assistant_response = result["response"]
                            response_id = result.get("request_id") or result.get("response_id")
                            chat_history_id = result.get("chat_history_id")
                            returned_session_id = result.get("session_id")

                            # Verify session_id consistency
                            if returned_session_id and returned_session_id != st.session_state.session_id:
                                print(f"Frontend: Session ID mismatch! Sent: {st.session_state.session_id}, Received: {returned_session_id}")
                            else:
                                print(f"Frontend: Session ID consistent: {st.session_state.session_id}")

                            # Generate a response_id if not provided by API
                            if not response_id:
                                response_id = str(uuid.uuid4())
                                print(f"Generated fallback response_id: {response_id}")

                            # Extract token usage details
                            prompt_tokens = result.get("prompt_tokens")
                            completion_tokens = result.get("completion_tokens")
                            total_tokens = result.get("total_tokens")
                            total_cost = result.get("total_cost")

                            # Clear placeholder and show response
                            message_placeholder.empty()
                            st.markdown(assistant_response)
                            st.info(f"prompt_tokens={prompt_tokens}, completion_tokens={completion_tokens}, total_tokens={total_tokens}, total_cost={total_cost}")

                            # Add assistant response to chat history with identifiers
                            assistant_message = {
                                "role": "assistant",
                                "content": assistant_response,
                                "timestamp": datetime.now().strftime("%H:%M:%S"),
                                "response_id": response_id,
                                "chat_history_id": chat_history_id,
                                "session_id": st.session_state.session_id  # Store session_id for reference
                            }
                            st.session_state.messages.append(assistant_message)
                            st.caption(f"*{assistant_message['timestamp']}*")

                            # Show feedback UI for this new message
                            message_id = f"msg_{len(st.session_state.messages) - 1}"
                            render_feedback_ui(
                                message_id=message_id,
                                message_index=len(st.session_state.messages) - 1
                            )
                            
                            print(f"Frontend: Chat completed for session {st.session_state.session_id}, total messages: {len(st.session_state.messages)}")

                        elif response.status_code == 429:
                            message_placeholder.error("Rate limit exceeded. Please wait before sending another message.")
                        elif response.status_code == 503:
                            message_placeholder.warning("ARB Chatbot Server busy. Request queued for processing.")
                        else:
                            error_detail = response.text
                            try:
                                error_json = response.json()
                                error_detail = error_json.get("detail", error_detail)
                            except:
                                pass
                            message_placeholder.error(f"Error {response.status_code}: {error_detail}. Please refresh the page and try again.")

                    except requests.exceptions.Timeout:
                        message_placeholder.error("Request timed out after 120 seconds. Please re-submit your query.")
                    except requests.exceptions.ConnectionError:
                        message_placeholder.error("Connection error. Is the FastAPI server running on port 8000?")
                    except requests.exceptions.RequestException as e:
                        message_placeholder.error(f"Request error: {str(e)}")

        # Sidebar
        with st.sidebar:
            # Admin status indicator
            if is_admin:
                st.success("🔧 Admin Access Enabled")
                st.write(f"**Admin Email:** {user_email}")
                st.divider()
            
            st.header("System Controls")
            if st.button("Refresh Status"):
                get_system_status_cached.clear()
            try:
                success, status_data = get_system_status_cached(headers={"Authorization": f"Bearer {CONTROLLER.get('access_token')}"})
                if success:
                    st.success("Chatbot System Status")
                    st.json(status_data)
                else:
                    st.error(f"Status check failed: {status_data}")
            except Exception as e:
                st.error(f"Could not fetch status: {e}")
            
            st.divider()
            
            # Feedback Statistics - Optimized with caching
            st.subheader("Feedback Stats")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("View My Feedback", use_container_width=True):
                    # Clear cache for fresh data when explicitly requested
                    get_user_feedback_cached.clear()
            
            with col2:
                if st.button("Overall Stats", use_container_width=True):
                    # Clear cache for fresh data when explicitly requested
                    get_feedback_stats_cached.clear()
            
            # Show cached feedback data
            try:
                headers = {"Authorization": f"Bearer {CONTROLLER.get('access_token')}"}
                success, feedback_data = get_user_feedback_cached(headers)
                if success and feedback_data.get("feedback_history"):
                    st.subheader("Your Feedback History")
                    for feedback in feedback_data["feedback_history"][:3]:  # Show last 3 for better performance
                        with st.expander(f"Feedback from {feedback['timestamp'][:10]}"):
                            st.write(f"**Rating:** {feedback.get('rating', 'N/A')}")
                            st.write(f"**Helpful:** {'Yes' if feedback.get('is_helpful') else 'No' if feedback.get('is_helpful') is False else 'N/A'}")
                            if feedback.get('feedback_text'):
                                st.write(f"**Comment:** {feedback['feedback_text']}")
                            if feedback.get('chat_question'):
                                st.write(f"**Question:** {feedback['chat_question'][:100]}...")
                elif success:
                    st.info("No feedback history found.")
            except Exception as e:
                st.warning("Feedback history temporarily unavailable")
            
            # Show cached stats
            try:
                headers = {"Authorization": f"Bearer {CONTROLLER.get('access_token')}"}
                success, stats = get_feedback_stats_cached(headers)
                if success:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Feedback", stats.get("total_feedback", 0))
                    with col2:
                        st.metric("Helpfulness Rate", f"{stats.get('helpfulness_rate', 0):.1f}%")
                    with col3:
                        if stats.get("average_rating"):
                            st.metric("Average Rating", f"{stats['average_rating']:.1f}/5")
            except Exception as e:
                st.warning("Stats temporarily unavailable")
            
            st.divider()
            
            # Session Controls
            if st.button("Clear Session", use_container_width=True):
                st.session_state.messages = []
                st.session_state.feedback_given = {}
                st.success("Session cleared!")
                st.rerun()
            
            if st.button("New Session", use_container_width=True):
                # Use backend-only session creation function
                try:
                    new_session_id = create_new_session(reset_chat_state=True)
                    if new_session_id:  # Only proceed if session was successfully created
                        st.success(f"New session created: {new_session_id[:8]}...")
                        st.rerun()
                except Exception as e:
                    # Error handling is already done in create_new_session function
                    # This catch is just to prevent the app from crashing
                    print(f"New session creation failed: {e}")
            
            st.divider()
            
            # API Health Check
            st.subheader("ARB Chatbot API Health")
            health_status, health_info = test_api_connection()
            
            if health_status:
                st.success("ARB Chatbot API is healthy")
                if health_info:
                    with st.expander("Health Details"):
                        st.json(health_info)
            else:
                st.error("ARB Chatbot API is down")
                st.error(f"Error: {health_info}")
                
                st.subheader("Troubleshooting Steps:")
                st.markdown("""
                1. **Check with Team ARB**
                --> POC: Harshit, Yogesh, Sri (Line Manager)
                """)
            
            st.divider()
            
            # Session Info
            st.subheader("Session Info")
            st.write(f"**Messages:** {len(st.session_state.messages)}")
            st.write(f"**Session ID:** `{st.session_state.session_id[:8]}...`")
            
            # Download chat history
            if st.session_state.messages:
                chat_history = "\n\n".join([
                    f"**{msg['role'].title()}** ({msg.get('timestamp', 'N/A')}):\n{msg['content']}"
                    for msg in st.session_state.messages
                ])
                
                st.download_button(
                    label="Download Chat",
                    data=chat_history,
                    file_name=f"chat_history_{st.session_state.session_id[:8]}.txt",
                    mime="text/plain",
                    use_container_width=True
                )

        # Footer
        st.markdown("---")
        st.markdown("**GSC ARB Chatbot** - Team ARB")

        # Cleanup function
        import atexit

        def cleanup_resources():
            try:
                db_thread_pool.shutdown(wait=False)
                if db:
                    db.close()
            except:
                pass

        # Register cleanup function
        atexit.register(cleanup_resources)