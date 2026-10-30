"""
Authentication module for TrafficFlow AI.
Implements PBKDF2-HMAC-SHA256 password hashing with salt and session management.
"""
import hashlib
import hmac
import secrets
import datetime
import re
from typing import Optional, Dict, Any, Tuple
from src.database import get_connection, log_event

ITERATIONS = 100_000

def hash_password(password: str, salt: Optional[str] = None) -> Tuple[str, str]:
    """Generates a secure PBKDF2-HMAC-SHA256 hash using a unique random salt."""
    if salt is None:
        salt = secrets.token_hex(16)
    pw_hash = hashlib.pbkdf2_hmac(
        'sha256',
        password.encode('utf-8'),
        salt.encode('utf-8'),
        ITERATIONS
    ).hex()
    return pw_hash, salt

def verify_password(password: str, stored_hash: str, salt: str) -> bool:
    """Verifies a password against the stored PBKDF2 hash using constant-time comparison."""
    computed_hash, _ = hash_password(password, salt)
    return hmac.compare_digest(computed_hash, stored_hash)

def validate_registration_inputs(username: str, email: str, password: str) -> Tuple[bool, str]:
    """Validates user registration fields."""
    if not username or len(username.strip()) < 3:
        return False, "Username must be at least 3 characters."
    if not re.match(r"^[a-zA-Z0-9_.-]+$", username):
        return False, "Username can only contain letters, numbers, underscores, and dashes."
    if not email or "@" not in email or "." not in email:
        return False, "Please enter a valid email address."
    if not password or len(password) < 6:
        return False, "Password must be at least 6 characters long."
    return True, ""

def register_user(username: str, email: str, password: str, full_name: str = "", role: str = "analyst") -> Tuple[bool, str]:
    """Registers a new user in SQLite database."""
    valid, msg = validate_registration_inputs(username, email, password)
    if not valid:
        return False, msg

    conn = get_connection()
    try:
        # Check if username or email already exists
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM users WHERE username = ? OR email = ?", (username.strip().lower(), email.strip().lower()))
        if cursor.fetchone() is not None:
            return False, "Username or email already registered."

        pw_hash, salt = hash_password(password)
        with conn:
            conn.execute(
                """
                INSERT INTO users (username, email, password_hash, salt, full_name, role)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (username.strip().lower(), email.strip().lower(), pw_hash, salt, full_name.strip(), role)
            )
        log_event("AUTH", f"User registered: {username}", username=username)
        return True, "Registration successful! You can now log in."
    except Exception as e:
        return False, f"Registration error: {str(e)}"
    finally:
        conn.close()

def authenticate_user(username: str, password: str) -> Tuple[bool, Optional[Dict[str, Any]], str]:
    """Authenticates username and password against SQLite database."""
    if not username or not password:
        return False, None, "Please provide both username and password."

    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, username, email, password_hash, salt, full_name, role FROM users WHERE username = ?",
            (username.strip().lower(),)
        )
        user = cursor.fetchone()
        if user is None:
            return False, None, "Invalid username or password."

        if not verify_password(password, user["password_hash"], user["salt"]):
            log_event("AUTH", f"Failed login attempt for: {username}", username=username, level="WARNING")
            return False, None, "Invalid username or password."

        # Update last login timestamp
        with conn:
            conn.execute("UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE id = ?", (user["id"],))

        user_dict = {
            "id": user["id"],
            "username": user["username"],
            "email": user["email"],
            "full_name": user["full_name"] or user["username"].capitalize(),
            "role": user["role"]
        }
        log_event("AUTH", f"User authenticated successfully: {username}", username=username)
        return True, user_dict, "Login successful."
    except Exception as e:
        return False, None, f"Database error during authentication: {str(e)}"
    finally:
        conn.close()

def seed_default_users():
    """Seeds default demo accounts if they do not exist."""
    defaults = [
        ("admin", "admin@trafficflow.ai", "Admin@Traffic2026", "Traffic System Admin", "administrator"),
        ("analyst", "analyst@trafficflow.ai", "Analyst@Traffic2026", "Senior Traffic Analyst", "analyst"),
        ("mba_executive", "mba@trafficflow.ai", "Executive@Traffic2026", "MBA Operations Director", "executive")
    ]
    conn = get_connection()
    cursor = conn.cursor()
    for username, email, pwd, full_name, role in defaults:
        cursor.execute("SELECT id FROM users WHERE username = ?", (username,))
        if cursor.fetchone() is None:
            pw_hash, salt = hash_password(pwd)
            with conn:
                conn.execute(
                    """
                    INSERT INTO users (username, email, password_hash, salt, full_name, role)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (username, email, pw_hash, salt, full_name, role)
                )
            log_event("SYSTEM", f"Seeded demo account: {username}")
    conn.close()
