"""
SQLite-based authentication utilities for the Flipkart hackathon project.
Lightweight, zero-configuration authentication system.
"""
import sqlite3
import bcrypt
import secrets
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Tuple
import os
from pathlib import Path


class SQLiteAuthManager:
    def __init__(self, db_path: str = None):
        """
        Initialize SQLite authentication manager.

        Args:
            db_path: Path to SQLite database file (if None, uses default location)
        """
        if db_path is None:
            # Create database in the Backend directory
            backend_dir = Path(__file__).parent
            db_path = backend_dir / "traffic_auth.db"

        self.db_path = str(db_path)
        self._init_database()

        print(f"[INFO] SQLiteAuth initialized with database: {self.db_path}")

    def _get_connection(self) -> sqlite3.Connection:
        """Get a connection to the SQLite database."""
        conn = sqlite3.connect(self.db_path)
        # Enable foreign key constraints
        conn.execute("PRAGMA foreign_keys = ON")
        # Return rows as dictionaries
        conn.row_factory = sqlite3.Row
        return conn

    def _init_database(self):
        """Initialize database tables if they don't exist."""
        with self._get_connection() as conn:
            # Users table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    role TEXT DEFAULT 'operator',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    is_active BOOLEAN DEFAULT 1
                )
            """)

            # Sessions table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    token TEXT UNIQUE NOT NULL,
                    user_id INTEGER NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP NOT NULL,
                    is_active BOOLEAN DEFAULT 1,
                    FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
                )
            """)

            # Create indexes for better performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_users_username ON users(username)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_sessions_token ON sessions(token)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_sessions_expires ON sessions(expires_at)")

            conn.commit()

    def hash_password(self, password: str) -> str:
        """Hash a password using bcrypt."""
        salt = bcrypt.gensalt()
        hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
        return hashed.decode('utf-8')

    def verify_password(self, password: str, hashed: str) -> bool:
        """Verify a password against its hash."""
        try:
            return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))
        except Exception:
            return False

    def generate_session_token(self) -> str:
        """Generate a secure random session token."""
        return secrets.token_urlsafe(32)

    def create_user(self, username: str, password: str, role: str = "operator") -> bool:
        """
        Create a new user.

        Args:
            username: Unique username
            password: Plain text password
            role: User role (operator, supervisor, admin, etc.)

        Returns:
            True if user created successfully, False if username already exists
        """
        try:
            hashed_password = self.hash_password(password)
            with self._get_connection() as conn:
                conn.execute(
                    "INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",
                    (username, hashed_password, role)
                )
                conn.commit()
            return True
        except sqlite3.IntegrityError:
            # Username already exists
            return False
        except Exception as e:
            print(f"[ERROR] Error creating user: {e}")
            return False

    def validate_user(self, username: str, password: str) -> Optional[Dict[str, Any]]:
        """
        Validate user credentials.

        Args:
            username: Username to validate
            password: Plain text password to validate

        Returns:
            User document if credentials are valid, None otherwise
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.execute(
                    "SELECT id, username, password_hash, role, is_active FROM users WHERE username = ? AND is_active = 1",
                    (username,)
                )
                row = cursor.fetchone()

                if row and self.verify_password(password, row['password_hash']):
                    return {
                        'id': row['id'],
                        'username': row['username'],
                        'role': row['role'],
                        'is_active': bool(row['is_active'])
                    }
                return None
        except Exception as e:
            print(f"[ERROR] Error validating user: {e}")
            return None

    def create_session(self, user_id: int) -> Tuple[str, datetime]:
        """
        Create a new session for a user.

        Args:
            user_id: User ID from the users table

        Returns:
            Tuple of (session_token, expires_at)
        """
        token = self.generate_session_token()
        expires_at = datetime.utcnow() + timedelta(hours=4)  # 4 hour sessions

        with self._get_connection() as conn:
            conn.execute(
                "INSERT INTO sessions (token, user_id, expires_at) VALUES (?, ?, ?)",
                (token, user_id, expires_at)
            )
            conn.commit()

        return token, expires_at

    def validate_session(self, token: str) -> Optional[Dict[str, Any]]:
        """
        Validate a session token and return associated user.

        Args:
            token: Session token to validate

        Returns:
            User document if session is valid, None otherwise
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.execute("""
                    SELECT u.id, u.username, u.role, u.is_active, s.expires_at
                    FROM users u
                    JOIN sessions s ON u.id = s.user_id
                    WHERE s.token = ? AND s.is_active = 1 AND u.is_active = 1
                """, (token,))
                row = cursor.fetchone()

                if row:
                    # Check if session has expired
                    expires_at = datetime.fromisoformat(row['expires_at']) if isinstance(row['expires_at'], str) else row['expires_at']
                    if expires_at > datetime.utcnow():
                        return {
                            'id': row['id'],
                            'username': row['username'],
                            'role': row['role'],
                            'is_active': bool(row['is_active'])
                        }
                return None
        except Exception as e:
            print(f"[ERROR] Error validating session: {e}")
            return None

    def invalidate_session(self, token: str) -> bool:
        """
        Invalidate a session (logout).

        Args:
            token: Session token to invalidate

        Returns:
            True if session was found and invalidated, False otherwise
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.execute(
                    "UPDATE sessions SET is_active = 0 WHERE token = ? AND is_active = 1",
                    (token,)
                )
                conn.commit()
                return cursor.rowcount > 0
        except Exception as e:
            print(f"[ERROR] Error invalidating session: {e}")
            return False

    def get_user_by_username(self, username: str) -> Optional[Dict[str, Any]]:
        """
        Get user by username.

        Args:
            username: Username to search for

        Returns:
            User document if found, None otherwise
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.execute(
                    "SELECT id, username, role, is_active FROM users WHERE username = ? AND is_active = 1",
                    (username,)
                )
                row = cursor.fetchone()

                if row:
                    return {
                        'id': row['id'],
                        'username': row['username'],
                        'role': row['role'],
                        'is_active': bool(row['is_active'])
                    }
                return None
        except Exception as e:
            print(f"[ERROR] Error getting user by username: {e}")
            return None

    def cleanup_expired_sessions(self):
        """Remove expired sessions from the database."""
        try:
            with self._get_connection() as conn:
                conn.execute(
                    "UPDATE sessions SET is_active = 0 WHERE expires_at < ? AND is_active = 1",
                    (datetime.utcnow(),)
                )
                conn.commit()
        except Exception as e:
            print(f"[ERROR] Error cleaning up expired sessions: {e}")

    def close(self):
        """Close database connections (SQLite handles this automatically, but good practice)."""
        pass  # SQLite connections are closed when context managers exit


# Global instance for easy access
_auth_manager = None

def get_auth_manager() -> SQLiteAuthManager:
    """Get or create the global SQLiteAuthManager instance."""
    global _auth_manager
    if _auth_manager is None:
        _auth_manager = SQLiteAuthManager()
    return _auth_manager

def init_default_users():
    """Initialize default users if none exist."""
    auth_manager = get_auth_manager()

    # Check if any users exist
    existing_user = auth_manager.get_user_by_username("admin")  # Just check one
    if not existing_user:
        # Create default admin user
        default_username = os.environ.get("DEFAULT_ADMIN_USERNAME", "admin")
        default_password = os.environ.get("DEFAULT_ADMIN_PASSWORD", "SecureAdminPass123!")

        if auth_manager.create_user(default_username, default_password, "admin"):
            print(f"[INFO] Created default admin user: {default_username}")
            print(f"[WARN] Please change this password after first login!")
        else:
            print(f"[WARN] Failed to create default admin user (may already exist)")


# Initialize default users on import (useful for development)
# Comment this out if you prefer to initialize only on app startup
# init_default_users()