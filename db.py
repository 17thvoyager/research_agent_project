import sqlite3
import hashlib
import os
import json

DB_FILE = "app_database_v2.db"

def init_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    # Users Table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash BLOB NOT NULL,
            salt BLOB NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Chat History Table — stores every Q&A per user
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS chats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            citations TEXT,
            research_gaps TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (username) REFERENCES users (username)
        )
    ''')

    # User Documents Table — tracks which user owns which file
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            filename TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(username, filename),
            FOREIGN KEY (username) REFERENCES users (username)
        )
    ''')
    
    conn.commit()
    conn.close()

def _hash_password(password: str, salt: bytes = None):
    if salt is None:
        salt = os.urandom(32)
    key = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, 100000)
    return key, salt

def register_user(username: str, password: str):
    username = username.lower().strip()
    key, salt = _hash_password(password)
    
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("INSERT INTO users (username, password_hash, salt) VALUES (?, ?, ?)", (username, key, salt))
        conn.commit()
        return True, "User registered successfully."
    except sqlite3.IntegrityError:
        return False, "Username already exists."
    finally:
        conn.close()

def verify_user(username: str, password: str):
    username = username.lower().strip()
    
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT password_hash, salt FROM users WHERE username = ?", (username,))
    row = cursor.fetchone()
    conn.close()
    
    if not row:
        return False
        
    stored_hash, salt = row
    key, _ = _hash_password(password, salt)
    
    return key == stored_hash

def save_chat_message(username: str, role: str, content: str, citations: list = None, research_gaps: list = None):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO chats (username, role, content, citations, research_gaps) VALUES (?, ?, ?, ?, ?)",
        (
            username.lower().strip(),
            role,
            content,
            json.dumps(citations) if citations else None,
            json.dumps(research_gaps) if research_gaps else None,
        )
    )
    conn.commit()
    conn.close()

def get_chat_history(username: str):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT role, content, citations, research_gaps, created_at FROM chats WHERE username = ? ORDER BY id ASC",
        (username.lower().strip(),)
    )
    rows = cursor.fetchall()
    conn.close()
    
    history = []
    for row in rows:
        history.append({
            "role":          row[0],
            "content":       row[1],
            "citations":     json.loads(row[2]) if row[2] else [],
            "research_gaps": json.loads(row[3]) if row[3] else [],
        })
    return history

def register_document(username: str, filename: str):
    """Link a filename to a username (idempotent)."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT OR IGNORE INTO user_documents (username, filename) VALUES (?, ?)",
        (username.lower().strip(), filename)
    )
    conn.commit()
    conn.close()

def get_user_documents(username: str):
    """Return list of filenames that belong to this user."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT filename FROM user_documents WHERE username = ? ORDER BY created_at ASC",
        (username.lower().strip(),)
    )
    rows = cursor.fetchall()
    conn.close()
    return [row[0] for row in rows]

def remove_user_document(username: str, filename: str):
    """Unlink a file from a user."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute(
        "DELETE FROM user_documents WHERE username = ? AND filename = ?",
        (username.lower().strip(), filename)
    )
    conn.commit()
    conn.close()

init_db()
