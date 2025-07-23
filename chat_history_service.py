"""
Chat history service for CodeEx
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

@dataclass
class Message:
    role: str
    content: str
    timestamp: datetime
    encrypted: bool = False

@dataclass
class StoragePolicy:
    max_sessions_per_user: int = 50
    max_messages_per_session: int = 1000
    retention_days: int = 30
    auto_archive_days: int = 7
    encryption_enabled: bool = True
    compress_old_messages: bool = True

class ChatHistoryService:
    def __init__(self):
        self.storage_policy = StoragePolicy()
        self.messages = {}  # Simple in-memory storage
    
    def save_message(self, session_id: str, role: str, content: str, user_id: int, encrypt: bool = True) -> None:
        """Save a message to the chat history"""
        if session_id not in self.messages:
            self.messages[session_id] = []
        
        message = Message(
            role=role,
            content=content,
            timestamp=datetime.now(),
            encrypted=encrypt
        )
        
        self.messages[session_id].append(message)
    
    def get_messages(self, session_id: str, user_id: int, decrypt: bool = True) -> List[Message]:
        """Get messages for a session"""
        return self.messages.get(session_id, [])
    
    def get_conversation_context(self, user_id: int, session_id: str, context_window: int = 10) -> Dict[str, Any]:
        """Get conversation context for a session"""
        messages = self.get_messages(session_id, user_id)
        recent_messages = [msg.content for msg in messages[-context_window:] if msg.role == 'user']
        
        return {
            'recent_messages': recent_messages,
            'context_memories': [],
            'session_id': session_id
        }
    
    def get_storage_analytics(self, user_id: int) -> Dict[str, Any]:
        """Get storage analytics for a user"""
        return {
            'total_sessions': len(self.messages),
            'total_messages': sum(len(msgs) for msgs in self.messages.values()),
            'encrypted_messages': sum(1 for msgs in self.messages.values() for msg in msgs if msg.encrypted),
            'oldest_message': datetime.now().isoformat() if not self.messages else min(
                (msg.timestamp for msgs in self.messages.values() for msg in msgs), 
                default=datetime.now()
            ).isoformat()
        }
    
    def cleanup_old_data(self, user_id: int) -> int:
        """Clean up old chat data"""
        return 0  # No cleanup in this simple implementation
    
    def update_storage_policy(self, **kwargs) -> None:
        """Update storage policy"""
        for key, value in kwargs.items():
            if hasattr(self.storage_policy, key):
                setattr(self.storage_policy, key, value)
    
    def export_user_data(self, user_id: int, include_encrypted: bool = False) -> Dict[str, Any]:
        """Export user data"""
        export_data = {
            'user_id': user_id,
            'export_date': datetime.now().isoformat(),
            'sessions': []
        }
        
        for session_id, messages in self.messages.items():
            session_data = {
                'session_id': session_id,
                'messages': [
                    {
                        'role': msg.role,
                        'content': msg.content,
                        'timestamp': msg.timestamp.isoformat(),
                        'encrypted': msg.encrypted
                    }
                    for msg in messages if include_encrypted or not msg.encrypted
                ]
            }
            export_data['sessions'].append(session_data)
        
        return export_data

# Global instance
chat_history_service = ChatHistoryService()