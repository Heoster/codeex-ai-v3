"""
🌐 Advanced Web Scraping Service for CodeEx AI
Real-time data extraction and knowledge enhancement
"""

import requests
from bs4 import BeautifulSoup
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import sqlite3
import threading
import time
import re
from urllib.parse import urljoin, urlparse
import hashlib
from dataclasses import dataclass, asdict
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

@dataclass
class ScrapingResult:
    """Structure for scraping results"""
    url: str
    title: str
    content: str
    metadata: Dict[str, Any]
    timestamp: datetime
    success: bool
    error: Optional[str] = None

@dataclass
class ScrapingSource:
    """Configuration for scraping sources"""
    name: str
    url: str
    selector: str
    update_frequency: int  # minutes
    enabled: bool = True
    last_scraped: Optional[datetime] = None

class WebScrapingService:
    """🕷️ Advanced Web Scraping Service with Real-time Updates"""
    
    def __init__(self, db_path="web_scraping.db"):
        self.db_path = db_path
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'CodeEx AI Web Scraper 1.0 (Educational Purpose)',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        })
        
        self.scraping_sources = self._load_default_sources()
        self.executor = ThreadPoolExecutor(max_workers=5)
        self.is_running = False
        self.scraping_thread = None
        
        self.init_database()
        
    def init_database(self):
        """Initialize SQLite database for scraping data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Scraping results table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS scraping_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                url TEXT NOT NULL,
                title TEXT,
                content TEXT,
                metadata TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                success BOOLEAN,
                error TEXT,
                content_hash TEXT
            )
        ''')
        
        # Scraping sources table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS scraping_sources (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                url TEXT NOT NULL,
                selector TEXT,
                update_frequency INTEGER DEFAULT 60,
                enabled BOOLEAN DEFAULT 1,
                last_scraped TIMESTAMP,
                success_rate REAL DEFAULT 1.0
            )
        ''')
        
        # Scraping logs table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS scraping_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_name TEXT,
                action TEXT,
                status TEXT,
                message TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
        
    def _load_default_sources(self) -> List[ScrapingSource]:
        """Load default scraping sources"""
        return [
            ScrapingSource(
                name="Wikipedia Featured Article",
                url="https://en.wikipedia.org/wiki/Main_Page",
                selector="#mp-tfa",
                update_frequency=360  # 6 hours
            ),
            ScrapingSource(
                name="Python News",
                url="https://www.python.org/jobs/",
                selector=".list-recent-jobs",
                update_frequency=120  # 2 hours
            ),
            ScrapingSource(
                name="Tech News",
                url="https://news.ycombinator.com/",
                selector=".storylink",
                update_frequency=60  # 1 hour
            ),
            ScrapingSource(
                name="GitHub Trending",
                url="https://github.com/trending",
                selector=".Box-row",
                update_frequency=180  # 3 hours
            ),
            ScrapingSource(
                name="Stack Overflow Questions",
                url="https://stackoverflow.com/questions",
                selector=".question-summary",
                update_frequency=30  # 30 minutes
            )
        ]
    
    def scrape_url(self, url: str, selector: str = None, timeout: int = 10) -> ScrapingResult:
        """🔍 Scrape a single URL with advanced parsing"""
        try:
            # Validate URL
            parsed_url = urlparse(url)
            if not parsed_url.scheme or not parsed_url.netloc:
                raise ValueError(f"Invalid URL: {url}")
            
            # Make request with timeout
            response = self.session.get(url, timeout=timeout)
            response.raise_for_status()
            
            # Parse HTML
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract title
            title_tag = soup.find('title')
            title = title_tag.get_text().strip() if title_tag else "No Title"
            
            # Extract content based on selector or use smart extraction
            if selector:
                content_elements = soup.select(selector)
                content = ' '.join([elem.get_text().strip() for elem in content_elements])
            else:
                content = self._smart_content_extraction(soup)
            
            # Extract metadata
            metadata = self._extract_metadata(soup, response)
            
            # Create content hash for deduplication
            content_hash = hashlib.md5(content.encode()).hexdigest()
            
            result = ScrapingResult(
                url=url,
                title=title,
                content=content[:5000],  # Limit content length
                metadata=metadata,
                timestamp=datetime.now(),
                success=True
            )
            
            # Store result in database
            self._store_result(result, content_hash)
            
            return result
            
        except requests.RequestException as e:
            logger.error(f"Request error scraping {url}: {e}")
            return ScrapingResult(
                url=url,
                title="",
                content="",
                metadata={},
                timestamp=datetime.now(),
                success=False,
                error=str(e)
            )
        except Exception as e:
            logger.error(f"Error scraping {url}: {e}")
            return ScrapingResult(
                url=url,
                title="",
                content="",
                metadata={},
                timestamp=datetime.now(),
                success=False,
                error=str(e)
            )
    
    def _smart_content_extraction(self, soup: BeautifulSoup) -> str:
        """🧠 Smart content extraction using heuristics"""
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "header", "footer", "aside"]):
            script.decompose()
        
        # Try to find main content areas
        content_selectors = [
            'main', 'article', '.content', '#content', '.post', '.entry',
            '.article-body', '.story-body', '.post-content'
        ]
        
        for selector in content_selectors:
            content_elem = soup.select_one(selector)
            if content_elem:
                return content_elem.get_text().strip()
        
        # Fallback to body content
        body = soup.find('body')
        if body:
            return body.get_text().strip()
        
        return soup.get_text().strip()
    
    def _extract_metadata(self, soup: BeautifulSoup, response) -> Dict[str, Any]:
        """📊 Extract metadata from HTML"""
        metadata = {
            'status_code': response.status_code,
            'content_type': response.headers.get('content-type', ''),
            'content_length': len(response.content),
            'encoding': response.encoding
        }
        
        # Extract meta tags
        meta_tags = soup.find_all('meta')
        for tag in meta_tags:
            name = tag.get('name') or tag.get('property')
            content = tag.get('content')
            if name and content:
                metadata[f'meta_{name}'] = content
        
        # Extract links
        links = soup.find_all('a', href=True)
        metadata['link_count'] = len(links)
        
        # Extract images
        images = soup.find_all('img', src=True)
        metadata['image_count'] = len(images)
        
        return metadata
    
    def _store_result(self, result: ScrapingResult, content_hash: str):
        """💾 Store scraping result in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Check if content already exists (deduplication)
        cursor.execute('SELECT id FROM scraping_results WHERE content_hash = ?', (content_hash,))
        if cursor.fetchone():
            conn.close()
            return  # Skip duplicate content
        
        cursor.execute('''
            INSERT INTO scraping_results 
            (url, title, content, metadata, timestamp, success, error, content_hash)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            result.url,
            result.title,
            result.content,
            json.dumps(result.metadata),
            result.timestamp,
            result.success,
            result.error,
            content_hash
        ))
        
        conn.commit()
        conn.close()
    
    def start_auto_scraping(self):
        """🚀 Start automatic scraping in background"""
        if self.is_running:
            return
        
        self.is_running = True
        self.scraping_thread = threading.Thread(target=self._auto_scraping_loop, daemon=True)
        self.scraping_thread.start()
        logger.info("Auto-scraping started")
    
    def stop_auto_scraping(self):
        """⏹️ Stop automatic scraping"""
        self.is_running = False
        if self.scraping_thread:
            self.scraping_thread.join(timeout=5)
        logger.info("Auto-scraping stopped")
    
    def _auto_scraping_loop(self):
        """🔄 Main auto-scraping loop"""
        while self.is_running:
            try:
                for source in self.scraping_sources:
                    if not source.enabled:
                        continue
                    
                    # Check if it's time to scrape this source
                    if self._should_scrape_source(source):
                        self._log_scraping_action(source.name, "scraping", "started", f"Scraping {source.url}")
                        
                        result = self.scrape_url(source.url, source.selector)
                        source.last_scraped = datetime.now()
                        
                        if result.success:
                            self._log_scraping_action(source.name, "scraping", "success", f"Successfully scraped {source.url}")
                        else:
                            self._log_scraping_action(source.name, "scraping", "error", f"Failed to scrape {source.url}: {result.error}")
                
                # Sleep for 1 minute before next check
                time.sleep(60)
                
            except Exception as e:
                logger.error(f"Error in auto-scraping loop: {e}")
                time.sleep(60)
    
    def _should_scrape_source(self, source: ScrapingSource) -> bool:
        """⏰ Check if source should be scraped based on frequency"""
        if not source.last_scraped:
            return True
        
        time_since_last = datetime.now() - source.last_scraped
        return time_since_last.total_seconds() >= (source.update_frequency * 60)
    
    def _log_scraping_action(self, source_name: str, action: str, status: str, message: str):
        """📝 Log scraping actions"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO scraping_logs (source_name, action, status, message)
            VALUES (?, ?, ?, ?)
        ''', (source_name, action, status, message))
        
        conn.commit()
        conn.close()
    
    def get_recent_results(self, limit: int = 50) -> List[Dict[str, Any]]:
        """📊 Get recent scraping results"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT url, title, content, metadata, timestamp, success, error
            FROM scraping_results
            ORDER BY timestamp DESC
            LIMIT ?
        ''', (limit,))
        
        results = []
        for row in cursor.fetchall():
            results.append({
                'url': row[0],
                'title': row[1],
                'content': row[2][:200] + '...' if len(row[2]) > 200 else row[2],
                'metadata': json.loads(row[3]) if row[3] else {},
                'timestamp': row[4],
                'success': row[5],
                'error': row[6]
            })
        
        conn.close()
        return results
    
    def get_scraping_stats(self) -> Dict[str, Any]:
        """📈 Get scraping statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Total results
        cursor.execute('SELECT COUNT(*) FROM scraping_results')
        total_results = cursor.fetchone()[0]
        
        # Success rate
        cursor.execute('SELECT COUNT(*) FROM scraping_results WHERE success = 1')
        successful_results = cursor.fetchone()[0]
        success_rate = (successful_results / total_results * 100) if total_results > 0 else 0
        
        # Recent activity (last 24 hours)
        cursor.execute('''
            SELECT COUNT(*) FROM scraping_results 
            WHERE timestamp > datetime('now', '-24 hours')
        ''')
        recent_activity = cursor.fetchone()[0]
        
        # Sources monitored
        sources_monitored = len([s for s in self.scraping_sources if s.enabled])
        
        conn.close()
        
        return {
            'total_results': total_results,
            'success_rate': round(success_rate, 1),
            'recent_activity': recent_activity,
            'sources_monitored': sources_monitored,
            'auto_scraping_active': self.is_running
        }
    
    def search_scraped_content(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """🔍 Search through scraped content"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT url, title, content, timestamp
            FROM scraping_results
            WHERE (title LIKE ? OR content LIKE ?) AND success = 1
            ORDER BY timestamp DESC
            LIMIT ?
        ''', (f'%{query}%', f'%{query}%', limit))
        
        results = []
        for row in cursor.fetchall():
            results.append({
                'url': row[0],
                'title': row[1],
                'content': row[2][:300] + '...' if len(row[2]) > 300 else row[2],
                'timestamp': row[3]
            })
        
        conn.close()
        return results
    
    def test_scraping(self, url: str = "https://httpbin.org/html") -> Dict[str, Any]:
        """🧪 Test scraping functionality"""
        try:
            result = self.scrape_url(url)
            
            return {
                'success': result.success,
                'url': result.url,
                'title': result.title,
                'content_length': len(result.content),
                'metadata': result.metadata,
                'error': result.error,
                'timestamp': result.timestamp.isoformat()
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

# Global scraping service instance
scraping_service = WebScrapingService()

def get_scraping_service() -> WebScrapingService:
    """Get the global scraping service instance"""
    return scraping_service