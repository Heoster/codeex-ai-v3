"""
📚 Wikipedia Knowledge Integration
Adds Wikipedia knowledge to the AI brain
"""

import requests
import json
import logging
from typing import Dict, List, Optional
from datetime import datetime
import re
import time

logger = logging.getLogger(__name__)


class WikipediaKnowledge:
    """Wikipedia knowledge integration for AI brain"""

    def __init__(self):
        self.base_url = "https://en.wikipedia.org/api/rest_v1"
        self.search_url = "https://en.wikipedia.org/w/api.php"
        self.knowledge_cache = {}
        self.search_cache = {}
        self.last_request_time = 0
        self.min_request_interval = 1.0  # Minimum 1 second between requests

    def search_wikipedia(self, query: str, limit: int = 5) -> List[Dict]:
        """Search Wikipedia for articles"""
        if query in self.search_cache:
            return self.search_cache[query]

        try:
            params = {
                'action': 'query',
                'format': 'json',
                'list': 'search',
                'srsearch': query,
                'srlimit': limit
            }

            response = requests.get(self.search_url, params=params, timeout=10)
            
            # Debug: Check response status and content
            if response.status_code != 200:
                logger.error(f"Wikipedia API returned status {response.status_code}")
                return []
            
            # Try to parse JSON with better error handling
            try:
                data = response.json()
            except json.JSONDecodeError as json_error:
                logger.error(f"Wikipedia JSON parse error: {json_error}")
                logger.error(f"Response content: {response.text[:200]}...")
                return []

            results = []
            if 'query' in data and 'search' in data['query']:
                for item in data['query']['search']:
                    results.append({
                        'title': item['title'],
                        'snippet': self.clean_html(item['snippet']),
                        'pageid': item['pageid'],
                        'size': item['size'],
                        'wordcount': item['wordcount'],
                        'timestamp': item['timestamp']
                    })

            self.search_cache[query] = results
            return results

        except Exception as e:
            logger.error(f"Wikipedia search error: {e}")
            return []

    def get_article_content(self, title: str) -> Optional[Dict]:
        """Get full article content from Wikipedia"""
        if title in self.knowledge_cache:
            return self.knowledge_cache[title]

        try:
            # Get page content
            url = f"{self.base_url}/page/summary/{title.replace(' ', '_')}"
            response = requests.get(url, timeout=10)

            if response.status_code == 200:
                data = response.json()

                article_data = {
                    'title': data.get('title', ''),
                    'extract': data.get('extract', ''),
                    'description': data.get('description', ''),
                    'url': data.get('content_urls', {}).get('desktop', {}).get('page', ''),
                    'thumbnail': data.get('thumbnail', {}).get('source', '') if data.get('thumbnail') else '',
                    'lang': data.get('lang', 'en'),
                    'timestamp': data.get('timestamp', ''),
                    'type': data.get('type', 'standard')
                }

                self.knowledge_cache[title] = article_data
                return article_data

        except Exception as e:
            logger.error(f"Error fetching Wikipedia article '{title}': {e}")

        return None

    def get_related_articles(self, title: str, limit: int = 5) -> List[Dict]:
        """Get related articles for a given title"""
        try:
            # Search for related content
            search_terms = title.split()[:3]  # Use first 3 words
            search_query = " ".join(search_terms)

            related = self.search_wikipedia(search_query, limit + 1)

            # Filter out the original article
            filtered = [
                article for article in related if article['title'].lower() != title.lower()]

            return filtered[:limit]

        except Exception as e:
            logger.error(f"Error getting related articles for '{title}': {e}")
            return []

    def clean_html(self, text: str) -> str:
        """Clean HTML tags from text"""
        if not text:
            return ""

        # Remove HTML tags
        clean_text = re.sub(r'<[^>]+>', '', text)

        # Clean up extra whitespace
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()

        return clean_text

    def extract_key_facts(self, content: str) -> List[str]:
        """Extract key facts from Wikipedia content"""
        if not content:
            return []

        facts = []

        # Split into sentences
        sentences = re.split(r'[.!?]+', content)

        # Filter for informative sentences
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 20 and len(sentence) < 200:
                # Look for factual patterns
                if any(keyword in sentence.lower() for keyword in [
                    'is a', 'was a', 'are', 'were', 'born', 'died', 'founded',
                    'created', 'developed', 'invented', 'discovered', 'located'
                ]):
                    facts.append(sentence)

        return facts[:5]  # Return top 5 facts

    def get_knowledge_for_query(self, query: str) -> Dict:
        """Get comprehensive knowledge for a query"""
        try:
            # Search for relevant articles
            search_results = self.search_wikipedia(query, 3)

            if not search_results:
                return {'found': False, 'message': 'No Wikipedia articles found'}

            # Get detailed content for the top result
            top_result = search_results[0]
            article_content = self.get_article_content(top_result['title'])

            if not article_content:
                return {'found': False, 'message': 'Could not retrieve article content'}

            # Get related articles
            related_articles = self.get_related_articles(
                top_result['title'], 3)

            # Extract key facts
            key_facts = self.extract_key_facts(article_content['extract'])

            return {
                'found': True,
                'main_article': article_content,
                'search_results': search_results,
                'related_articles': related_articles,
                'key_facts': key_facts,
                'query': query,
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Error getting knowledge for query '{query}': {e}")
            return {'found': False, 'error': str(e)}

    def format_knowledge_response(self, knowledge_data: Dict) -> str:
        """Format knowledge data into a readable response"""
        if not knowledge_data.get('found'):
            return "I couldn't find relevant information on Wikipedia for that topic."

        main_article = knowledge_data['main_article']
        key_facts = knowledge_data.get('key_facts', [])

        response = f"**{main_article['title']}**\n\n"

        if main_article.get('description'):
            response += f"{main_article['description']}\n\n"

        # Add extract (summary)
        if main_article.get('extract'):
            extract = main_article['extract'][:500]  # Limit length
            if len(main_article['extract']) > 500:
                extract += "..."
            response += f"{extract}\n\n"

        # Add key facts
        if key_facts:
            response += "**Key Facts:**\n"
            for i, fact in enumerate(key_facts[:3], 1):
                response += f"{i}. {fact}\n"
            response += "\n"

        # Add related topics
        related = knowledge_data.get('related_articles', [])
        if related:
            response += "**Related Topics:** "
            related_titles = [article['title'] for article in related[:3]]
            response += ", ".join(related_titles)
            response += "\n\n"

        # Add source
        if main_article.get('url'):
            response += f"*Source: [Wikipedia]({main_article['url']})*"

        return response


# Global instance
wikipedia_knowledge = WikipediaKnowledge()
