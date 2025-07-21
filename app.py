from flask import Flask, render_template, request, jsonify, send_file
import json
import re
from datetime import datetime
from typing import Dict, List, Optional, Any
from groq import Groq
import logging
from fuzzywuzzy import fuzz
from collections import defaultdict
import os
import io

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

class SearchBot:
    def __init__(self, groq_api_key: str, json_file_path: str):
        """
        Initialize the search bot
        
        Args:
            groq_api_key: Groq API key
            json_file_path: Path to JSON file with indexed data
        """
        self.groq_client = Groq(api_key=groq_api_key)
        self.json_file_path = json_file_path
        self.indexed_data = []
        self.load_json_data()
    
    def load_json_data(self):
        """Load data from JSON file"""
        try:
            with open(self.json_file_path, 'r', encoding='utf-8') as f:
                self.indexed_data = json.load(f)
            logger.info(f"Loaded {len(self.indexed_data)} items from JSON file")
        except Exception as e:
            logger.error(f"Error loading JSON file: {str(e)}")
            self.indexed_data = []
    
    def search_by_keywords(self, query: str, threshold: int = 70) -> List[Dict]:
        """Search data using keywords"""
        results = []
        query_lower = query.lower()
        
        for item in self.indexed_data:
            score = 0
            matches = []
            
            # Search in keywords
            for keyword in item.get('keywords', []):
                similarity = fuzz.ratio(query_lower, keyword.lower())
                if similarity > threshold:
                    score += similarity * 2
                    matches.append(f"keyword: {keyword}")
            
            # Search in description
            description = item.get('description', '')
            if description:
                similarity = fuzz.partial_ratio(query_lower, description.lower())
                if similarity > threshold:
                    score += similarity
                    matches.append(f"description: {description[:100]}...")
            
            # Search in title
            title = item.get('title', '')
            if title:
                similarity = fuzz.partial_ratio(query_lower, title.lower())
                if similarity > threshold:
                    score += similarity * 1.5
                    matches.append(f"title: {title}")
            
            # Search in main_topics
            for topic in item.get('main_topics', []):
                similarity = fuzz.ratio(query_lower, topic.lower())
                if similarity > threshold:
                    score += similarity * 1.2
                    matches.append(f"topic: {topic}")
            
            if score > 0:
                item_copy = item.copy()
                item_copy['search_score'] = score
                item_copy['matches'] = matches
                results.append(item_copy)
        
        # Sort by score
        results.sort(key=lambda x: x['search_score'], reverse=True)
        return results
    
    def get_statistics_summary(self) -> Dict:
        """Generate statistical summary of data"""
        if not self.indexed_data:
            return {}
        
        stats = {
            'total_pages': len(self.indexed_data),
            'content_types': defaultdict(int),
            'languages': defaultdict(int),
            'top_keywords': defaultdict(int),
            'top_topics': defaultdict(int)
        }
        
        for item in self.indexed_data:
            stats['content_types'][item.get('content_type', 'unknown')] += 1
            stats['languages'][item.get('language', 'unknown')] += 1
            
            for keyword in item.get('keywords', []):
                stats['top_keywords'][keyword.lower()] += 1
            
            for topic in item.get('main_topics', []):
                stats['top_topics'][topic.lower()] += 1
        
        # Convert to normal dicts and get top 10
        stats['content_types'] = dict(stats['content_types'])
        stats['languages'] = dict(stats['languages'])
        stats['top_keywords'] = dict(sorted(stats['top_keywords'].items(), key=lambda x: x[1], reverse=True)[:10])
        stats['top_topics'] = dict(sorted(stats['top_topics'].items(), key=lambda x: x[1], reverse=True)[:10])
        
        return stats
    
    def analyze_query_with_ai(self, user_query: str, search_results: List[Dict]) -> str:
        """Analyze user query and results with AI"""
        try:
            # Prepare data for AI
            results_summary = []
            for i, result in enumerate(search_results[:10]):
                summary = {
                    'url': result['url'],
                    'title': result['title'],
                    'description': result['description'],
                    'keywords': result['keywords'],
                    'content_type': result['content_type'],
                    'main_topics': result['main_topics'],
                    'relevance_score': result.get('relevance_score', 0),
                    'matches': result.get('matches', [])
                }
                results_summary.append(summary)
            
            # General statistics
            stats = self.get_statistics_summary()
            
            # AI prompt
            prompt = f"""
You are an expert assistant helping users find information in an indexed web pages database.

USER QUESTION: {user_query}

SEARCH RESULTS ({len(search_results)} total results):
{json.dumps(results_summary, indent=2, ensure_ascii=False)}

GENERAL DATABASE STATISTICS:
{json.dumps(stats, indent=2, ensure_ascii=False)}

INSTRUCTIONS:
1. Analyze the user's question and search results
2. Provide a detailed and structured response that includes:
   - A direct answer to the question
   - The best relevant results with explanations
   - Specific links the user should visit
   - Suggestions to refine the search if needed
   - Concrete actions the user can take

3. Structure the response professionally and helpfully
4. If there aren't enough results, suggest alternatives or search modifications
5. Always include the most relevant URLs in the response

RESPONSE FORMAT:
- Use clear headings
- List the most relevant results
- Provide concrete actions
- Write in Italian
"""

            # AI call
            response = self.groq_client.chat.completions.create(
                model="deepseek-r1-distill-llama-70b",
                messages=[
                    {"role": "system", "content": "You are an expert assistant for web information search. Always provide detailed, structured and helpful responses."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=2000
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Error in AI analysis: {str(e)}")
            return f"Error analyzing question: {str(e)}"
    
    def search_and_answer(self, user_query: str, max_results: int = 20) -> Dict:
        """Search for information and generate complete response"""
        logger.info(f"Processing query: {user_query}")
        
        # Search in data
        search_results = self.search_by_keywords(user_query)[:max_results]
        
        # Generate AI response
        ai_response = self.analyze_query_with_ai(user_query, search_results)
        
        return {
            'query': user_query,
            'results_count': len(search_results),
            'search_results': search_results,
            'ai_response': ai_response,
            'timestamp': datetime.now().isoformat()
        }

# Configuration
GROQ_API_KEY = "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"  # Replace with your API key
JSON_FILE = "indexed_content.json"  # JSON file generated by previous bot

# Initialize search bot
search_bot = SearchBot(groq_api_key=GROQ_API_KEY, json_file_path=JSON_FILE)

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    """Search endpoint"""
    try:
        data = request.get_json()
        query = data.get('query', '').strip()
        
        if not query:
            return jsonify({'error': 'No query provided'}), 400
        
        # Perform search
        result = search_bot.search_and_answer(query)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/stats')
def get_stats():
    """Get database statistics"""
    try:
        stats = search_bot.get_statistics_summary()
        return jsonify(stats)
    except Exception as e:
        logger.error(f"Stats error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/download/<format>')
def download_results(format):
    """Download search results"""
    try:
        # This would be called with the last search results
        # For now, return a simple example
        content = "Search results would be exported here"
        
        if format == 'txt':
            output = io.StringIO()
            output.write(content)
            output.seek(0)
            
            return send_file(
                io.BytesIO(output.getvalue().encode()),
                mimetype='text/plain',
                as_attachment=True,
                download_name='search_results.txt'
            )
        
        return jsonify({'error': 'Unsupported format'}), 400
        
    except Exception as e:
        logger.error(f"Download error: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    if not os.path.exists('templates'):
        os.makedirs('templates')
    
    app.run(debug=True, host='0.0.0.0', port=8080)
