import csv
import json
import requests
from bs4 import BeautifulSoup
import time
from groq import Groq
import re
from urllib.parse import urljoin, urlparse
import logging
from typing import Dict, List, Optional
import os
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AIWebsiteIndexer:
    def __init__(self, groq_api_key: str):
        """Initialize the AI Website Indexer with Groq API key."""
        self.groq_client = Groq(api_key=groq_api_key)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
    def is_valid_url(self, url: str) -> bool:
        """Check if URL is valid and not a system/protection page."""
        skip_patterns = [
            '/cdn-cgi/',
            '/email-protection',
            'javascript:',
            'mailto:',
            'tel:',
            '#',
            'void(0)'
        ]
        
        for pattern in skip_patterns:
            if pattern in url.lower():
                return False
        
        return url.startswith(('http://', 'https://'))
    
    def extract_webpage_content(self, url: str) -> Dict[str, str]:
        """Extract clean content from a webpage."""
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove unwanted elements
            for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside', '.navigation', '.nav', '.menu', '.sidebar', '.breadcrumb']):
                element.decompose()
            
            # Extract title
            title = soup.find('title')
            title_text = title.get_text().strip() if title else ""
            
            # Extract meta description
            meta_desc = soup.find('meta', attrs={'name': 'description'})
            description = meta_desc.get('content', '').strip() if meta_desc else ""
            
            # Extract main content
            main_content = ""
            content_selectors = ['main', 'article', '[role="main"]', '.content', '#content', '.main-content']
            
            for selector in content_selectors:
                content_element = soup.select_one(selector)
                if content_element:
                    main_content = content_element.get_text(separator=' ', strip=True)
                    break
            
            # If no main content found, use body
            if not main_content:
                body = soup.find('body')
                if body:
                    main_content = body.get_text(separator=' ', strip=True)
            
            # Clean content
            main_content = re.sub(r'\s+', ' ', main_content).strip()
            
            # Truncate to avoid token limits
            main_content = main_content[:4000]
            
            # Extract some links
            links = []
            for link in soup.find_all('a', href=True):
                href = link.get('href')
                if href:
                    absolute_url = urljoin(url, href)
                    link_text = link.get_text().strip()
                    if link_text:
                        links.append({"url": absolute_url, "text": link_text})
            
            return {
                "title": title_text,
                "description": description,
                "main_content": main_content,
                "links": links[:20]
            }
            
        except Exception as e:
            logger.error(f"Error extracting content from {url}: {str(e)}")
            return {
                "title": "",
                "description": "",
                "main_content": "",
                "links": []
            }
    
    def analyze_with_groq(self, url: str, content: Dict[str, str]) -> Dict:
        """Analyze webpage content using Groq AI."""
        try:
            # Check if we have enough content
            if len(content['main_content']) < 50:
                logger.warning(f"Not enough content to analyze for {url}")
                return self._create_fallback_result(url, content, "Insufficient content")
            
            # Prepare prompt
            prompt = f"""Analyze this webpage content and return a JSON response:

URL: {url}
Title: {content['title']}
Description: {content['description']}
Content: {content['main_content']}

Return JSON with this exact structure:
{{
    "url": "{url}",
    "title": "clean title",
    "description": "2-3 sentence description of the page",
    "keywords": ["keyword1", "keyword2", "keyword3", "keyword4", "keyword5"],
    "valuable_content": ["valuable aspect 1", "valuable aspect 2", "valuable aspect 3"],
    "content_type": "academic/news/blog/product/service/documentation/other",
    "main_topics": ["topic1", "topic2", "topic3"],
    "target_audience": "who would find this useful"
}}

Return only valid JSON, no additional text."""
            
            response = self.groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": "You are a web content analyzer. Return only valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=800
            )
            
            # Get response text
            response_text = response.choices[0].message.content.strip()
            
            # Clean response
            if response_text.startswith('```json'):
                response_text = response_text[7:-3]
            elif response_text.startswith('```'):
                response_text = response_text[3:-3]
            
            # Parse JSON
            result = json.loads(response_text)
            
            # Add metadata
            result["indexed_at"] = datetime.now().isoformat()
            result["content_length"] = len(content['main_content'])
            result["has_meta_description"] = bool(content['description'])
            result["internal_links_count"] = len(content['links'])
            
            logger.info(f"Successfully analyzed: {result.get('title', 'No title')}")
            return result
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error for {url}: {str(e)}")
            return self._create_fallback_result(url, content, "JSON parsing failed")
        except Exception as e:
            logger.error(f"Error analyzing {url}: {str(e)}")
            return self._create_fallback_result(url, content, f"Analysis error: {str(e)}")
    
    def _create_fallback_result(self, url: str, content: Dict[str, str], error_msg: str) -> Dict:
        """Create fallback result when AI analysis fails."""
        title = content.get('title', 'No title found')
        description = content.get('description', 'No description available')
        
        # Extract basic keywords
        keywords = []
        text = f"{title} {description}".lower()
        common_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        words = re.findall(r'\b\w+\b', text)
        keywords = [word for word in words if len(word) > 3 and word not in common_words][:5]
        
        return {
            "url": url,
            "title": title,
            "description": description,
            "keywords": keywords,
            "valuable_content": [f"Basic extraction - {error_msg}"],
            "content_type": "unknown",
            "main_topics": keywords[:3],
            "target_audience": "unknown",
            "indexed_at": datetime.now().isoformat(),
            "content_length": len(content['main_content']),
            "has_meta_description": bool(content['description']),
            "internal_links_count": len(content['links']),
            "error": error_msg
        }
    
    def load_urls_from_csv(self, csv_file: str) -> List[str]:
        """Load URLs from CSV file."""
        urls = []
        try:
            with open(csv_file, 'r', encoding='utf-8') as file:
                reader = csv.reader(file)
                for row in reader:
                    if row and row[0].strip():
                        url = row[0].strip()
                        if self.is_valid_url(url):
                            urls.append(url)
                        else:
                            logger.info(f"Skipping invalid URL: {url}")
                            
        except FileNotFoundError:
            logger.error(f"CSV file {csv_file} not found")
        except Exception as e:
            logger.error(f"Error reading CSV file: {str(e)}")
            
        return urls
    
    def process_urls(self, urls: List[str], output_file: str, delay: float = 1.0) -> List[Dict]:
        """Process URLs and save results."""
        results = []
        
        for i, url in enumerate(urls, 1):
            logger.info(f"Processing {i}/{len(urls)}: {url}")
            
            try:
                # Extract content
                content = self.extract_webpage_content(url)
                
                if content['main_content']:
                    # Analyze with AI
                    analysis = self.analyze_with_groq(url, content)
                    results.append(analysis)
                else:
                    logger.warning(f"No content extracted from {url}")
                    
            except Exception as e:
                logger.error(f"Error processing {url}: {str(e)}")
                
            # Rate limiting
            time.sleep(delay)
        
        # Save results
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            logger.info(f"Results saved to {output_file}")
            
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")
        
        return results
    
    def run_indexer(self, csv_file: str = "unimi_links.csv", output_file: str = "indexed_websites.json", max_urls: int = 100):
        """Main method to run the indexer."""
        logger.info("Starting AI Website Indexer...")
        
        # Load URLs
        urls = self.load_urls_from_csv(csv_file)
        
        if not urls:
            logger.error("No valid URLs found in CSV file")
            return
        
        # Limit URLs for testing
        if len(urls) > max_urls:
            logger.info(f"Limiting to first {max_urls} URLs (found {len(urls)} total)")
            urls = urls[:max_urls]
        
        logger.info(f"Processing {len(urls)} URLs")
        
        # Process URLs
        results = self.process_urls(urls, output_file)
        
        # Print summary
        if results:
            print(f"\n{'='*50}")
            print("INDEXING SUMMARY")
            print(f"{'='*50}")
            print(f"Total websites processed: {len(results)}")
            print(f"Results saved to: {output_file}")
            
            # Show samples
            print("\nSample results:")
            for i, result in enumerate(results[:3]):
                print(f"\n{i+1}. {result['title']}")
                print(f"   URL: {result['url']}")
                print(f"   Type: {result['content_type']}")
                print(f"   Keywords: {', '.join(result['keywords'][:3])}")


def main():
    """Main function."""
    # Set your Groq API key here
    GROQ_API_KEY = "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"
    
    if GROQ_API_KEY == "your_groq_api_key_here":
        print("Please set your Groq API key in the GROQ_API_KEY variable")
        return
    
    # Initialize and run indexer
    indexer = AIWebsiteIndexer(GROQ_API_KEY)
    indexer.run_indexer(
        csv_file="unimi_links.csv",
        output_file="indexed_websites.json",
        max_urls=100
    )


if __name__ == "__main__":
    main()


