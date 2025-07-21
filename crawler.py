import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import time
import csv
from collections import deque
import logging

class WebCrawler:
    def __init__(self, base_url, max_pages=500, delay=1):
        self.base_url = base_url
        self.domain = urlparse(base_url).netloc
        self.max_pages = max_pages
        self.delay = delay
        
        # Data structures to track progress
        self.visited_urls = set()
        self.all_links = set()
        self.to_visit = deque([base_url])
        self.failed_urls = []
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Session for connection pooling
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
    
    def is_valid_url(self, url):
        """Check if URL belongs to the target domain"""
        parsed = urlparse(url)
        return parsed.netloc == self.domain
    
    def normalize_url(self, url):
        """Remove fragments and normalize URL"""
        parsed = urlparse(url)
        return f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
    
    def get_page_content(self, url):
        """Fetch page content with error handling"""
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            return response.text
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error fetching {url}: {e}")
            self.failed_urls.append(url)
            return None
    
    def extract_links(self, html_content, base_url):
        """Extract all links from HTML content"""
        soup = BeautifulSoup(html_content, 'html.parser')
        links = set()
        
        # Find all anchor tags with href attributes
        for link in soup.find_all('a', href=True):
            href = link['href']
            full_url = urljoin(base_url, href)
            normalized_url = self.normalize_url(full_url)
            
            if self.is_valid_url(normalized_url):
                links.add(normalized_url)
        
        return links
    
    def crawl(self):
        """Main crawling function"""
        pages_crawled = 0
        
        self.logger.info(f"Starting crawl of {self.base_url}")
        
        while self.to_visit and pages_crawled < self.max_pages:
            current_url = self.to_visit.popleft()
            
            if current_url in self.visited_urls:
                continue
            
            self.logger.info(f"Crawling ({pages_crawled + 1}/{self.max_pages}): {current_url}")
            
            # Fetch page content
            html_content = self.get_page_content(current_url)
            if not html_content:
                continue
            
            # Mark as visited
            self.visited_urls.add(current_url)
            pages_crawled += 1
            
            # Extract links
            page_links = self.extract_links(html_content, current_url)
            self.all_links.update(page_links)
            
            # Add new links to queue
            for link in page_links:
                if link not in self.visited_urls:
                    self.to_visit.append(link)
            
            # Be respectful - add delay between requests
            time.sleep(self.delay)
        
        self.logger.info(f"Crawling completed. Visited {len(self.visited_urls)} pages")
        self.logger.info(f"Found {len(self.all_links)} unique links")
    
    def save_results(self, filename="unimi_links.csv"):
        """Save results to CSV file"""
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['URL', 'Status'])
            
            for url in sorted(self.all_links):
                status = 'Visited' if url in self.visited_urls else 'Discovered'
                writer.writerow([url, status])
        
        self.logger.info(f"Results saved to {filename}")
    
    def print_summary(self):
        """Print crawling summary"""
        print(f"\n=== CRAWLING SUMMARY ===")
        print(f"Base URL: {self.base_url}")
        print(f"Pages visited: {len(self.visited_urls)}")
        print(f"Total unique links found: {len(self.all_links)}")
        print(f"Failed URLs: {len(self.failed_urls)}")
        
        if self.failed_urls:
            print(f"\nFailed URLs:")
            for url in self.failed_urls[:10]:  # Show first 10 failed URLs
                print(f"  - {url}")
            if len(self.failed_urls) > 10:
                print(f"  ... and {len(self.failed_urls) - 10} more")
        
        print(f"\nSample of discovered links:")
        for i, url in enumerate(sorted(self.all_links)[:10]):
            print(f"  {i+1}. {url}")
        if len(self.all_links) > 10:
            print(f"  ... and {len(self.all_links) - 10} more")

def main():
    # Configuration
    base_url = "https://www.unimi.it/it"
    max_pages = 10000  # Adjust based on your needs
    delay = 1  # Delay between requests (seconds)
    
    # Create and run crawler
    crawler = WebCrawler(base_url, max_pages, delay)
    crawler.crawl()
    
    # Save results and print summary
    crawler.save_results()
    crawler.print_summary()

if __name__ == "__main__":
    main()