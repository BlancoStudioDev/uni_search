#!/usr/bin/env python3

import requests
from bs4 import BeautifulSoup, Comment
import selenium.webdriver as webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException
import urllib.parse
import re
import time
import os
import json
from datetime import datetime
import hashlib
import sys

class FastWebScraper:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
        })
        
    def setup_selenium_driver(self):
        chrome_options = Options()
        chrome_options.add_argument('--headless')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--disable-gpu')
        chrome_options.add_argument('--disable-logging')
        chrome_options.add_argument('--disable-extensions')
        chrome_options.add_argument('--disable-images')
        chrome_options.add_argument('--disable-plugins')
        chrome_options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
        
        try:
            return webdriver.Chrome(options=chrome_options)
        except:
            try:
                firefox_options = webdriver.FirefoxOptions()
                firefox_options.add_argument('--headless')
                return webdriver.Firefox(options=firefox_options)
            except:
                return None

    def scrape_requests(self, url):
        try:
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
            return response.text
        except:
            return None

    def scrape_selenium(self, url):
        driver = self.setup_selenium_driver()
        if not driver:
            return None
        try:
            driver.get(url)
            WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
            return driver.page_source
        except:
            return None
        finally:
            driver.quit()

    def clean_html(self, html):
        if not html:
            return None
        soup = BeautifulSoup(html, 'lxml')
        for tag in ['script', 'style', 'iframe', 'noscript', 'meta', 'link']:
            for element in soup.find_all(tag):
                element.decompose()
        for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
            comment.extract()
        return soup

    def clean_text(self, text):
        if not text:
            return ""
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        text = re.sub(r'\.{4,}', '...', text)
        text = re.sub(r'-{4,}', '---', text)
        return text.strip()

    def extract_metadata(self, soup, url):
        parsed_url = urllib.parse.urlparse(url)
        metadata = {
            'url': url,
            'domain': parsed_url.netloc,
            'path': parsed_url.path,
            'scraped_at': datetime.now().isoformat(),
            'title': '',
            'description': '',
            'keywords': [],
            'language': '',
            'word_count': 0,
            'content_hash': ''
        }
        
        title_tag = soup.find('title')
        if title_tag:
            metadata['title'] = self.clean_text(title_tag.get_text())
        
        meta_mappings = {
            ('name', 'description'): 'description',
            ('name', 'keywords'): 'keywords',
            ('http-equiv', 'content-language'): 'language',
        }
        
        for meta_tag in soup.find_all('meta'):
            for (attr, value), key in meta_mappings.items():
                if meta_tag.get(attr) == value:
                    content = meta_tag.get('content', '').strip()
                    if content:
                        if key == 'keywords':
                            metadata[key] = [k.strip() for k in content.split(',') if k.strip()]
                        else:
                            metadata[key] = content
        
        full_text = soup.get_text()
        clean_full_text = self.clean_text(full_text)
        metadata['word_count'] = len(clean_full_text.split())
        metadata['content_hash'] = hashlib.md5(clean_full_text.encode()).hexdigest()
        
        return metadata

    def extract_sections(self, soup):
        sections = {'header': '', 'navigation': '', 'main': '', 'article': '', 'aside': '', 'footer': ''}
        section_selectors = {
            'header': ['header', '.header', '#header'],
            'navigation': ['nav', '.nav', '.menu'],
            'main': ['main', '.main', '.main-content'],
            'article': ['article', '.article', '.post'],
            'aside': ['aside', '.sidebar'],
            'footer': ['footer', '.footer']
        }
        
        for section_name, selectors in section_selectors.items():
            texts = []
            for selector in selectors:
                for elem in soup.select(selector):
                    text = self.clean_text(elem.get_text())
                    if text and len(text) > 10:
                        texts.append(text)
            sections[section_name] = '\n\n'.join(texts)
        
        return sections

    def extract_elements(self, soup):
        elements = {'headings': [], 'paragraphs': []}
        
        for i, tag in enumerate(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'], 1):
            for heading in soup.find_all(tag):
                text = self.clean_text(heading.get_text())
                if text:
                    elements['headings'].append({
                        'level': i,
                        'text': text
                    })
        
        for p in soup.find_all('p'):
            text = self.clean_text(p.get_text())
            if text and len(text) > 20:
                elements['paragraphs'].append({'text': text})
        
        return elements

    def extract_links(self, soup, base_url):
        base_domain = urllib.parse.urlparse(base_url).netloc
        links = {'internal': [], 'external': [], 'documents': []}
        
        for link in soup.find_all('a', href=True):
            href = link.get('href', '').strip()
            if not href or href.startswith(('mailto:', 'tel:', '#', 'javascript:')):
                continue
            
            try:
                absolute_url = urllib.parse.urljoin(base_url, href)
                parsed_url = urllib.parse.urlparse(absolute_url)
                domain = parsed_url.netloc.lower()
                path = parsed_url.path.lower()
                
                link_data = {
                    'url': absolute_url,
                    'text': self.clean_text(link.get_text()),
                    'domain': domain
                }
                
                doc_extensions = ['.pdf', '.doc', '.docx', '.xls', '.xlsx', '.zip', '.txt']
                if any(path.endswith(ext) for ext in doc_extensions):
                    links['documents'].append(link_data)
                elif domain == base_domain:
                    links['internal'].append(link_data)
                else:
                    links['external'].append(link_data)
            except:
                continue
        
        return links

    def extract_media(self, soup, base_url):
        return {'images': []}

    def create_record(self, soup, url):
        metadata = self.extract_metadata(soup, url)
        sections = self.extract_sections(soup)
        elements = self.extract_elements(soup)
        links = self.extract_links(soup, url)
        media = self.extract_media(soup, url)
        
        return {
            **metadata,
            'content_sections': sections,
            'structured_elements': elements,
            'links': links,
            'media': media,
            'stats': {
                'total_links': sum(len(link_list) for link_list in links.values()),
                'internal_links': len(links['internal']),
                'external_links': len(links['external']),
                'document_links': len(links['documents']),
                'total_images': len(media['images']),
                'total_headings': len(elements['headings']),
                'total_paragraphs': len(elements['paragraphs'])
            }
        }

    def save_jsonl(self, record, filename):
        try:
            with open(filename, 'a', encoding='utf-8') as f:
                json.dump(record, f, ensure_ascii=False, separators=(',', ':'))
                f.write('\n')
            return True
        except:
            return False

    def scrape(self, url, output_file):
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        
        html = self.scrape_requests(url)
        if not html or len(html) < 1000:
            html = self.scrape_selenium(url)
        
        if not html:
            return None
        
        soup = self.clean_html(html)
        if not soup:
            return None
        
        record = self.create_record(soup, url)
        
        if self.save_jsonl(record, output_file):
            return record
        return None

def batch_scrape(urls, output_file):
    scraper = FastWebScraper()
    results = []
    
    for i, url in enumerate(urls, 1):
        print(f"[{i}/{len(urls)}] {url}")
        result = scraper.scrape(url, output_file)
        if result:
            print(f"OK - {result['word_count']} words, {result['stats']['total_links']} links")
            results.append(result)
        else:
            print("FAILED")
    
    return results

def main():
    # Default output file for all scraping
    default_output = "scraped_data.jsonl"
    
    if len(sys.argv) < 2:
        print("Usage:")
        print(f"  python scraper.py <url>")
        print(f"  python scraper.py --batch <urls_file>")
        print(f"All data saved to: {default_output}")
        return
    
    if sys.argv[1] == '--batch':
        if len(sys.argv) < 3:
            print(f"Batch mode requires: python scraper.py --batch <urls_file>")
            return
        
        urls_file = sys.argv[2]
        
        try:
            with open(urls_file, 'r') as f:
                urls = [line.strip() for line in f if line.strip()]
            
            results = batch_scrape(urls, default_output)
            print(f"\nCompleted: {len(results)}/{len(urls)} pages scraped to {default_output}")
        except Exception as e:
            print(f"Error: {e}")
    
    else:
        url = sys.argv[1]
        
        scraper = FastWebScraper()
        result = scraper.scrape(url, default_output)
        
        if result:
            print(f"Success: {result['word_count']} words, {result['stats']['total_links']} links")
            print(f"Saved to: {default_output}")
        else:
            print("Failed to scrape")

if __name__ == "__main__":
    main()