import asyncio
import csv
import json
import re
from datetime import datetime
from typing import Dict, List, Optional, Set
import aiohttp
import pandas as pd
from groq import Groq
import logging
from urllib.parse import urljoin, urlparse
import time
from bs4 import BeautifulSoup
import requests
from markdownify import markdownify as md
import os

# Configurazione logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WebIndexerBot:
    def __init__(self, groq_api_key: str, max_concurrent: int = 5, max_links: int = 1, cooldown: float = 1.0):
        """
        Inizializza il bot di indicizzazione web
        
        Args:
            groq_api_key: Chiave API di Groq
            max_concurrent: Numero massimo di richieste simultanee
            max_links: Numero massimo di link da analizzare per sessione
            cooldown: Tempo di attesa in secondi tra le richieste
        """
        self.groq_client = Groq(api_key=groq_api_key)
        self.max_concurrent = max_concurrent
        self.max_links = max_links
        self.cooldown = cooldown
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.results = []
        self.indexed_urls: Set[str] = set()  # Tracking URLs già indicizzati
        
    def load_existing_results(self, output_file: str) -> List[Dict]:
        """
        Carica i risultati esistenti dal file JSON se esiste
        
        Args:
            output_file: Percorso del file JSON esistente
            
        Returns:
            Lista dei risultati esistenti
        """
        if not os.path.exists(output_file):
            logger.info(f"File {output_file} non esiste. Iniziando da zero.")
            return []
        
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                existing_results = json.load(f)
                
            # Estrai gli URL già indicizzati
            for result in existing_results:
                if 'url' in result:
                    self.indexed_urls.add(result['url'].strip())
            
            logger.info(f"Caricati {len(existing_results)} risultati esistenti")
            logger.info(f"URLs già indicizzati: {len(self.indexed_urls)}")
            
            return existing_results
            
        except Exception as e:
            logger.error(f"Errore nel caricare il file esistente {output_file}: {str(e)}")
            return []
    
    def find_last_indexed_position(self, csv_file_path: str, url_column: str = 'url') -> int:
        """
        Trova la posizione dell'ultimo URL indicizzato nel CSV
        
        Args:
            csv_file_path: Percorso del file CSV
            url_column: Nome della colonna contenente gli URL
            
        Returns:
            Indice dell'ultimo URL indicizzato (-1 se nessuno trovato)
        """
        try:
            df = pd.read_csv(csv_file_path)
            urls = df[url_column].dropna().tolist()
            
            # Trova l'ultimo URL indicizzato
            last_index = -1
            for i, url in enumerate(urls):
                if isinstance(url, str) and url.strip() in self.indexed_urls:
                    last_index = i
            
            logger.info(f"Ultimo URL indicizzato trovato alla posizione: {last_index}")
            return last_index
            
        except Exception as e:
            logger.error(f"Errore nel trovare l'ultima posizione indicizzata: {str(e)}")
            return -1
    
    def save_progress_checkpoint(self, output_file: str, current_results: List[Dict]):
        """
        Salva un checkpoint dei progressi durante l'elaborazione
        
        Args:
            output_file: Percorso del file di output
            current_results: Risultati correnti da salvare
        """
        checkpoint_file = output_file.replace('.json', '_checkpoint.json')
        try:
            with open(checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(current_results, f, indent=2, ensure_ascii=False)
            logger.info(f"Checkpoint salvato: {len(current_results)} risultati")
        except Exception as e:
            logger.error(f"Errore nel salvare checkpoint: {str(e)}")

    async def extract_clean_content(self, url: str) -> Optional[Dict]:
        """
        Estrae il contenuto pulito da un URL usando requests + BeautifulSoup
        
        Args:
            url: URL da analizzare
            
        Returns:
            Dizionario con contenuto pulito o None se errore
        """
        try:
            # Headers per sembrare un browser normale
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'it-IT,it;q=0.9,en;q=0.8',
                'Accept-Encoding': 'gzip, deflate, br',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1'
            }
            
            # Fai la richiesta HTTP
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
                async with session.get(url, headers=headers) as response:
                    if response.status != 200:
                        logger.error(f"HTTP {response.status} per {url}")
                        return None
                    
                    html_content = await response.text()
            
            # Parsifica con BeautifulSoup
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Rimuovi script, style e altri elementi non necessari
            for script in soup(["script", "style", "nav", "header", "footer", "aside", "noscript"]):
                script.decompose()
            
            # Estrai il titolo
            title = soup.find('title')
            title_text = title.get_text().strip() if title else ''
            
            # Estrai il contenuto principale
            # Cerca prima contenitori comuni per il contenuto principale
            main_content = None
            for selector in ['main', 'article', '.content', '#content', '.main-content', '.post-content']:
                main_content = soup.select_one(selector)
                if main_content:
                    break
            
            # Se non trova un contenitore principale, usa il body
            if not main_content:
                main_content = soup.find('body')
            
            if not main_content:
                logger.error(f"Nessun contenuto trovato per {url}")
                return None
            
            # Converti in markdown
            markdown_content = md(str(main_content))
            
            # Pulisci il markdown
            markdown_content = re.sub(r'\n\s*\n', '\n\n', markdown_content)  # Rimuovi righe vuote eccessive
            markdown_content = re.sub(r'\s+', ' ', markdown_content)  # Normalizza spazi
            
            # Estrai i link interni
            internal_links = []
            base_domain = urlparse(url).netloc
            
            for link in main_content.find_all('a', href=True):
                href = link.get('href')
                text = link.get_text().strip()
                
                if href and text:
                    # Converti link relativi in assoluti
                    if href.startswith('/'):
                        href = urljoin(url, href)
                    
                    # Controlla se è un link interno
                    if base_domain in href or href.startswith('/'):
                        internal_links.append({
                            'url': href,
                            'text': text
                        })
            
            return {
                'url': url,
                'title': title_text,
                'markdown_content': markdown_content[:8000] if markdown_content else '',  # Limita a 8k caratteri
                'internal_links': internal_links[:20],  # Limita a 20 link
                'extraction_time': datetime.now().isoformat(),
                'success': True
            }
                    
        except Exception as e:
            logger.error(f"Errore durante l'estrazione da {url}: {str(e)}")
            return None
    
    def analyze_with_groq(self, content_data: Dict) -> Dict:
        """
        Analizza il contenuto con Groq AI per estrarre informazioni strutturate
        
        Args:
            content_data: Dati del contenuto estratto
            
        Returns:
            Dizionario con analisi strutturata
        """
        try:
            # Prepara il prompt per Groq
            prompt = f"""
Analizza il seguente contenuto web e fornisci una risposta in formato JSON con le seguenti informazioni:

URL: {content_data['url']}
Titolo: {content_data['title']}
Contenuto: {content_data['markdown_content']}

Fornisci la risposta in questo formato JSON:
{{
    "keywords": ["keyword1", "keyword2", "keyword3"],
    "description": "Breve descrizione del contenuto (max 200 caratteri)",
    "relevance_score": 0.8,
    "content_type": "article|product|service|homepage|about|contact|other",
    "main_topics": ["topic1", "topic2"],
    "language": "it|en|other",
    "sentiment": "positive|negative|neutral",
    "target_audience": "general|technical|business|educational",
    "content_quality": "high|medium|low"
}}

Estrai solo le informazioni più rilevanti e significative. Le parole chiave devono essere specifiche e utili per la ricerca.
"""

            # Salva il contenuto del primo link per debug
            if not hasattr(self, 'first_link_debug_content'):
                self.first_link_debug_content = {
                    'url': content_data['url'],
                    'title': content_data['title'],
                    'markdown_content': content_data['markdown_content'],
                    'prompt_sent_to_ai': prompt,
                    'extraction_time': content_data['extraction_time']
                }

            # Chiamata a Groq
            response = self.groq_client.chat.completions.create(
                model="llama3-70b-8192",
                messages=[
                    {"role": "system", "content": "Sei un esperto analista di contenuti web. Rispondi sempre in formato JSON valido."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )
            
            # Estrai la risposta JSON
            response_text = response.choices[0].message.content.strip()
            
            # Salva anche la risposta dell'AI per il primo link
            if hasattr(self, 'first_link_debug_content') and 'ai_response' not in self.first_link_debug_content:
                self.first_link_debug_content['ai_response'] = response_text
            
            # Cerca di estrarre JSON dalla risposta
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                analysis = json.loads(json_str)
            else:
                # Fallback se non trova JSON
                analysis = {
                    "keywords": [],
                    "description": "Contenuto non analizzabile",
                    "relevance_score": 0.1,
                    "content_type": "other",
                    "main_topics": [],
                    "language": "unknown",
                    "sentiment": "neutral",
                    "target_audience": "general",
                    "content_quality": "low"
                }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Errore nell'analisi Groq per {content_data['url']}: {str(e)}")
            return {
                "keywords": [],
                "description": "Errore nell'analisi",
                "relevance_score": 0.0,
                "content_type": "other",
                "main_topics": [],
                "language": "unknown",
                "sentiment": "neutral",
                "target_audience": "general",
                "content_quality": "low"
            }
    
    async def process_url(self, url: str) -> Optional[Dict]:
        """
        Processa un singolo URL: estrae contenuto e analizza con AI
        
        Args:
            url: URL da processare
            
        Returns:
            Dizionario con tutti i dati processati
        """
        async with self.semaphore:  # Limita le richieste concorrenti
            logger.info(f"Processando: {url}")
            
            # Estrai contenuto pulito
            content_data = await self.extract_clean_content(url)
            if not content_data:
                return None
            
            # Cooldown tra le richieste
            await asyncio.sleep(self.cooldown)
            
            # Analizza con Groq (operazione sincrona)
            analysis = self.analyze_with_groq(content_data)
            
            # Combina tutti i dati
            result = {
                "url": url,
                "title": content_data["title"],
                "keywords": analysis["keywords"],
                "description": analysis["description"],
                "relevance_score": analysis["relevance_score"],
                "content_type": analysis["content_type"],
                "main_topics": analysis["main_topics"],
                "language": analysis["language"],
                "sentiment": analysis["sentiment"],
                "target_audience": analysis["target_audience"],
                "content_quality": analysis["content_quality"],
                "internal_links": content_data["internal_links"],
                "indexed_at": datetime.now().isoformat(),
                "word_count": len(content_data["markdown_content"].split()) if content_data["markdown_content"] else 0
            }
            
            logger.info(f"Completato: {url}")
            return result
    
    async def process_csv_file(self, csv_file_path: str, output_file: str, url_column: str = 'url') -> List[Dict]:
        """
        Processa tutti gli URL da un file CSV, riprendendo dall'ultimo punto
        
        Args:
            csv_file_path: Percorso del file CSV
            output_file: Percorso del file di output per il tracking
            url_column: Nome della colonna contenente gli URL
            
        Returns:
            Lista di dizionari con i risultati
        """
        logger.info(f"Caricamento URLs da {csv_file_path}")
        
        # Carica risultati esistenti
        existing_results = self.load_existing_results(output_file)
        
        # Leggi il CSV
        try:
            df = pd.read_csv(csv_file_path)
            urls = df[url_column].dropna().tolist()
            logger.info(f"Trovati {len(urls)} URLs totali nel CSV")
            
            # Trova la posizione di partenza
            last_index = self.find_last_indexed_position(csv_file_path, url_column)
            start_index = last_index + 1
            
            # Estrai solo gli URL non ancora processati
            remaining_urls = []
            for i, url in enumerate(urls[start_index:], start_index):
                if url and isinstance(url, str) and url.strip() not in self.indexed_urls:
                    remaining_urls.append((i, url.strip()))
            
            logger.info(f"URLs rimanenti da processare: {len(remaining_urls)}")
            
            # Limita il numero di URL da processare in questa sessione
            if len(remaining_urls) > self.max_links:
                remaining_urls = remaining_urls[:self.max_links]
                logger.info(f"Limitato a {self.max_links} URLs per questa sessione")
            
            if not remaining_urls:
                logger.info("Tutti gli URLs sono già stati processati!")
                return existing_results
            
        except Exception as e:
            logger.error(f"Errore nel leggere il CSV: {str(e)}")
            return existing_results
        
        # Processa gli URL rimanenti
        tasks = []
        for position, url in remaining_urls:
            logger.info(f"Aggiungendo alla coda (posizione CSV {position}): {url}")
            task = self.process_url(url)
            tasks.append(task)
        
        # Esegui tutte le task con checkpoint periodici
        logger.info(f"Iniziando elaborazione di {len(tasks)} URLs...")
        new_results = []
        all_results = existing_results.copy()
        
        # Processa in batch con checkpoint
        batch_size = min(10, len(tasks))
        for i in range(0, len(tasks), batch_size):
            batch_tasks = tasks[i:i + batch_size]
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            # Processa risultati del batch
            for result in batch_results:
                if isinstance(result, dict) and result is not None:
                    new_results.append(result)
                    all_results.append(result)
                    self.indexed_urls.add(result['url'])
                elif isinstance(result, Exception):
                    logger.error(f"Errore durante il processing: {str(result)}")
            
            # Salva checkpoint
            if new_results:
                self.save_progress_checkpoint(output_file, all_results)
            
            logger.info(f"Batch completato. Nuovi risultati: {len(new_results)}, Totale: {len(all_results)}")
        
        self.results = all_results
        logger.info(f"Elaborazione completata. Nuovi risultati: {len(new_results)}, Totale: {len(all_results)}")
        return all_results
    
    def save_to_json(self, output_file: str, pretty_print: bool = True):
        """
        Salva i risultati in formato JSON
        
        Args:
            output_file: Percorso del file di output
            pretty_print: Se formattare il JSON in modo leggibile
        """
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                if pretty_print:
                    json.dump(self.results, f, indent=2, ensure_ascii=False)
                else:
                    json.dump(self.results, f, ensure_ascii=False)
            
            logger.info(f"Risultati salvati in {output_file}")
            logger.info(f"Totale pagine indicizzate: {len(self.results)}")
            
            # Rimuovi il file checkpoint se esiste
            checkpoint_file = output_file.replace('.json', '_checkpoint.json')
            if os.path.exists(checkpoint_file):
                os.remove(checkpoint_file)
                logger.info("File checkpoint rimosso dopo salvataggio finale")
            
        except Exception as e:
            logger.error(f"Errore nel salvare il JSON: {str(e)}")
    
    def get_statistics(self) -> Dict:
        """
        Restituisce statistiche sui risultati processati
        
        Returns:
            Dizionario con le statistiche
        """
        if not self.results:
            return {}
        
        total_pages = len(self.results)
        content_types = {}
        languages = {}
        avg_relevance = 0
        quality_distribution = {}
        
        for result in self.results:
            # Conta i tipi di contenuto
            content_type = result.get('content_type', 'unknown')
            content_types[content_type] = content_types.get(content_type, 0) + 1
            
            # Conta le lingue
            language = result.get('language', 'unknown')
            languages[language] = languages.get(language, 0) + 1
            
            # Calcola rilevanza media
            avg_relevance += result.get('relevance_score', 0)
            
            # Distribuzione qualità
            quality = result.get('content_quality', 'unknown')
            quality_distribution[quality] = quality_distribution.get(quality, 0) + 1
        
        avg_relevance = avg_relevance / total_pages if total_pages > 0 else 0
        
        stats = {
            'total_pages': total_pages,
            'content_types': content_types,
            'languages': languages,
            'average_relevance_score': round(avg_relevance, 2),
            'quality_distribution': quality_distribution,
            'generated_at': datetime.now().isoformat(),
            'total_indexed_urls': len(self.indexed_urls)
        }
        
        # Aggiungi il contenuto del primo link per debug
        if self.results and hasattr(self, 'first_link_debug_content'):
            stats['first_link_debug'] = self.first_link_debug_content
        
        return stats


# Esempio di utilizzo
async def main():
    # Configurazione
    GROQ_API_KEY = "GROQ_API_KEY"  # Sostituisci con la tua chiave API
    CSV_FILE = "unimi_links.csv"  # Il tuo file CSV con i link
    OUTPUT_FILE = "indexed_content.json"
    STATS_FILE = "indexing_stats.json"
    
    # Inizializza il bot
    bot = WebIndexerBot(
        groq_api_key=GROQ_API_KEY, 
        max_concurrent=3,
        max_links=400, 
        cooldown=2
    )
    
    # Processa tutti gli URL (riprende dall'ultimo)
    results = await bot.process_csv_file(CSV_FILE, OUTPUT_FILE, url_column='URL')
    
    # Salva i risultati
    bot.save_to_json(OUTPUT_FILE)
    
    # Genera e salva statistiche
    stats = bot.get_statistics()
    with open(STATS_FILE, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    print(f"Indicizzazione completata!")
    print(f"Pagine totali nel database: {len(results)}")
    print(f"URLs unici indicizzati: {len(bot.indexed_urls)}")
    print(f"Risultati salvati in: {OUTPUT_FILE}")
    print(f"Statistiche salvate in: {STATS_FILE}")

if __name__ == "__main__":
    asyncio.run(main())