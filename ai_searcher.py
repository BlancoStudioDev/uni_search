import json
import re
from datetime import datetime
from typing import Dict, List, Optional, Any
from groq import Groq
import logging
from fuzzywuzzy import fuzz
from collections import defaultdict
import os

# Configurazione logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SearchBot:
    def __init__(self, groq_api_key: str, json_file_path: str):
        """
        Inizializza il bot di ricerca
        
        Args:
            groq_api_key: Chiave API di Groq
            json_file_path: Percorso del file JSON con i dati indicizzati
        """
        self.groq_client = Groq(api_key=groq_api_key)
        self.json_file_path = json_file_path
        self.indexed_data = []
        self.load_json_data()
    
    def load_json_data(self):
        """
        Carica i dati dal file JSON
        """
        try:
            with open(self.json_file_path, 'r', encoding='utf-8') as f:
                self.indexed_data = json.load(f)
            logger.info(f"Caricati {len(self.indexed_data)} elementi dal file JSON")
        except Exception as e:
            logger.error(f"Errore nel caricare il file JSON: {str(e)}")
            self.indexed_data = []
    
    def search_by_keywords(self, query: str, threshold: int = 70) -> List[Dict]:
        """
        Cerca nei dati usando le parole chiave
        
        Args:
            query: Query di ricerca
            threshold: Soglia di similarità (0-100)
            
        Returns:
            Lista di risultati ordinati per rilevanza
        """
        results = []
        query_lower = query.lower()
        
        for item in self.indexed_data:
            score = 0
            matches = []
            
            # Cerca nelle keywords
            for keyword in item.get('keywords', []):
                similarity = fuzz.ratio(query_lower, keyword.lower())
                if similarity > threshold:
                    score += similarity * 2  # Peso maggiore per keywords
                    matches.append(f"keyword: {keyword}")
            
            # Cerca nella descrizione
            description = item.get('description', '')
            if description:
                similarity = fuzz.partial_ratio(query_lower, description.lower())
                if similarity > threshold:
                    score += similarity
                    matches.append(f"description: {description[:100]}...")
            
            # Cerca nel titolo
            title = item.get('title', '')
            if title:
                similarity = fuzz.partial_ratio(query_lower, title.lower())
                if similarity > threshold:
                    score += similarity * 1.5  # Peso intermedio per titolo
                    matches.append(f"title: {title}")
            
            # Cerca nei main_topics
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
        
        # Ordina per punteggio
        results.sort(key=lambda x: x['search_score'], reverse=True)
        return results
    
    def filter_by_criteria(self, criteria: Dict) -> List[Dict]:
        """
        Filtra i dati per criteri specifici
        
        Args:
            criteria: Dizionario con i criteri di filtro
            
        Returns:
            Lista di elementi filtrati
        """
        filtered = []
        
        for item in self.indexed_data:
            match = True
            
            # Filtra per tipo di contenuto
            if criteria.get('content_type') and item.get('content_type') != criteria['content_type']:
                match = False
            
            # Filtra per lingua
            if criteria.get('language') and item.get('language') != criteria['language']:
                match = False
            
            # Filtra per qualità
            if criteria.get('content_quality') and item.get('content_quality') != criteria['content_quality']:
                match = False
            
            # Filtra per target audience
            if criteria.get('target_audience') and item.get('target_audience') != criteria['target_audience']:
                match = False
            
            # Filtra per sentiment
            if criteria.get('sentiment') and item.get('sentiment') != criteria['sentiment']:
                match = False
            
            # Filtra per punteggio di rilevanza minimo
            if criteria.get('min_relevance_score'):
                if item.get('relevance_score', 0) < criteria['min_relevance_score']:
                    match = False
            
            if match:
                filtered.append(item)
        
        return filtered
    
    def get_statistics_summary(self) -> Dict:
        """
        Genera un riassunto statistico dei dati
        
        Returns:
            Dizionario con statistiche
        """
        if not self.indexed_data:
            return {}
        
        stats = {
            'total_pages': len(self.indexed_data),
            'content_types': defaultdict(int),
            'languages': defaultdict(int),
            'quality_distribution': defaultdict(int),
            'sentiment_distribution': defaultdict(int),
            'target_audiences': defaultdict(int),
            'avg_relevance_score': 0,
            'top_keywords': defaultdict(int),
            'top_topics': defaultdict(int)
        }
        
        total_relevance = 0
        
        for item in self.indexed_data:
            stats['content_types'][item.get('content_type', 'unknown')] += 1
            stats['languages'][item.get('language', 'unknown')] += 1
            stats['quality_distribution'][item.get('content_quality', 'unknown')] += 1
            stats['sentiment_distribution'][item.get('sentiment', 'unknown')] += 1
            stats['target_audiences'][item.get('target_audience', 'unknown')] += 1
            
            total_relevance += item.get('relevance_score', 0)
            
            # Conta keywords e topics
            for keyword in item.get('keywords', []):
                stats['top_keywords'][keyword.lower()] += 1
            
            for topic in item.get('main_topics', []):
                stats['top_topics'][topic.lower()] += 1
        
        stats['avg_relevance_score'] = total_relevance / len(self.indexed_data)
        
        # Converti in dict normali e prendi i top 10
        stats['content_types'] = dict(stats['content_types'])
        stats['languages'] = dict(stats['languages'])
        stats['quality_distribution'] = dict(stats['quality_distribution'])
        stats['sentiment_distribution'] = dict(stats['sentiment_distribution'])
        stats['target_audiences'] = dict(stats['target_audiences'])
        stats['top_keywords'] = dict(sorted(stats['top_keywords'].items(), key=lambda x: x[1], reverse=True)[:10])
        stats['top_topics'] = dict(sorted(stats['top_topics'].items(), key=lambda x: x[1], reverse=True)[:10])
        
        return stats
    
    def analyze_query_with_ai(self, user_query: str, search_results: List[Dict]) -> str:
        """
        Analizza la query dell'utente e i risultati con l'AI per generare una risposta dettagliata
        
        Args:
            user_query: Domanda dell'utente
            search_results: Risultati della ricerca
            
        Returns:
            Risposta dettagliata generata dall'AI
        """
        try:
            # Prepara i dati per l'AI
            results_summary = []
            for i, result in enumerate(search_results[:10]):  # Limita a 10 risultati
                summary = {
                    'url': result['url'],
                    'title': result['title'],
                    'description': result['description'],
                    'keywords': result['keywords'],
                    'content_type': result['content_type'],
                    'main_topics': result['main_topics'],
                    'relevance_score': result['relevance_score'],
                    'matches': result.get('matches', [])
                }
                results_summary.append(summary)
            
            # Statistiche generali
            stats = self.get_statistics_summary()
            
            # Prompt per l'AI
            prompt = f"""
Sei un assistente esperto che aiuta gli utenti a trovare informazioni in un database di pagine web indicizzate.

DOMANDA DELL'UTENTE: {user_query}

RISULTATI DELLA RICERCA ({len(search_results)} risultati totali):
{json.dumps(results_summary, indent=2, ensure_ascii=False)}

STATISTICHE GENERALI DEL DATABASE:
{json.dumps(stats, indent=2, ensure_ascii=False)}

ISTRUZIONI:
1. Analizza la domanda dell'utente e i risultati della ricerca
2. Fornisci una risposta dettagliata e strutturata che includa:
   - Una risposta diretta alla domanda
   - I migliori risultati pertinenti con spiegazioni
   - Link specifici che l'utente dovrebbe visitare
   - Suggerimenti per affinare la ricerca se necessario
   - Azioni concrete che l'utente può intraprendere

3. Struttura la risposta in modo professionale e utile
4. Se non ci sono risultati sufficienti, suggerisci alternative o modifiche alla ricerca
5. Includi sempre gli URL più rilevanti nella risposta

FORMATO DELLA RISPOSTA:
- Usa intestazioni chiare
- Elenca i risultati più rilevanti
- Fornisci azioni concrete
- Scrivi in italiano
"""

            # Chiamata all'AI
            response = self.groq_client.chat.completions.create(
                model="deepseek-r1-distill-llama-70b",
                messages=[
                    {"role": "system", "content": "Sei un assistente esperto per la ricerca di informazioni web. Fornisci sempre risposte dettagliate, strutturate e utili."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=2000
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Errore nell'analisi AI: {str(e)}")
            return f"Errore nell'analisi della domanda: {str(e)}"
    
    def search_and_answer(self, user_query: str, max_results: int = 20) -> Dict:
        """
        Cerca informazioni e genera una risposta completa
        
        Args:
            user_query: Domanda dell'utente
            max_results: Numero massimo di risultati da considerare
            
        Returns:
            Dizionario con risultati e risposta AI
        """
        logger.info(f"Processando query: {user_query}")
        
        # Cerca nei dati
        search_results = self.search_by_keywords(user_query)[:max_results]
        
        # Genera risposta AI
        ai_response = self.analyze_query_with_ai(user_query, search_results)
        
        return {
            'query': user_query,
            'results_count': len(search_results),
            'search_results': search_results,
            'ai_response': ai_response,
            'timestamp': datetime.now().isoformat()
        }
    
    def save_response_to_file(self, response_data: Dict, output_format: str = 'txt') -> str:
        """
        Salva la risposta in un file
        
        Args:
            response_data: Dati della risposta
            output_format: Formato del file ('txt' o 'doc')
            
        Returns:
            Percorso del file creato
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        query_clean = re.sub(r'[^\w\s-]', '', response_data['query'])[:50]
        filename = f"risposta_{query_clean}_{timestamp}.{output_format}"
        
        try:
            if output_format == 'txt':
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(f"DOMANDA: {response_data['query']}\n")
                    f.write(f"DATA: {response_data['timestamp']}\n")
                    f.write(f"RISULTATI TROVATI: {response_data['results_count']}\n")
                    f.write("=" * 80 + "\n\n")
                    f.write("RISPOSTA:\n")
                    f.write(response_data['ai_response'])
                    f.write("\n\n" + "=" * 80 + "\n")
                    f.write("DETTAGLI RISULTATI:\n\n")
                    
                    for i, result in enumerate(response_data['search_results'][:5], 1):
                        f.write(f"{i}. {result['title']}\n")
                        f.write(f"   URL: {result['url']}\n")
                        f.write(f"   Descrizione: {result['description']}\n")
                        f.write(f"   Keywords: {', '.join(result['keywords'])}\n")
                        f.write(f"   Tipo: {result['content_type']}\n")
                        f.write(f"   Punteggio: {result.get('search_score', 0)}\n\n")
            
            elif output_format == 'doc':
                # Per un file .doc semplice (RTF)
                with open(filename.replace('.doc', '.rtf'), 'w', encoding='utf-8') as f:
                    f.write(r'{\rtf1\ansi\deff0 {\fonttbl {\f0 Times New Roman;}}')
                    f.write(r'\f0\fs24 ')
                    f.write(f"DOMANDA: {response_data['query']}\\par ")
                    f.write(f"DATA: {response_data['timestamp']}\\par ")
                    f.write(f"RISULTATI TROVATI: {response_data['results_count']}\\par\\par ")
                    f.write("RISPOSTA:\\par ")
                    f.write(response_data['ai_response'].replace('\n', '\\par '))
                    f.write('}')
                filename = filename.replace('.doc', '.rtf')
            
            logger.info(f"Risposta salvata in: {filename}")
            return filename
            
        except Exception as e:
            logger.error(f"Errore nel salvare il file: {str(e)}")
            return ""


# Esempio di utilizzo
def main():
    # Configurazione
    GROQ_API_KEY = "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"  # Sostituisci con la tua chiave API
    JSON_FILE = "indexed_content.json"  # File JSON generato dal bot precedente
    
    # Inizializza il bot di ricerca
    search_bot = SearchBot(groq_api_key=GROQ_API_KEY, json_file_path=JSON_FILE)
    
    # Esempi di domande
    example_queries = [
        "Come posso iscrivermi all'università?",
        "Quali sono i corsi di laurea disponibili?",
        "Informazioni sulle borse di studio",
        "Dove posso trovare informazioni sui servizi per studenti?",
        "Quali sono i requisiti di ammissione?"
    ]
    
    print("=== SEARCH BOT DEMO ===")
    print(f"Database caricato: {len(search_bot.indexed_data)} pagine")
    print("\nEsempi di domande:")
    for i, query in enumerate(example_queries, 1):
        print(f"{i}. {query}")
    
    # Modalità interattiva
    while True:
        print("\n" + "="*50)
        user_query = input("Inserisci la tua domanda (o 'quit' per uscire): ")
        
        if user_query.lower() in ['quit', 'exit', 'q']:
            break
        
        if not user_query.strip():
            continue
        
        # Cerca e genera risposta
        response = search_bot.search_and_answer(user_query)
        
        # Mostra risultati
        print(f"\nTrovati {response['results_count']} risultati per: '{user_query}'")
        print("\nRISPOSTA AI:")
        print(response['ai_response'])
        
        # Chiedi se salvare
        save_choice = input("\nVuoi salvare la risposta in un file? (y/n): ")
        if save_choice.lower() == 'y':
            format_choice = input("Formato (txt/doc): ").lower()
            if format_choice not in ['txt', 'doc']:
                format_choice = 'txt'
            
            filename = search_bot.save_response_to_file(response, format_choice)
            if filename:
                print(f"Risposta salvata in: {filename}")


if __name__ == "__main__":
    main()