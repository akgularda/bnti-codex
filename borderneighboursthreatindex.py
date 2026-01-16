import feedparser
from datetime import datetime, timedelta
from dateutil import parser as date_parser
from transformers import pipeline
import os
import pandas as pd
import concurrent.futures
import time
import logging
import math
import json
import socket
import numpy as np
from googletrans import Translator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set socket timeout to prevent hanging on bad feeds
socket.setdefaulttimeout(10)

class BNTIAnalyzer:
    def __init__(self):
        self.output_path = os.getcwd()
        self.history_file = os.path.join(self.output_path, "bnti_history.csv")
        
        # TRANSLATOR (For Report Summaries Only)
        self.translator = Translator()

        logger.info("Loading Multilingual Zero-Shot Model (XLM-RoBERTa)...")
        self.classifier = pipeline("zero-shot-classification", model="joeddav/xlm-roberta-large-xnli")
        
        # SCIENTIFIC WEIGHTING SYSTEM (Modified Goldstein Scale)
        self.category_weights = {
            "military conflict": 10.0,    # WAR / KINETIC
            "terrorist act": 9.0,         # TERROR
            "violent protest": 7.0,       # RIOT / CIVIL UNREST
            "political crisis": 6.0,      # DIPLOMATIC TENSION
            "economic crisis": 4.0,       # MARKET CRASH/INFLATION
            "humanitarian crisis": 3.0,   # REFUGEE/DISASTER
            "peaceful diplomacy": -2.0,   # TREATY/ALLIANCE
            "neutral news": 0.0           # NOISE
        }
        
        self.candidate_labels = list(self.category_weights.keys())

        # RSS Feeds Configuration (Expanded with 15+ backup feeds for resilience)
        self.rss_urls = {
            "Armenia": [
                "https://a1plus.am/en/rss",
                "https://hetq.am/en/rss",
                "https://armenianweekly.com/feed",
                "https://massispost.com/feed",
                "https://asbarez.com/feed",
                "https://www.azatutyun.am/api/z-pqp_eqypt"  # RFE/RL Armenia
            ],
            "Georgia": [
                "https://civil.ge/feed", 
                "https://georgiatoday.ge/feed",
                "https://jam-news.net/feed",
                "https://oc-media.org/feed",  # Open Caucasus Media
                "https://www.rferl.org/api/zktpyei_tgt"  # RFE/RL Georgia
            ],
            "Greece": [
                "https://www.in.gr/feed/?rid=2&pid=250&la=1&si=1", 
                "https://feeds.feedburner.com/newsbombgr",
                "https://www.naftemporiki.gr/feed",
                "https://www.protothema.gr/rss",
                "https://gtp.gr/rss.asp",  # GTP Headlines
                "https://www.keeptalkinggreece.com/feed"
            ],
            "Iran": [
                "https://www.tehrantimes.com/rss",
                "https://www.tehrantimes.com/rss/pl/617",  # Breaking news
                "https://en.irna.ir/rss.aspx",  # IRNA English
                "https://www.tasnimnews.com/en/rss",
                "https://financialtribune.com/rss",
                "https://www.radiofarda.com/api/z-riqtvqmt"  # RFE/RL Iran
            ],
            "Iraq": [
                "https://www.iraqinews.com/feed",
                "https://shafaq.com/en/rss",
                "https://iraq-businessnews.com/feed",
                "https://www.basnews.com/en/rss",
                "https://www.newarab.com/rss",  # The New Arab
                "https://www.rudaw.net/english/rss"
            ],
            "Syria": [
                "https://www.sana.sy/en/?feed=rss2",
                "https://english.enabbaladi.net/feed",
                "https://syrianobserver.com/feed",
                "https://npasyria.com/en/feed",  # North Press Agency
                "https://www.newarab.com/rss",
                "https://www.middleeasteye.net/rss"
            ],
            "Bulgaria": [
                "https://www.novinite.com/rss",
                "https://sofiaglobe.com/feed",
                "https://www.bta.bg/en/rss",
                "https://bnr.bg/en/rss",  # Radio Bulgaria
                "https://www.dnevnik.bg/rss"
            ],
            "Turkey": [
                "https://www.hurriyetdailynews.com/rss",
                "https://www.dailysabah.com/rss",
                "https://www.duvarenglish.com/rss",
                "https://www.aa.com.tr/en/rss/default?cat=turkey",  # Anadolu Agency
                "https://ahvalnews.com/rss.xml"
            ]
        }
        
        self.now = datetime.now()
        self.start_of_yesterday = datetime(self.now.year, self.now.month, self.now.day) - timedelta(days=1)

    def fetch_feed_entries(self, country, url):
        entries = []
        import requests
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry
        
        # Robust session with retries
        session = requests.Session()
        retries = Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
        session.mount('https://', HTTPAdapter(max_retries=retries))
        session.mount('http://', HTTPAdapter(max_retries=retries))
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (compatible; BNTI-Bot/1.0; +http://monarchcastle.tech)',
            'Accept': 'application/rss+xml, application/xml, text/xml, */*'
        }

        try:
            # Fetch raw content first with requests to control headers/timeout
            response = session.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            
            # Parse the content
            feed = feedparser.parse(response.content)
            
            if not hasattr(feed, 'entries'): return []
                
            for entry in feed.entries:
                if not hasattr(entry, 'link') or not entry.link: continue
                if not hasattr(entry, 'title') or not entry.title: continue

                published_date_str = entry.get('published', None)
                if published_date_str:
                    try:
                        published_date = date_parser.parse(published_date_str).replace(tzinfo=None)
                        # Relaxed date check for robustness (last 48 hours)
                        if published_date >= (self.now - timedelta(days=2)):
                            entries.append(entry)
                    except ValueError:
                        continue
                else:
                    # If no date, assume it's recent enough given we just fetched it
                    entries.append(entry)
            
            # Fallback
            if not entries and feed.entries:
                valid_entries = [e for e in feed.entries if hasattr(e, 'link') and hasattr(e, 'title')]
                entries = valid_entries[:5]
                
        except Exception as e:
            logger.error(f"Error fetching {url}: {e}")
            # Last ditch attempt with simple feedparser if requests failed (rare but possible with some weird redirects)
            try:
                feed = feedparser.parse(url)
                if hasattr(feed, 'entries') and feed.entries:
                    return feed.entries[:5]
            except:
                pass
                
        return entries

    # Keywords that indicate non-threatening news (false positive filter)
    FALSE_POSITIVE_KEYWORDS = [
        'taxi', 'car accident', 'traffic', 'collision', 'crash', 'vehicle',
        'football', 'soccer', 'basketball', 'tennis', 'sports', 'match', 'game', 'score',
        'weather', 'forecast', 'temperature', 'rain', 'sunny',
        'recipe', 'cooking', 'restaurant', 'food',
        'celebrity', 'entertainment', 'movie', 'music', 'concert',
        'tourism', 'travel', 'hotel', 'vacation',
        'stock market', 'shares', 'trading', 'dividend'
    ]

    def is_false_positive(self, title):
        """Check if a headline is likely a false positive (non-threatening news)."""
        title_lower = title.lower()
        for keyword in self.FALSE_POSITIVE_KEYWORDS:
            if keyword in title_lower:
                return True
        return False

    def analyze_news(self, titles):
        if not titles: return 0.0, []
        try:
            results = self.classifier(titles, self.candidate_labels, multi_label=False)
        except Exception as e:
            logger.error(f"Classification failed: {e}")
            return 0.0, []
        
        threat_score = 0.0
        details = []
        if isinstance(results, dict): results = [results]

        for res in results:
            top_label = res['labels'][0]
            top_score = res['scores'][0]
            title_text = res['sequence']
            
            # FALSE POSITIVE FILTER: Skip obvious non-threatening news
            if self.is_false_positive(title_text):
                weight = 0
            # HIGH-THREAT CATEGORIES need higher confidence (0.55+ instead of 0.4)
            elif top_label in ['military conflict', 'terrorist act'] and top_score < 0.55:
                weight = 0
            # MEDIUM-THREAT CATEGORIES use standard threshold (0.45)
            elif top_label in ['violent protest', 'political crisis'] and top_score < 0.45:
                weight = 0
            # LOW-THREAT and NEUTRAL use original threshold
            elif top_score < 0.4:
                weight = 0
            else:
                weight = self.category_weights.get(top_label, 0)
            
            contribution = weight * top_score
            threat_score += contribution
            
            details.append({
                "sequence": res['sequence'],
                "label": top_label,
                "score": top_score,
                "contribution": contribution
            })
        return threat_score, details

    def process_country(self, country, urls):
        logger.info(f"Processing {country}...")
        all_entries = []
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.fetch_feed_entries, country, url) for url in urls]
            for future in concurrent.futures.as_completed(futures):
                all_entries.extend(future.result())
        
        if not all_entries: return country, 0.0, []

        seen_links = set()
        unique_entries = []
        for e in all_entries:
            if hasattr(e, 'link') and e.link and e.link not in seen_links:
                unique_entries.append(e)
                seen_links.add(e.link)
        
        unique_entries = unique_entries[:15] 

        titles_map = {e.link: e.title for e in unique_entries}
        original_titles = list(titles_map.values())
        
        base_threat_score, analysis_details = self.analyze_news(original_titles)
        
        final_report_data = []
        for i, detail in enumerate(analysis_details):
            original_entry = unique_entries[i]
            entry_data = {
                "title": original_entry.title,
                "translated_title": None, # Filled later for top threats
                "link": original_entry.link,
                "date": original_entry.get('published', 'N/A'),
                "category": detail['label'],
                "confidence": detail['score'],
                "weight": detail['contribution']
            }
            final_report_data.append(entry_data)
        
        return country, base_threat_score, final_report_data

    def calculate_final_index(self, raw_score):
        if raw_score <= 0: return 1.0
        return min(math.log10(1 + raw_score) * 3.5, 10.0)

    def detect_and_enrich_metadata(self, events):
        """adds AI metadata to all events for transparency"""
        for e in events:
            # Metadata for AI transparency
            e["ai_model"] = "XLM-RoBERTa-Large-XNLI"
            e["ai_confidence_score"] = f"{e.get('confidence', 0)*100:.1f}%"
            
            # Simple language heuristic
            if e["title"].isascii():
                e["detected_lang"] = "en"
                e["is_translated"] = False
            else:
                e["detected_lang"] = "local" # approximations
                e["is_translated"] = False # will be updated if selected for translation

    def translate_top_threats(self, dashboard_data):
        """Translates only the top 15 most weighted events to English with metadata."""
        all_events = []
        for c in dashboard_data["countries"]:
            self.detect_and_enrich_metadata(dashboard_data["countries"][c]["events"])
            for e in dashboard_data["countries"][c]["events"]:
                e["country"] = c
                all_events.append(e)
        
        # Sort by weight descending (Top 15 for better coverage)
        top_list = sorted(all_events, key=lambda x: x['weight'], reverse=True)[:15]
        
        logger.info(f"Translating Top {len(top_list)} Threats...")
        for event in top_list:
            if event.get("is_translated"): continue

            try:
                if event["detected_lang"] == "en":
                    event["translated_title"] = event["title"]
                    event["is_translated"] = False
                else:
                    trans = self.translator.translate(event["title"], dest='en')
                    event["translated_title"] = trans.text
                    event["is_translated"] = True
                    event["translation_engine"] = "Google Neural MT"
                    time.sleep(0.5) 
            except Exception as e:
                logger.warning(f"Translation failed: {e}")
                event["translated_title"] = event["title"]

        return top_list

    def load_history(self):
        """Loads historical index data from CSV."""
        if os.path.exists(self.history_file):
            try:
                df = pd.read_csv(self.history_file)
                return df.to_dict('records')
            except Exception:
                return []
        return []

    def save_history(self, final_index, country_results=None, status="UNKNOWN"):
        """Appends comprehensive run results to history CSV for predictions and archival."""
        # Base record
        new_record = {
            "timestamp": datetime.now().isoformat(),
            "main_index": round(final_index, 2),
            "status": status
        }
        
        # Add per-country indices if available
        country_order = ["Armenia", "Georgia", "Greece", "Iran", "Iraq", "Syria", "Bulgaria"]
        total_signals = 0
        
        for country in country_order:
            if country_results and country in country_results:
                new_record[f"{country.lower()}_idx"] = country_results[country].get("index", 0)
                new_record[f"{country.lower()}_signals"] = len(country_results[country].get("events", []))
                total_signals += len(country_results[country].get("events", []))
            else:
                new_record[f"{country.lower()}_idx"] = 0
                new_record[f"{country.lower()}_signals"] = 0
        
        new_record["total_signals"] = total_signals
        
        df_new = pd.DataFrame([new_record])
        
        if os.path.exists(self.history_file):
            # Check if we need to add new columns (migration)
            try:
                existing_df = pd.read_csv(self.history_file, nrows=0)
                existing_cols = set(existing_df.columns)
                new_cols = set(df_new.columns)
                
                if new_cols != existing_cols:
                    # Schema changed, recreate with all columns
                    full_df = pd.read_csv(self.history_file)
                    for col in new_cols - existing_cols:
                        full_df[col] = 0 if 'idx' in col or 'signals' in col else ''
                    full_df = pd.concat([full_df, df_new], ignore_index=True)
                    full_df.to_csv(self.history_file, mode='w', header=True, index=False)
                else:
                    df_new.to_csv(self.history_file, mode='a', header=False, index=False)
            except Exception:
                df_new.to_csv(self.history_file, mode='a', header=False, index=False)
        else:
            df_new.to_csv(self.history_file, mode='w', header=True, index=False)
            
    def generate_forecast(self, history):
        """Generates a simple linear forecast for the next 6 hours."""
        if len(history) < 2:
            return [] # Not enough data
            
        # Extract last 10 points max for trend
        recent = history[-10:]
        y = [float(h['index']) for h in recent]
        x = np.arange(len(y))
        
        # Linear fit (y = mx + b)
        if len(x) > 1:
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            
            # Forecast next 3 standard steps (e.g. if steps are runs, this is next 3 runs)
            # Assuming ~2-6 hour intervals, 3 steps is approx "next hours"
            future_x = np.arange(len(y), len(y) + 3)
            future_y = p(future_x)
            
            forecast_points = []
            last_time = parser.parse(recent[-1]['timestamp']) if 'timestamp' in recent[-1] else datetime.now()
            
            for i, val in enumerate(future_y):
                # Fake time increment for viz purpose (e.g. +4 hours per run assumption)
                next_time = last_time + timedelta(hours=4 * (i+1))
                forecast_points.append({
                    "timestamp": next_time.isoformat(),
                    "index": round(max(1.0, min(10.0, float(val))), 2), # Clamp between 1-10
                    "type": "forecast"
                })
            return forecast_points
        return []

    def save_snapshot(self, country_results, turkey_index_so_far, status="SCANNING_NETWORKS"):
        if turkey_index_so_far == 0 and country_results:
             current_total = sum(d['raw_score'] for d in country_results.values())
             turkey_index_so_far = self.calculate_final_index(current_total)

        # Get history for graph
        history = self.load_history()
        
        # Add current running point to history view (even if not saved to CSV yet)
        display_history = history.copy()
        if status != "INITIALIZING_MODELS": 
             display_history.append({
                 "timestamp": datetime.now().isoformat(),
                 "index": round(turkey_index_so_far, 2),
                 "type": "live"
             })

        # Generate Forecast if we have enough history
        forecast = []

        # Calculate next update time (next hour)
        next_hour = (datetime.now().replace(minute=0, second=0, microsecond=0) + timedelta(hours=1))

        dashboard_data = {
            "meta": {
                "generated_at": datetime.now().isoformat(),
                "main_index": round(turkey_index_so_far, 2),
                "status": status,
                "active_scan": True,
                "next_update": next_hour.isoformat(),
                "version": "2.0.0"
            },
            "countries": country_results,
            "history": display_history,
            "forecast": forecast,
            "methodology": {
                "name": "Modified Goldstein Scale",
                "description": "AI-powered threat classification using XLM-RoBERTa multilingual model with category-weighted scoring",
                "weights": self.category_weights,
                "formula": "ThreatIndex = log10(1 + Σ(category_weight × confidence)) × 3.5",
                "scale": {
                    "min": 1.0,
                    "max": 10.0,
                    "thresholds": {
                        "STABLE": [1.0, 4.0],
                        "ELEVATED": [4.0, 7.0],
                        "CRITICAL": [7.0, 10.0]
                    }
                }
            }
        }
        
        # If complete, do translation pass & SAVE history
        if status.startswith("COMPLETE") or status == "CRITICAL" or status == "ELEVATED" or status == "STABLE":
            self.translate_top_threats(dashboard_data)
            # Only save to CSV if this is a 'real' final result logic (simple check: if we have > 3 countries)
            if len(country_results) > 3:
                self.save_history(turkey_index_so_far, country_results, status)
                # Re-generate forecast now that we saved it
                real_history = self.load_history()
                
                # Enhanced Forecast with Confidence Scores
                if len(real_history) >= 2:
                    # Use main_index column
                    y = [float(h.get('main_index', h.get('index', 5.0))) for h in real_history[-24:]]
                    x = np.arange(len(y))
                    
                    # Linear regression
                    z = np.polyfit(x, y, 1)
                    p = np.poly1d(z)
                    
                    # Calculate R² for confidence baseline
                    y_pred = p(x)
                    ss_res = np.sum((np.array(y) - y_pred) ** 2)
                    ss_tot = np.sum((np.array(y) - np.mean(y)) ** 2)
                    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.5
                    
                    last_ts = datetime.fromisoformat(real_history[-1]['timestamp'])
                    forecast = []
                    
                    # Forecast next 6 hours (hourly)
                    for i in range(1, 7):
                        val = p(len(y) + i - 1)
                        ts = last_ts + timedelta(hours=i)
                        # Confidence decreases with distance
                        confidence = max(0.3, min(0.95, r_squared * (1 - i * 0.08)))
                        forecast.append({
                            "timestamp": ts.isoformat(),
                            "index": round(max(1.0, min(10.0, val)), 2),
                            "confidence": round(confidence, 2),
                            "type": "forecast"
                        })
                    dashboard_data["forecast"] = forecast

        js_path = os.path.join(self.output_path, "bnti_data.js")
        json_path = os.path.join(self.output_path, "bnti_data.json")
        try:
            with open(js_path, "w", encoding="utf-8") as f:
                json_str = json.dumps(dashboard_data, indent=2, ensure_ascii=False)
                f.write(f"window.BNTI_DATA = {json_str};")
            
            # Save pure JSON for AJAX fetching
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(dashboard_data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logger.error(f"Error saving snapshot: {e}")

    def run(self):
        os.makedirs(self.output_path, exist_ok=True)
        country_results = {}
        total_raw_threat = 0.0
        
        self.save_snapshot({}, 0.0, "INITIALIZING_MODELS")
        
        for country, urls in self.rss_urls.items():
            if not urls: continue
            
            self.save_snapshot(country_results, total_raw_threat, f"SCANNING: {country.upper()}")
            
            _, raw_score, data = self.process_country(country, urls)
            if country == "Greece": raw_score *= 0.6
            
            final_index = self.calculate_final_index(raw_score)
            total_raw_threat += raw_score
            sorted_events = sorted(data, key=lambda x: x['weight'], reverse=True)
            
            country_results[country] = {
                "index": round(final_index, 2),
                "raw_score": round(raw_score, 2),
                "events": sorted_events
            }
            
            turkey_idx = self.calculate_final_index(total_raw_threat)
            self.save_snapshot(country_results, turkey_idx, f"ANALYZING: {country.upper()}")

        final_turkey_index = self.calculate_final_index(total_raw_threat)
        final_status = "CRITICAL" if final_turkey_index > 7.0 else "ELEVATED" if final_turkey_index > 4.0 else "STABLE"
        
        logging.info("Starting Final Translation Pass...")
        self.save_snapshot(country_results, final_turkey_index, final_status)
        logger.info(f"Analysis Complete. Turkey Index: {final_turkey_index:.2f}")

if __name__ == "__main__":
    analyzer = BNTIAnalyzer()
    analyzer.run()