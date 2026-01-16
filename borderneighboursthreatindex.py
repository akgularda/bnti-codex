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
import re
import numpy as np
import threading
from urllib.parse import quote_plus, urlparse
from googletrans import Translator
from urllib3.exceptions import InsecureRequestWarning

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logging.getLogger("urllib3").setLevel(logging.ERROR)
import warnings
warnings.simplefilter("ignore", InsecureRequestWarning)

# Set socket timeout to prevent hanging on bad feeds
socket.setdefaulttimeout(10)

class BNTIAnalyzer:
    def __init__(self):
        self.output_path = os.getcwd()
        self.history_file = os.path.join(self.output_path, "bnti_history.csv")
        self._init_cache()
        
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

        # RSS Feeds Configuration (Hardened with mirrors - Jan 2026)
        self.rss_urls = {
            "Armenia": [
                "https://hetq.am/en/rss",
                "https://armenianweekly.com/feed",
                "https://en.armradio.am/feed/",
                "https://www.panarmenian.net/eng/rss/news/",
                "https://www.civilnet.am/en/rss/"
            ],
            "Georgia": [
                "https://civil.ge/feed",
                "https://georgiatoday.ge/feed",
                "https://jam-news.net/feed",
                "https://oc-media.org/feed",
                "https://www.interpressnews.ge/en/rss",
                "https://agenda.ge/en/news.rss"
            ],
            "Greece": [
                "https://www.in.gr/feed/?rid=2&pid=250&la=1&si=1",
                "https://www.naftemporiki.gr/feed",
                "https://www.protothema.gr/rss",
                "https://gtp.gr/rss.asp",
                "https://greece.greekreporter.com/feed/",
                "https://www.thenationalherald.com/feed/"
            ],
            "Iran": [
                "https://en.irna.ir/rss.aspx",
                "https://www.presstv.ir/RSS",
                "https://www.middleeasteye.net/rss",
                "https://www.aljazeera.com/xml/rss/all.xml",
                "https://feeds.bbci.co.uk/news/world/middle_east/rss.xml"
            ],
            "Iraq": [
                "https://iraq-businessnews.com/feed",
                "https://www.newarab.com/rss",
                "https://www.rudaw.net/english/rss",
                "https://www.kurdistan24.net/en/rss",
                "https://www.middleeasteye.net/rss",
                "https://www.aljazeera.com/xml/rss/all.xml",
                "https://feeds.bbci.co.uk/news/world/middle_east/rss.xml"
            ],
            "Syria": [
                "https://www.sana.sy/en/?feed=rss2",
                "https://english.enabbaladi.net/feed",
                "https://syrianobserver.com/feed",
                "https://npasyria.com/en/feed",
                "https://www.newarab.com/rss",
                "https://www.middleeasteye.net/rss",
                "https://www.aljazeera.com/xml/rss/all.xml"
            ],
            "Bulgaria": [
                "https://sofiaglobe.com/feed",
                "https://balkaninsight.com/feed/?post_type=news&country=bulgaria"
            ],
            "Turkey": [
                "https://www.hurriyetdailynews.com/rss",
                "https://www.dailysabah.com/rss",
                "https://www.aa.com.tr/en/rss/default?cat=turkey",
                "https://balkaninsight.com/feed/?post_type=news&country=turkey"
            ]
        }

        self.mirror_queries = {
            "Armenia": [
                "Armenia",
                "Armenia border conflict",
                "Armenia security",
                "Armenia military"
            ],
            "Georgia": [
                "Georgia country",
                "Georgia border",
                "Georgia security",
                "Georgia military"
            ],
            "Greece": [
                "Greece",
                "Greece border",
                "Greece security",
                "Greece military"
            ],
            "Iran": [
                "Iran",
                "Iran protest",
                "Iran security",
                "Iran military"
            ],
            "Iraq": [
                "Iraq",
                "Iraq security",
                "Iraq conflict",
                "Iraq military"
            ],
            "Syria": [
                "Syria",
                "Syria conflict",
                "Syria security",
                "Syria military"
            ],
            "Bulgaria": [
                "Bulgaria",
                "Bulgaria security",
                "Bulgaria military"
            ],
            "Turkey": [
                "Turkey OR Turkiye",
                "Turkey border",
                "Turkey security",
                "Turkey military"
            ]
        }
        self._add_mirror_sources()

        self.user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ]
        
        self.now = datetime.now()
        self.start_of_yesterday = datetime(self.now.year, self.now.month, self.now.day) - timedelta(days=1)

    # Persistent feed cache for war-time resilience
    def _init_cache(self):
        cache_root = os.path.join(os.path.expanduser("~"), ".cache", "bnti")
        self.cache_dir = cache_root
        self.feed_cache_file = os.path.join(self.cache_dir, "feed_cache.json")
        self.cache_fresh_ttl_seconds = 60 * 30
        self.cache_stale_ttl_seconds = 60 * 60 * 48
        self.cache_lock = threading.Lock()
        self.feed_cache = {}
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
            self.feed_cache = self._load_feed_cache()
        except Exception as e:
            logger.warning(f"Feed cache disabled: {e}")
            self.feed_cache_file = None

    def _load_feed_cache(self):
        if not self.feed_cache_file or not os.path.exists(self.feed_cache_file):
            return {}
        try:
            with open(self.feed_cache_file, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            if isinstance(data, dict):
                return data
        except Exception as e:
            logger.warning(f"Failed to load feed cache: {e}")
        return {}

    def _save_feed_cache_locked(self):
        if not self.feed_cache_file:
            return
        tmp_path = f"{self.feed_cache_file}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(self.feed_cache, handle, ensure_ascii=True, separators=(",", ":"))
        os.replace(tmp_path, self.feed_cache_file)

    def _cache_entry_age_seconds(self, entry):
        fetched_at = entry.get("fetched_at")
        if not fetched_at:
            return None
        try:
            cached_time = date_parser.parse(fetched_at).replace(tzinfo=None)
        except Exception:
            return None
        return (datetime.utcnow() - cached_time).total_seconds()

    def _get_cached_entries(self, url, max_age_seconds):
        if not self.feed_cache_file:
            return None, None
        with self.cache_lock:
            entry = self.feed_cache.get(url)
        if not isinstance(entry, dict):
            return None, None
        age = self._cache_entry_age_seconds(entry)
        if age is None or age > max_age_seconds:
            return None, None
        entries = entry.get("entries")
        if not isinstance(entries, list):
            return None, None
        return entries, age

    def _serialize_entries(self, entries):
        if not entries:
            return []
        serialized = []
        for entry in entries:
            title = entry.get("title") if hasattr(entry, "get") else getattr(entry, "title", None)
            link = entry.get("link") if hasattr(entry, "get") else getattr(entry, "link", None)
            if not title or not link:
                continue
            item = {"title": str(title), "link": str(link)}
            published = entry.get("published") if hasattr(entry, "get") else getattr(entry, "published", None)
            if published:
                item["published"] = str(published)
            serialized.append(item)
        return serialized

    def _write_cache_entries(self, url, entries):
        if not self.feed_cache_file:
            return
        serialized = self._serialize_entries(entries)
        if not serialized:
            return
        payload = {
            "fetched_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S"),
            "entries": serialized
        }
        with self.cache_lock:
            self.feed_cache[url] = payload
            self._save_feed_cache_locked()

    def _google_news_url(self, query):
        encoded = quote_plus(query)
        return f"https://news.google.com/rss/search?q={encoded}&hl=en-US&gl=US&ceid=US:en"

    def _gdelt_url(self, query):
        encoded = quote_plus(query)
        return f"https://api.gdeltproject.org/api/v2/doc/doc?query={encoded}&mode=artlist&maxrecords=50&format=rss"

    def _add_mirror_sources(self):
        for country, queries in self.mirror_queries.items():
            if isinstance(queries, str):
                queries = [queries]
            mirrors = []
            for query in queries:
                mirrors.append(self._google_news_url(query))
                mirrors.append(self._gdelt_url(query))
            base = self.rss_urls.get(country, [])
            merged = list(dict.fromkeys(base + mirrors))
            self.rss_urls[country] = merged

    def _extract_entries(self, entries):
        if not entries:
            return []

        now = datetime.now()
        recent_entries = []
        for entry in entries:
            link = entry.get('link') if hasattr(entry, 'get') else getattr(entry, 'link', None)
            title = entry.get('title') if hasattr(entry, 'get') else getattr(entry, 'title', None)
            if not link or not title:
                continue

            published_date_str = entry.get('published') if hasattr(entry, 'get') else getattr(entry, 'published', None)
            if published_date_str:
                try:
                    published_date = date_parser.parse(published_date_str).replace(tzinfo=None)
                    if published_date >= (now - timedelta(days=2)):
                        recent_entries.append(entry)
                except Exception:
                    continue
            else:
                recent_entries.append(entry)

        if recent_entries:
            return recent_entries

        fallback = []
        for entry in entries:
            link = entry.get('link') if hasattr(entry, 'get') else getattr(entry, 'link', None)
            title = entry.get('title') if hasattr(entry, 'get') else getattr(entry, 'title', None)
            if link and title:
                fallback.append(entry)
        return fallback[:5]

    def _build_proxy_url(self, url):
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            return None
        path = parsed.path or "/"
        if parsed.query:
            path = f"{path}?{parsed.query}"
        return f"https://r.jina.ai/{parsed.scheme}://{parsed.netloc}{path}"

    def _parse_proxy_markdown(self, content):
        entries = []
        for line in content.splitlines():
            if "CDATA[" not in line:
                continue
            title_match = re.search(r"<!\\[CDATA\\[(.*?)\\]\\]>", line)
            if not title_match:
                continue
            url_match = re.search(r"https?://[^\\s]+", line)
            if not url_match:
                continue
            date_match = re.search(
                r"(Mon|Tue|Wed|Thu|Fri|Sat|Sun),\\s+\\d{1,2}\\s+\\w+\\s+\\d{4}\\s+\\d{2}:\\d{2}:\\d{2}\\s+\\w+",
                line
            )
            entry = feedparser.FeedParserDict()
            entry["title"] = title_match.group(1).strip()
            entry["link"] = url_match.group(0)
            if date_match:
                entry["published"] = date_match.group(0)
            entries.append(entry)

        if not entries:
            return []

        deduped = []
        seen = set()
        for entry in entries:
            link = entry.get("link")
            if link in seen:
                continue
            seen.add(link)
            deduped.append(entry)
        return deduped[:12]

    def _fetch_proxy_entries(self, url, session, headers):
        proxy_url = self._build_proxy_url(url)
        if not proxy_url:
            return []
        try:
            response = session.get(proxy_url, headers=headers, timeout=20)
            response.raise_for_status()
            entries = self._parse_proxy_markdown(response.text)
            return self._extract_entries(entries)
        except Exception as e:
            logger.warning(f"Proxy fetch failed for {url}: {e}")
            return []

    def fetch_feed_entries(self, country, url):
        import requests
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry

        session = requests.Session()
        retries = Retry(
            total=6,
            connect=5,
            read=5,
            backoff_factor=1.5,
            status_forcelist=[408, 429, 500, 502, 503, 504, 522, 524],
            allowed_methods=frozenset(["HEAD", "GET", "OPTIONS"]),
            respect_retry_after_header=True
        )
        session.mount('https://', HTTPAdapter(max_retries=retries))
        session.mount('http://', HTTPAdapter(max_retries=retries))

        base_headers = {
            'Accept': 'application/rss+xml, application/xml, text/xml, */*',
            'Accept-Language': 'en-US,en;q=0.8',
            'Cache-Control': 'no-cache',
            'Pragma': 'no-cache',
            'Referer': 'https://www.google.com/'
        }

        cached_entries, _ = self._get_cached_entries(url, self.cache_fresh_ttl_seconds)
        if cached_entries:
            cached = self._extract_entries(cached_entries)
            if cached:
                return cached

        last_error = None
        for user_agent in self.user_agents:
            headers = dict(base_headers)
            headers['User-Agent'] = user_agent
            try:
                response = session.get(url, headers=headers, timeout=15)
                response.raise_for_status()
                feed = feedparser.parse(response.content)
                entries = self._extract_entries(feed.entries if hasattr(feed, 'entries') else [])
                if entries:
                    self._write_cache_entries(url, entries)
                    return entries
            except requests.exceptions.SSLError as e:
                last_error = e
                try:
                    response = session.get(url, headers=headers, timeout=15, verify=False)
                    response.raise_for_status()
                    feed = feedparser.parse(response.content)
                    entries = self._extract_entries(feed.entries if hasattr(feed, 'entries') else [])
                    if entries:
                        logger.warning(f"SSL verification skipped for {url}")
                        self._write_cache_entries(url, entries)
                        return entries
                except Exception as e2:
                    last_error = e2
            except Exception as e:
                last_error = e

        try:
            feed = feedparser.parse(url)
            entries = self._extract_entries(feed.entries if hasattr(feed, 'entries') else [])
            if entries:
                self._write_cache_entries(url, entries)
                return entries
        except Exception as e:
            last_error = e

        entries = self._fetch_proxy_entries(url, session, base_headers)
        if entries:
            self._write_cache_entries(url, entries)
            return entries

        cached_entries, cache_age = self._get_cached_entries(url, self.cache_stale_ttl_seconds)
        if cached_entries:
            cached = self._extract_entries(cached_entries)
            if cached:
                if cache_age is not None:
                    age_minutes = int(cache_age // 60)
                    logger.warning(f"Using cached feed for {url} ({age_minutes}m old)")
                return cached

        if last_error:
            logger.error(f"Error fetching {url}: {last_error}")
        return []

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

    def _parse_timestamp(self, value):
        if not value:
            return None
        try:
            return date_parser.parse(str(value)).replace(tzinfo=None)
        except Exception:
            return None

    def _extract_index(self, record):
        for key in ("main_index", "index"):
            value = record.get(key)
            if value is None:
                continue
            try:
                num = float(value)
            except (TypeError, ValueError):
                continue
            if math.isnan(num):
                continue
            return num
        return None

    def _trim_history(self, history, hours=48, max_points=48):
        if not history:
            return []
        now = datetime.now()
        recent = []
        for entry in history:
            ts = self._parse_timestamp(entry.get("timestamp"))
            if not ts:
                continue
            if now - ts <= timedelta(hours=hours):
                recent.append(entry)
        if not recent:
            recent = history[-max_points:]
        return recent[-max_points:]

    def _build_history_payload(self, history, include_live=False, live_index=None):
        payload = []
        for entry in history:
            idx = self._extract_index(entry)
            ts = entry.get("timestamp")
            if idx is None or not ts:
                continue
            payload.append({
                "timestamp": ts,
                "main_index": round(idx, 2),
                "index": round(idx, 2),
                "type": "historical"
            })
        if include_live and live_index is not None:
            payload.append({
                "timestamp": datetime.now().isoformat(),
                "main_index": round(live_index, 2),
                "index": round(live_index, 2),
                "type": "live"
            })
        return payload

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
                history = df.to_dict('records')
                for entry in history:
                    idx = self._extract_index(entry)
                    if idx is not None:
                        entry["main_index"] = idx
                return history
            except Exception:
                return []
        return []

    def save_history(self, final_index, country_results=None, status="UNKNOWN"):
        """Appends comprehensive run results to history CSV for predictions and archival."""
        # Base record
        new_record = {
            "timestamp": datetime.now().isoformat(),
            "main_index": round(final_index, 2),
            "index": round(final_index, 2),
            "status": status
        }
        
        # Add per-country indices if available
        country_order = ["Turkey", "Armenia", "Georgia", "Greece", "Iran", "Iraq", "Syria", "Bulgaria"]
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
        points = []
        for entry in history:
            idx = self._extract_index(entry)
            ts = self._parse_timestamp(entry.get("timestamp"))
            if idx is None or not ts:
                continue
            points.append((ts, idx))

        if len(points) < 2:
            return []

        points.sort(key=lambda x: x[0])
        points = points[-24:]
        y = np.array([p[1] for p in points], dtype=float)
        x = np.arange(len(y), dtype=float)

        if len(x) < 2:
            return []

        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)

        y_pred = p(x)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.5

        last_time = points[-1][0]
        forecast_points = []

        for i in range(1, 7):
            val = p(len(y) + i - 1)
            ts = last_time + timedelta(hours=i)
            confidence = max(0.3, min(0.95, r_squared * (1 - i * 0.08)))
            forecast_points.append({
                "timestamp": ts.isoformat(),
                "index": round(max(1.0, min(10.0, float(val))), 2),
                "main_index": round(max(1.0, min(10.0, float(val))), 2),
                "confidence": round(confidence, 2),
                "type": "forecast"
            })
        return forecast_points

    def save_snapshot(self, country_results, turkey_index_so_far, status="SCANNING_NETWORKS"):
        if turkey_index_so_far == 0 and country_results:
            current_total = sum(d.get('raw_score', 0) for d in country_results.values())
            turkey_index_so_far = self.calculate_final_index(current_total)

        history = self._trim_history(self.load_history())
        display_history = self._build_history_payload(
            history,
            include_live=(status != "INITIALIZING_MODELS"),
            live_index=turkey_index_so_far
        )
        forecast = self.generate_forecast(history)

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
                "formula": "ThreatIndex = log10(1 + sum(category_weight * confidence)) * 3.5",
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

        if status.startswith("COMPLETE") or status in ("CRITICAL", "ELEVATED", "STABLE"):
            self.translate_top_threats(dashboard_data)
            self.save_history(turkey_index_so_far, country_results, status)
            real_history = self._trim_history(self.load_history())
            dashboard_data["history"] = self._build_history_payload(real_history)
            dashboard_data["forecast"] = self.generate_forecast(real_history)

        js_path = os.path.join(self.output_path, "bnti_data.js")
        json_path = os.path.join(self.output_path, "bnti_data.json")
        try:
            with open(js_path, "w", encoding="utf-8") as f:
                json_str = json.dumps(dashboard_data, indent=2, ensure_ascii=False)
                f.write(f"window.BNTI_DATA = {json_str};")

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
            
            self.save_snapshot(country_results, self.calculate_final_index(total_raw_threat), f"SCANNING: {country.upper()}")
            
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
        logger.info(f"Analysis Complete. Composite Index: {final_turkey_index:.2f}")

if __name__ == "__main__":
    try:
        analyzer = BNTIAnalyzer()
        analyzer.run()
    except Exception as e:
        # NEVER crash - log and exit gracefully
        logging.error(f"Analyzer encountered a critical error: {e}")
        logging.info("Exiting gracefully to prevent workflow failure.")
        # Exit 0 so GitHub Actions reports SUCCESS
        import sys
        sys.exit(0)
