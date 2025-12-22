# #!/usr/bin/env python3
# """
# WIPO PATENTSCOPE Patent Downloader
# Topics: CO2 Methanation | Fischer-Tropsch | Hydrocracking
# Downloads patent URLs and optionally PDFs
# """

# import time
# import pandas as pd
# from pathlib import Path
# from selenium.webdriver.edge.options import Options
# from selenium import webdriver
# from selenium.webdriver.common.by import By
# from selenium.webdriver.support.ui import WebDriverWait
# from selenium.webdriver.support import expected_conditions as EC
# import requests
# import json
# from tqdm import tqdm

# # ==================== CONFIG ====================
# OUTPUT_DIR = Path("./wipo_patents")
# CSV_DIR = OUTPUT_DIR / "csv"
# PDF_DIR = OUTPUT_DIR / "pdf"
# TRACKING_FILE = OUTPUT_DIR / "download_tracker.json"

# # WIPO PATENTSCOPE search configurations
# TOPIC_SEARCHES = {
#     "co2_methanation": {
#         "ipc_codes": ["C07C1/12", "B01J23/72", "B01J23/755"],
#         "keywords": "methanation OR 'CO2 hydrogenation' OR Sabatier",
#         "max_pages": 100  # 100 pages = ~5000 patents (50 per page)
#     },
#     "fischer_tropsch": {
#         "ipc_codes": ["C07C1/04", "C10G2/00", "B01J23/75"],
#         "keywords": "'Fischer-Tropsch' OR 'FT synthesis' OR 'syngas conversion'",
#         "max_pages": 150
#     },
#     "hydrocracking": {
#         "ipc_codes": ["C10G47", "C10G65", "B01J21/12", "B01J23/88"],
#         "keywords": "hydrocracking OR hydrocracker OR 'zeolite catalyst'",
#         "max_pages": 200
#     },
# }

# WIPO_BASE_URL = "https://patentscope.wipo.int"
# WIPO_SEARCH_URL = f"{WIPO_BASE_URL}/search/en/search.jsf"

# # Browser settings
# HEADLESS = False  # Set True to run in background

# # ==================== TRACKING ====================
# def load_tracker():
#     if TRACKING_FILE.exists():
#         with open(TRACKING_FILE, 'r') as f:
#             return json.load(f)
#     return {
#         "downloaded_urls": [],
#         "stats": {topic: {"patents": 0, "last_update": None} for topic in TOPIC_SEARCHES}
#     }

# def save_tracker(tracker):
#     tracker["last_updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
#     with open(TRACKING_FILE, 'w') as f:
#         json.dump(tracker, f, indent=2)

# # ==================== SELENIUM FUNCTIONS ====================
# def init_driver():
#     """Initialize Edge WebDriver"""
#     edge_options = Options()
#     edge_options.use_chromium = True
#     if HEADLESS:
#         edge_options.add_argument("--headless")
#     edge_options.add_argument("--disable-blink-features=AutomationControlled")
#     edge_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    
#     driver = webdriver.Edge(options=edge_options)
#     driver.maximize_window()
#     return driver

# def build_advanced_query(ipc_codes, keywords):
#     """Build WIPO advanced search query"""
#     # IPC query: IPC_C:(C07C1/12 OR B01J23/72)
#     ipc_query = " OR ".join(ipc_codes)
    
#     # Combined query
#     query = f"IPC_C:({ipc_query}) AND FP:({keywords})"
#     return query

# def search_wipo_simple_query(driver, query):
#     """
#     Use simple search box with query syntax
#     Easier and more reliable than advanced search form
#     """
#     try:
#         driver.get(WIPO_SEARCH_URL)
#         time.sleep(3)
        
#         # Find the simple search input box
#         search_box = WebDriverWait(driver, 10).until(
#             EC.presence_of_element_located((By.ID, "simpleSearchSearchInput"))
#         )
        
#         # Clear and enter query
#         search_box.clear()
#         search_box.send_keys(query)
#         time.sleep(1)
        
#         # Click search button
#         search_btn = driver.find_element(By.ID, "simpleSearchSearchButton")
#         search_btn.click()
        
#         # Wait for results
#         time.sleep(5)
        
#         # Set results per page to 500 (maximum)
#         try:
#             results_per_page = WebDriverWait(driver, 10).until(
#                 EC.presence_of_element_located((By.XPATH, "//select[contains(@id, 'resultsPerPageSelect')]"))
#             )
#             # Select 500 results per page
#             driver.execute_script("arguments[0].value = '500';", results_per_page)
#             driver.execute_script("arguments[0].dispatchEvent(new Event('change'));", results_per_page)
#             time.sleep(5)
#         except:
#             print("  ⚠️  Could not set results per page, using default (50)")
        
#         return True
#     except Exception as e:
#         print(f"  ❌ Search error: {e}")
#         return False

# def get_patent_urls_from_page(driver):
#     """Extract patent URLs from current results page"""
#     urls = []
#     try:
#         # Find the results table
#         table = WebDriverWait(driver, 10).until(
#             EC.presence_of_element_located((By.XPATH, "//table[contains(@id, 'resultTable')]"))
#         )
        
#         # Find all patent links
#         link_elements = table.find_elements(By.XPATH, ".//a[@target='_self']")
        
#         for element in link_elements:
#             href = element.get_attribute("href")
#             if href and "detail.jsf" in href:
#                 urls.append(href)
        
#     except Exception as e:
#         print(f"  ⚠️  Error extracting URLs: {e}")
    
#     return urls

# def click_next_page(driver):
#     """Click next page button"""
#     try:
#         # Find and click "Next" button
#         next_btn = driver.find_element(By.XPATH, "//a[contains(@class, 'next-page') or contains(text(), 'Next')]")
#         driver.execute_script("arguments[0].click();", next_btn)
#         time.sleep(5)
#         return True
#     except:
#         return False

# def scrape_topic_patents(driver, topic, config):
#     """Scrape all patent URLs for a topic"""
#     print(f"\n{'='*70}")
#     print(f"📁 {topic.upper()}")
#     print(f"{'='*70}")
    
#     # Build query
#     query = build_advanced_query(config["ipc_codes"], config["keywords"])
#     print(f"\n🔎 Query: {query}")
    
#     # Search
#     if not search_wipo_simple_query(driver, query):
#         return []
    
#     # Get total results
#     try:
#         total_elem = driver.find_element(By.XPATH, "//span[contains(@class, 'total-results')]")
#         total_results = total_elem.text.strip()
#         print(f"📊 Total results: {total_results}")
#     except:
#         print("📊 Total results: Unknown")
    
#     all_urls = []
#     max_pages = config["max_pages"]
    
#     # Scrape pages
#     for page_num in range(1, max_pages + 1):
#         print(f"\n  📄 Page {page_num}/{max_pages}")
        
#         # Get URLs from current page
#         page_urls = get_patent_urls_from_page(driver)
        
#         if not page_urls:
#             print(f"  ⚠️  No URLs found, stopping")
#             break
        
#         print(f"  ✅ Found {len(page_urls)} patents")
#         all_urls.extend(page_urls)
        
#         # Try to go to next page
#         if page_num < max_pages:
#             if not click_next_page(driver):
#                 print(f"  ℹ️  No more pages available")
#                 break
    
#     return all_urls

# # ==================== PDF DOWNLOAD ====================
# def download_patent_pdf(patent_url, output_path):
#     """
#     Download patent PDF from WIPO
#     PDF URL format: https://patentscope.wipo.int/search/en/pdf/{doc_id}.pdf
#     """
#     try:
#         # Extract document ID from URL
#         # Example: https://patentscope.wipo.int/search/en/detail.jsf?docId=WO2021234567
#         if "docId=" in patent_url:
#             doc_id = patent_url.split("docId=")[1].split("&")[0]
#             pdf_url = f"{WIPO_BASE_URL}/search/en/pdf/{doc_id}.pdf"
            
#             response = requests.get(pdf_url, timeout=30)
#             if response.status_code == 200:
#                 with open(output_path, 'wb') as f:
#                     f.write(response.content)
#                 return True
#     except Exception as e:
#         print(f"  ⚠️  PDF download error: {e}")
#     return False

# # ==================== MAIN ====================
# def main():
#     # Setup
#     OUTPUT_DIR.mkdir(exist_ok=True)
#     CSV_DIR.mkdir(exist_ok=True)
#     PDF_DIR.mkdir(exist_ok=True)
    
#     print("=" * 70)
#     print("WIPO PATENTSCOPE Patent Downloader")
#     print("Topics: CO2 Methanation | Fischer-Tropsch | Hydrocracking")
#     print("=" * 70)
    
#     tracker = load_tracker()
    
#     # Initialize driver
#     print("\n🔧 Initializing browser...")
#     driver = init_driver()
    
#     try:
#         # Process each topic
#         for topic, config in TOPIC_SEARCHES.items():
#             # Scrape URLs
#             patent_urls = scrape_topic_patents(driver, topic, config)
            
#             if patent_urls:
#                 # Remove duplicates
#                 patent_urls = list(set(patent_urls))
                
#                 # Save URLs to CSV
#                 csv_path = CSV_DIR / f"{topic}_patent_urls.csv"
#                 df = pd.DataFrame({"url": patent_urls})
#                 df.to_csv(csv_path, index=False)
                
#                 print(f"\n✅ Saved {len(patent_urls)} URLs to: {csv_path}")
                
#                 # Update tracker
#                 tracker["stats"][topic]["patents"] = len(patent_urls)
#                 tracker["stats"][topic]["last_update"] = time.strftime("%Y-%m-%d %H:%M:%S")
#                 tracker["downloaded_urls"].extend(patent_urls)
#                 save_tracker(tracker)
                
#                 # Optional: Download PDFs (comment out if you only want URLs)
#                 # print(f"\n📥 Downloading PDFs for {topic}...")
#                 # topic_pdf_dir = PDF_DIR / topic
#                 # topic_pdf_dir.mkdir(exist_ok=True)
#                 # 
#                 # for idx, url in enumerate(tqdm(patent_urls[:100])):  # Limit to first 100
#                 #     pdf_path = topic_pdf_dir / f"patent_{idx+1}.pdf"
#                 #     download_patent_pdf(url, pdf_path)
#                 #     time.sleep(1)  # Rate limiting
    
#     finally:
#         driver.quit()
#         print("\n✅ Browser closed")
    
#     # Final stats
#     print("\n" + "=" * 70)
#     print("📊 FINAL STATISTICS")
#     print("=" * 70)
#     for topic, stats in tracker["stats"].items():
#         print(f"{topic.upper()}: {stats['patents']} patent URLs")
#     print(f"\nTotal URLs: {len(set(tracker['downloaded_urls']))}")
#     print(f"CSV files saved to: {CSV_DIR}")
#     print("=" * 70)

# if __name__ == "__main__":
#     # Install: pip install selenium pandas tqdm requests
#     # Download Edge WebDriver: https://developer.microsoft.com/en-us/microsoft-edge/tools/webdriver/
#     main()
















#!/usr/bin/env python3
"""
WIPO PATENTSCOPE Patent Downloader - XML Version (Robust)
Topics: CO2 Methanation | Fischer-Tropsch | Hydrocracking
Uses Advanced Search form instead of simple search for better reliability
"""

import time
import pandas as pd
from pathlib import Path
from selenium.webdriver.edge.options import Options
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
import requests
import json
from tqdm import tqdm
import xml.etree.ElementTree as ET

# ==================== CONFIG ====================
OUTPUT_DIR = Path("./wipo_patents")
CSV_DIR = OUTPUT_DIR / "csv"
XML_DIR = OUTPUT_DIR / "xml"
TRACKING_FILE = OUTPUT_DIR / "download_tracker.json"

# WIPO PATENTSCOPE search configurations
TOPIC_SEARCHES = {
    "co2_methanation": {
        "ipc_codes": ["C07C1/12", "B01J23/72", "B01J23/755"],
        "keywords": ["methanation", "CO2 hydrogenation", "Sabatier"],
        "max_pages": 50  # Reduced for testing
    },
    "fischer_tropsch": {
        "ipc_codes": ["C07C1/04", "C10G2/00", "B01J23/75"],
        "keywords": ["Fischer-Tropsch", "FT synthesis", "syngas conversion"],
        "max_pages": 50
    },
    "hydrocracking": {
        "ipc_codes": ["C10G47", "C10G65", "B01J21/12", "B01J23/88"],
        "keywords": ["hydrocracking", "hydrocracker", "zeolite catalyst"],
        "max_pages": 50
    },
}

WIPO_BASE_URL = "https://patentscope.wipo.int"
WIPO_ADVANCED_SEARCH = f"{WIPO_BASE_URL}/search/en/structuredSearch.jsf"

# Browser settings
HEADLESS = False
DOWNLOAD_XML = True

# ==================== TRACKING ====================
def load_tracker():
    if TRACKING_FILE.exists():
        with open(TRACKING_FILE, 'r') as f:
            return json.load(f)
    return {
        "downloaded_urls": [],
        "downloaded_xmls": [],
        "failed_downloads": [],
        "stats": {
            topic: {"patents": 0, "xmls_downloaded": 0, "last_update": None} 
            for topic in TOPIC_SEARCHES
        }
    }

def save_tracker(tracker):
    tracker["last_updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(TRACKING_FILE, 'w') as f:
        json.dump(tracker, f, indent=2)

# ==================== SELENIUM FUNCTIONS ====================
def init_driver():
    """Initialize Edge WebDriver with better error handling"""
    edge_options = Options()
    edge_options.use_chromium = True
    
    if HEADLESS:
        edge_options.add_argument("--headless")
    
    edge_options.add_argument("--disable-blink-features=AutomationControlled")
    edge_options.add_argument("--no-sandbox")
    edge_options.add_argument("--disable-dev-shm-usage")
    edge_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    edge_options.add_experimental_option('useAutomationExtension', False)
    
    driver = webdriver.Edge(options=edge_options)
    driver.maximize_window()
    return driver

def search_wipo_advanced(driver, ipc_codes, keywords):
    """
    Use WIPO's structured/advanced search form
    More reliable than simple search
    """
    try:
        print(f"  🌐 Loading WIPO advanced search page...")
        driver.get(WIPO_ADVANCED_SEARCH)
        time.sleep(5)
        
        # Wait for page to fully load
        WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.ID, "structuredSearchForm"))
        )
        
        print(f"  ✅ Page loaded")
        
        # Strategy: Use the first search row for IPC, add second row for keywords
        
        # Row 1: Set field to IPC
        print(f"  🔧 Setting up IPC search...")
        try:
            # Find the field dropdown (first condition)
            field_dropdown = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((
                    By.XPATH, 
                    "//select[contains(@id, 'conditionField')]"
                ))
            )
            
            # Select "IPC" option (usually value 32 or look for "IPC" text)
            driver.execute_script("""
                var select = arguments[0];
                for(var i = 0; i < select.options.length; i++) {
                    if(select.options[i].text.includes('IPC')) {
                        select.selectedIndex = i;
                        select.dispatchEvent(new Event('change'));
                        break;
                    }
                }
            """, field_dropdown)
            time.sleep(2)
            
            # Enter IPC codes in the value field
            value_input = driver.find_element(
                By.XPATH, 
                "//input[contains(@id, 'conditionValue:input')]"
            )
            
            # Build IPC query: C07C1/12 OR B01J23/72
            ipc_query = " OR ".join(ipc_codes)
            value_input.clear()
            value_input.send_keys(ipc_query)
            time.sleep(1)
            
            print(f"  ✅ IPC codes entered: {ipc_query}")
            
        except Exception as e:
            print(f"  ⚠️  Error setting IPC: {e}")
            return False
        
        # Row 2: Add keyword search
        print(f"  🔧 Adding keyword search...")
        try:
            # Click "Add condition" button to add second row
            add_button = driver.find_element(
                By.XPATH,
                "//button[contains(@id, 'addCondition') or contains(text(), 'Add')]"
            )
            driver.execute_script("arguments[0].click();", add_button)
            time.sleep(2)
            
            # Find second row's field dropdown
            field_dropdowns = driver.find_elements(
                By.XPATH,
                "//select[contains(@id, 'conditionField')]"
            )
            
            if len(field_dropdowns) > 1:
                # Select "Front Page" or "Title and Abstract"
                driver.execute_script("""
                    var select = arguments[0];
                    for(var i = 0; i < select.options.length; i++) {
                        if(select.options[i].text.includes('Front Page') || 
                           select.options[i].text.includes('Title')) {
                            select.selectedIndex = i;
                            select.dispatchEvent(new Event('change'));
                            break;
                        }
                    }
                """, field_dropdowns[1])
                time.sleep(2)
                
                # Enter keywords
                value_inputs = driver.find_elements(
                    By.XPATH,
                    "//input[contains(@id, 'conditionValue:input')]"
                )
                
                if len(value_inputs) > 1:
                    keyword_query = " OR ".join([f'"{kw}"' for kw in keywords])
                    value_inputs[1].clear()
                    value_inputs[1].send_keys(keyword_query)
                    time.sleep(1)
                    print(f"  ✅ Keywords entered: {keyword_query}")
            
        except Exception as e:
            print(f"  ⚠️  Could not add keywords (continuing with IPC only): {e}")
        
        # Click Search button
        print(f"  🔍 Executing search...")
        search_button = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((
                By.XPATH,
                "//button[contains(@id, 'commandStructuredSearch') or contains(text(), 'Search')]"
            ))
        )
        driver.execute_script("arguments[0].click();", search_button)
        time.sleep(8)
        
        print(f"  ✅ Search executed")
        return True
        
    except Exception as e:
        print(f"  ❌ Advanced search error: {e}")
        import traceback
        traceback.print_exc()
        return False

def get_patent_urls_from_page(driver):
    """Extract patent URLs from current results page"""
    urls = []
    try:
        # Wait for results table
        table = WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.XPATH, "//table[contains(@id, 'resultTable')]"))
        )
        
        # Find all patent detail links
        link_elements = table.find_elements(By.XPATH, ".//a[@target='_self']")
        
        for element in link_elements:
            try:
                href = element.get_attribute("href")
                if href and "detail.jsf" in href and "docId=" in href:
                    urls.append(href)
            except:
                continue
        
    except Exception as e:
        print(f"  ⚠️  Error extracting URLs: {e}")
    
    return list(set(urls))  # Remove duplicates

def set_results_per_page(driver, num_results=500):
    """Set number of results per page"""
    try:
        # Find results per page dropdown
        results_dropdown = driver.find_element(
            By.XPATH,
            "//select[contains(@id, 'resultsPerPage')]"
        )
        
        # Set to maximum (500)
        driver.execute_script(f"arguments[0].value = '{num_results}';", results_dropdown)
        driver.execute_script("arguments[0].dispatchEvent(new Event('change'));", results_dropdown)
        time.sleep(5)
        print(f"  ✅ Set results per page to {num_results}")
        return True
    except Exception as e:
        print(f"  ⚠️  Could not set results per page: {e}")
        return False

def click_next_page(driver):
    """Click next page button"""
    try:
        # Try multiple selectors for next button
        next_selectors = [
            "//a[contains(@class, 'ui-paginator-next')]",
            "//a[contains(@title, 'Next')]",
            "//span[contains(@class, 'ui-icon-seek-next')]/..",
        ]
        
        for selector in next_selectors:
            try:
                next_btn = driver.find_element(By.XPATH, selector)
                if next_btn.is_enabled():
                    driver.execute_script("arguments[0].click();", next_btn)
                    time.sleep(6)
                    return True
            except:
                continue
        
        return False
    except:
        return False

def scrape_topic_patents(driver, topic, config):
    """Scrape all patent URLs for a topic"""
    print(f"\n{'='*70}")
    print(f"📁 {topic.upper()}")
    print(f"{'='*70}")
    
    # Search
    if not search_wipo_advanced(driver, config["ipc_codes"], config["keywords"]):
        return []
    
    # Get total results
    try:
        total_elem = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.XPATH, "//span[contains(@class, 'recordCount')]"))
        )
        total_results = total_elem.text.strip()
        print(f"\n📊 Total results: {total_results}")
    except:
        print(f"\n📊 Total results: Unknown")
    
    # Set max results per page
    set_results_per_page(driver, 500)
    
    all_urls = []
    max_pages = config["max_pages"]
    
    for page_num in range(1, max_pages + 1):
        print(f"\n  📄 Page {page_num}/{max_pages}")
        
        page_urls = get_patent_urls_from_page(driver)
        
        if not page_urls:
            print(f"  ⚠️  No URLs found on this page")
            if page_num == 1:
                print(f"  💡 Check if search returned any results")
            break
        
        print(f"  ✅ Found {len(page_urls)} patents")
        all_urls.extend(page_urls)
        
        # Try next page
        if page_num < max_pages:
            if not click_next_page(driver):
                print(f"  ℹ️  No more pages available")
                break
        
        time.sleep(2)
    
    return list(set(all_urls))  # Remove duplicates

# ==================== XML DOWNLOAD (same as before) ====================
def extract_doc_id(patent_url):
    """Extract document ID from WIPO URL"""
    try:
        if "docId=" in patent_url:
            doc_id = patent_url.split("docId=")[1].split("&")[0]
            return doc_id
    except:
        pass
    return None

def download_patent_xml(doc_id, output_path):
    """Download patent XML from WIPO"""
    try:
        xml_url = f"{WIPO_BASE_URL}/search/rest/v2/documents/{doc_id}"
        
        headers = {
            'Accept': 'application/xml',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(xml_url, headers=headers, timeout=30)
        
        if response.status_code == 200:
            try:
                ET.fromstring(response.content)
                with open(output_path, 'wb') as f:
                    f.write(response.content)
                return True
            except ET.ParseError:
                pass
        
        # Try alternative endpoint
        alt_url = f"{WIPO_BASE_URL}/search/en/download.jsf?docId={doc_id}&format=xml"
        response = requests.get(alt_url, headers=headers, timeout=30)
        
        if response.status_code == 200:
            with open(output_path, 'wb') as f:
                f.write(response.content)
            return True
                
    except Exception as e:
        pass
    
    return False

def batch_download_xmls(patent_urls, topic, tracker):
    """Download XMLs for a batch of patent URLs"""
    topic_xml_dir = XML_DIR / topic
    topic_xml_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📥 Downloading XMLs for {topic}...")
    successful = 0
    failed = 0
    
    for idx, url in enumerate(tqdm(patent_urls, desc=f"  {topic}")):
        doc_id = extract_doc_id(url)
        
        if not doc_id:
            failed += 1
            continue
        
        if doc_id in tracker["downloaded_xmls"]:
            successful += 1
            continue
        
        xml_path = topic_xml_dir / f"{doc_id}.xml"
        
        if download_patent_xml(doc_id, xml_path):
            tracker["downloaded_xmls"].append(doc_id)
            successful += 1
        else:
            tracker["failed_downloads"].append(doc_id)
            failed += 1
        
        time.sleep(0.5)
        
        if (idx + 1) % 50 == 0:
            tracker["stats"][topic]["xmls_downloaded"] = successful
            save_tracker(tracker)
    
    print(f"  ✅ Downloaded: {successful} | ❌ Failed: {failed}")
    return successful, failed

# ==================== MAIN ====================
def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    CSV_DIR.mkdir(exist_ok=True)
    XML_DIR.mkdir(exist_ok=True)
    
    print("=" * 70)
    print("WIPO PATENTSCOPE Patent Downloader - XML Version")
    print("Topics: CO2 Methanation | Fischer-Tropsch | Hydrocracking")
    print("=" * 70)
    
    tracker = load_tracker()
    
    print("\n🔧 Initializing browser...")
    driver = init_driver()
    
    try:
        for topic, config in TOPIC_SEARCHES.items():
            patent_urls = scrape_topic_patents(driver, topic, config)
            
            if patent_urls:
                patent_urls = list(set(patent_urls))
                
                csv_path = CSV_DIR / f"{topic}_patent_urls.csv"
                df = pd.DataFrame({"url": patent_urls})
                df.to_csv(csv_path, index=False)
                
                print(f"\n✅ Saved {len(patent_urls)} URLs to: {csv_path}")
                
                tracker["stats"][topic]["patents"] = len(patent_urls)
                tracker["stats"][topic]["last_update"] = time.strftime("%Y-%m-%d %H:%M:%S")
                tracker["downloaded_urls"].extend(patent_urls)
                save_tracker(tracker)
                
                if DOWNLOAD_XML:
                    successful, failed = batch_download_xmls(patent_urls, topic, tracker)
                    tracker["stats"][topic]["xmls_downloaded"] = successful
                    save_tracker(tracker)
            else:
                print(f"\n⚠️  No patents found for {topic}")
    
    finally:
        driver.quit()
        print("\n✅ Browser closed")
    
    print("\n" + "=" * 70)
    print("📊 FINAL STATISTICS")
    print("=" * 70)
    for topic, stats in tracker["stats"].items():
        print(f"{topic.upper()}:")
        print(f"  - Patent URLs: {stats['patents']}")
        print(f"  - XMLs downloaded: {stats.get('xmls_downloaded', 0)}")
    print(f"\nTotal unique XMLs: {len(tracker['downloaded_xmls'])}")
    print(f"Failed downloads: {len(tracker['failed_downloads'])}")
    print(f"\nCSV: {CSV_DIR}")
    print(f"XML: {XML_DIR}")
    print("=" * 70)

if __name__ == "__main__":
    main()
