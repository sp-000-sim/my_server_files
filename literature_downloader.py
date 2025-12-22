#!/usr/bin/env python3
"""
Parallel PDF Downloader for OpenAlex Papers
Uses ThreadPoolExecutor for concurrent downloads
"""

import pandas as pd
import requests
from pathlib import Path
from tqdm import tqdm
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from urllib.parse import urlparse

# Try to import python-magic, fallback to basic validation
try:
    import magic
    HAS_MAGIC = True
except ImportError:
    HAS_MAGIC = False
    print("⚠️  python-magic not found, using basic PDF validation")

METADATA_DIR = Path("./literature/metadata")
PDF_DIR = Path("./literature/pdfs")
FAILED_LOG = Path("./literature/failed_downloads.txt")

# ==================== CONFIG ====================
NUM_WORKERS = 20  # Number of parallel download threads
TIMEOUT = 30  # Download timeout in seconds
RATE_LIMIT_DELAY = 0.1  # Delay between requests per worker (seconds)

# Thread-safe counters
class DownloadStats:
    def __init__(self):
        self.lock = threading.Lock()
        self.successful = 0
        self.failed = 0
        self.skipped = 0
        self.failed_reasons = {}
    
    def add_success(self):
        with self.lock:
            self.successful += 1
    
    def add_failed(self, reason):
        with self.lock:
            self.failed += 1
            self.failed_reasons[reason] = self.failed_reasons.get(reason, 0) + 1
    
    def add_skipped(self):
        with self.lock:
            self.skipped += 1
    
    def get_stats(self):
        with self.lock:
            return {
                'successful': self.successful,
                'failed': self.failed,
                'skipped': self.skipped,
                'failed_reasons': dict(self.failed_reasons)
            }

# ==================== VALIDATION ====================
def is_valid_pdf(file_path):
    """Check if file is actually a PDF"""
    try:
        if HAS_MAGIC:
            mime = magic.Magic(mime=True)
            file_type = mime.from_file(str(file_path))
            return file_type == 'application/pdf'
        else:
            # Fallback: check file signature
            with open(file_path, 'rb') as f:
                header = f.read(4)
                return header == b'%PDF'
    except Exception as e:
        return False

# ==================== DOWNLOAD FUNCTION ====================
def download_pdf_smart(url, output_path, timeout=30):
    """
    Smart PDF download with validation
    Returns: (success, reason)
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'application/pdf,application/octet-stream,*/*'
        }
        
        # HEAD request first to check content-type
        try:
            head_response = requests.head(url, headers=headers, timeout=5, allow_redirects=True)
            content_type = head_response.headers.get('Content-Type', '').lower()
            
            # Skip if clearly HTML
            if 'text/html' in content_type:
                return False, "HTML page"
        except:
            pass  # Continue with download if HEAD fails
        
        # Download with streaming
        response = requests.get(url, headers=headers, timeout=timeout, stream=True, allow_redirects=True)
        
        if response.status_code != 200:
            return False, f"HTTP {response.status_code}"
        
        # Check content type from actual response
        content_type = response.headers.get('Content-Type', '').lower()
        if 'text/html' in content_type:
            return False, "HTML response"
        
        # Download to file
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        
        # Validate file size
        file_size = output_path.stat().st_size
        if file_size < 1000:  # Less than 1KB
            output_path.unlink()
            return False, "File too small"
        
        # Validate PDF format
        if not is_valid_pdf(output_path):
            output_path.unlink()
            return False, "Invalid PDF format"
        
        return True, "Success"
        
    except requests.Timeout:
        if output_path.exists():
            output_path.unlink()
        return False, "Timeout"
    except requests.ConnectionError:
        if output_path.exists():
            output_path.unlink()
        return False, "Connection error"
    except Exception as e:
        if output_path.exists():
            output_path.unlink()
        return False, str(e)[:50]

# ==================== WORKER FUNCTION ====================
def download_worker(task):
    """
    Worker function for parallel downloads
    task = (url, output_path, paper_info)
    Returns: (success, reason, paper_info)
    """
    url, output_path, paper_info = task
    
    # Check if already exists and valid
    if output_path.exists():
        if is_valid_pdf(output_path):
            return True, "Already exists", paper_info
        else:
            output_path.unlink()  # Remove invalid file
    
    # Download
    success, reason = download_pdf_smart(url, output_path, timeout=TIMEOUT)
    
    # Rate limiting
    time.sleep(RATE_LIMIT_DELAY)
    
    return success, reason, paper_info

# ==================== LOGGING ====================
def log_failed_download(topic, url, reason, paper_info):
    """Thread-safe logging of failed downloads"""
    lock = threading.Lock()
    with lock:
        with open(FAILED_LOG, 'a', encoding='utf-8') as f:
            title = paper_info.get('title', 'Unknown')
            doi = paper_info.get('doi', 'N/A')
            f.write(f"{topic}\t{doi}\t{url}\t{reason}\t{title}\n")

# ==================== MAIN DOWNLOAD FUNCTION ====================
def download_pdfs_for_topic_parallel(csv_path, topic, num_workers=8):
    """Download PDFs for one topic using parallel workers"""
    print(f"\n{'='*70}")
    print(f"📥 {topic.upper()}")
    print(f"{'='*70}")
    
    # Read metadata
    df = pd.read_csv(csv_path)
    print(f"📊 Total papers: {len(df)}")
    
    # Filter open access only
    oa_df = df[df['is_open_access'] == True].copy()
    print(f"🔓 Open access: {len(oa_df)}")
    
    # Create topic PDF directory
    topic_pdf_dir = PDF_DIR / topic
    topic_pdf_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare download tasks
    tasks = []
    for idx, row in oa_df.iterrows():
        oa_url = row.get('oa_url')
        
        if pd.isna(oa_url) or not oa_url:
            continue
        
        # Create filename from DOI or OpenAlex ID
        # FIX: Handle NaN values properly
        doi = row.get('doi', '')
        openalex_id = row.get('openalex_id', '')
        
        # Convert to string and handle NaN
        if pd.notna(doi) and doi:
            doi_str = str(doi)
            filename = doi_str.replace('/', '_').replace('\\', '_').replace(':', '_') + '.pdf'
        elif pd.notna(openalex_id) and openalex_id:
            openalex_str = str(openalex_id)
            filename = openalex_str.replace('/', '_') + '.pdf'
        else:
            filename = f"paper_{idx}.pdf"
        
        # Sanitize filename (remove any remaining invalid characters)
        filename = "".join(c for c in filename if c.isalnum() or c in ('_', '-', '.')).rstrip()
        
        # Ensure filename is not empty
        if not filename or filename == '.pdf':
            filename = f"paper_{idx}.pdf"
        
        pdf_path = topic_pdf_dir / filename
        
        # Paper info for logging (convert to string to avoid NaN issues)
        paper_info = {
            'title': str(row.get('title', '')) if pd.notna(row.get('title')) else '',
            'doi': str(doi) if pd.notna(doi) and doi else 'N/A',
            'openalex_id': str(openalex_id) if pd.notna(openalex_id) and openalex_id else 'N/A'
        }
        
        tasks.append((oa_url, pdf_path, paper_info))
    
    print(f"📋 Tasks to process: {len(tasks)}")
    print(f"👷 Using {num_workers} parallel workers\n")
    
    # Initialize stats
    stats = DownloadStats()
    
    # Execute downloads in parallel
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        future_to_task = {executor.submit(download_worker, task): task for task in tasks}
        
        # Process completed downloads with progress bar
        with tqdm(total=len(tasks), desc="  Downloading", unit=" PDFs") as pbar:
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                url, output_path, paper_info = task
                
                try:
                    success, reason, paper_info = future.result()
                    
                    if success:
                        if reason == "Already exists":
                            stats.add_skipped()
                        else:
                            stats.add_success()
                    else:
                        stats.add_failed(reason)
                        log_failed_download(topic, url, reason, paper_info)
                    
                except Exception as e:
                    stats.add_failed(f"Worker error: {str(e)[:30]}")
                    log_failed_download(topic, url, f"Worker error: {e}", paper_info)
                
                pbar.update(1)
    
    # Get final stats
    final_stats = stats.get_stats()
    
    print(f"\n✅ Successfully downloaded: {final_stats['successful']}")
    print(f"⏭️  Already existed: {final_stats['skipped']}")
    print(f"❌ Failed: {final_stats['failed']}")
    
    if final_stats['failed_reasons']:
        print(f"\n📊 Failure reasons:")
        for reason, count in sorted(final_stats['failed_reasons'].items(), key=lambda x: -x[1])[:10]:
            print(f"  - {reason}: {count}")
    
    return final_stats['successful'], final_stats['failed'], final_stats['skipped']

# ==================== MAIN ====================
def main():
    PDF_DIR.mkdir(exist_ok=True)
    
    # Create failed log with header
    if not FAILED_LOG.exists():
        with open(FAILED_LOG, 'w', encoding='utf-8') as f:
            f.write("Topic\tDOI\tURL\tReason\tTitle\n")
    
    print("=" * 70)
    print("Parallel PDF Downloader for OpenAlex Papers")
    print(f"Workers: {NUM_WORKERS} | Timeout: {TIMEOUT}s")
    print("=" * 70)
    
    # Find all metadata CSVs
    csv_files = list(METADATA_DIR.glob("*_papers.csv"))
    
    if not csv_files:
        print("\n❌ No metadata CSV files found!")
        print(f"Expected location: {METADATA_DIR}")
        return
    
    print(f"\nFound {len(csv_files)} metadata files")
    
    total_success = 0
    total_failed = 0
    total_skipped = 0
    
    start_time = time.time()
    
    # Process each topic
    for csv_path in csv_files:
        topic = csv_path.stem.replace('_papers', '')
        success, failed, skipped = download_pdfs_for_topic_parallel(
            csv_path, 
            topic, 
            num_workers=NUM_WORKERS
        )
        total_success += success
        total_failed += failed
        total_skipped += skipped
    
    elapsed_time = time.time() - start_time
    
    print("\n" + "=" * 70)
    print("📊 FINAL STATISTICS")
    print("=" * 70)
    print(f"✅ Total PDFs downloaded: {total_success}")
    print(f"⏭️  Already existed: {total_skipped}")
    print(f"❌ Total failed: {total_failed}")
    print(f"⏱️  Time taken: {elapsed_time:.1f}s")
    if total_success > 0:
        print(f"📈 Download rate: {(total_success/elapsed_time)*60:.1f} PDFs/minute")
    print(f"\n📁 PDFs saved to: {PDF_DIR}")
    print(f"📋 Failed URLs log: {FAILED_LOG}")
    print("=" * 70)

if __name__ == "__main__":
    # Install: pip install pandas requests tqdm python-magic-bin
    main()
