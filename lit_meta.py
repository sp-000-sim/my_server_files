#!/usr/bin/env python3
"""
OpenAlex Metadata Collector
Collects paper metadata only (no downloads)
"""

import requests
import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import time

OUTPUT_DIR = Path("./literature")
METADATA_DIR = OUTPUT_DIR / "metadata"
OPENALEX_API = "https://api.openalex.org/works"

TOPIC_QUERIES = {
    "co2_methanation": [
        "CO2 methanation catalyst",
        "carbon dioxide methanation nickel",
        "Sabatier reaction CO2",
        "CO2 hydrogenation methane",
        "methanation ruthenium catalyst"
    ],
    "fischer_tropsch": [
        "Fischer-Tropsch synthesis catalyst",
        "FT synthesis cobalt iron",
        "syngas to hydrocarbons",
        "Fischer-Tropsch reactor design",
        "FT wax hydrocracking"
    ],
    "hydrocracking": [
        "hydrocracking catalyst zeolite",
        "hydrocracking NiMo CoMo",
        "heavy oil hydrocracking",
        "hydrocracking mechanism",
        "bifunctional hydrocracking catalyst"
    ]
}

YEARS_FROM = 2015
YEARS_TO = 2025
MAX_RESULTS_PER_QUERY = 1000

def search_openalex(query, from_year=2015, to_year=2025, per_page=200, cursor="*"):
    params = {
        "search": query,
        "filter": f"publication_year:{from_year}-{to_year},type:article",
        "per-page": per_page,
        "cursor": cursor,
        "mailto": "researcher@example.com"
    }
    
    try:
        response = requests.get(OPENALEX_API, params=params, timeout=30)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"  ⚠️  API error: {e}")
        return None

def get_all_papers_for_query(query, from_year, to_year, max_results):
    all_papers = []
    cursor = "*"
    
    with tqdm(desc=f"  Fetching", unit=" papers") as pbar:
        while len(all_papers) < max_results:
            data = search_openalex(query, from_year, to_year, per_page=200, cursor=cursor)
            
            if not data or "results" not in data:
                break
            
            papers = data["results"]
            if not papers:
                break
            
            all_papers.extend(papers)
            pbar.update(len(papers))
            
            meta = data.get("meta", {})
            next_cursor = meta.get("next_cursor")
            
            if not next_cursor or len(all_papers) >= max_results:
                break
            
            cursor = next_cursor
            time.sleep(0.1)
    
    return all_papers[:max_results]

def reconstruct_abstract(inverted_index):
    try:
        if not inverted_index:
            return ""
        words = [""] * (max(max(positions) for positions in inverted_index.values()) + 1)
        for word, positions in inverted_index.items():
            for pos in positions:
                words[pos] = word
        return " ".join(words)
    except:
        return ""

def extract_paper_metadata(paper):
    """Fixed metadata extraction with null checks"""
    try:
        # Safe get with defaults
        paper_id = paper.get("id", "") or ""
        doi = paper.get("doi", "") or ""
        title = paper.get("title") or ""
        
        # Handle None in ID/DOI
        if paper_id:
            paper_id = str(paper_id).replace("https://openalex.org/", "")
        if doi:
            doi = str(doi).replace("https://doi.org/", "")
        
        # Authors
        authors = []
        for authorship in paper.get("authorships", []) or []:
            author = authorship.get("author") or {}
            author_name = author.get("display_name")
            if author_name:
                authors.append(author_name)
        
        # Open access
        oa = paper.get("open_access") or {}
        is_oa = oa.get("is_oa", False)
        oa_url = oa.get("oa_url") or ""
        oa_status = oa.get("oa_status") or ""
        
        # Primary location (journal)
        primary_location = paper.get("primary_location") or {}
        source = primary_location.get("source") or {}
        journal = source.get("display_name") or ""
        
        # Abstract
        abstract_inverted = paper.get("abstract_inverted_index")
        abstract = reconstruct_abstract(abstract_inverted) if abstract_inverted else ""
        
        # Concepts
        concepts = []
        for c in (paper.get("concepts") or [])[:5]:
            if c and c.get("display_name"):
                concepts.append(c["display_name"])
        
        metadata = {
            "openalex_id": paper_id,
            "doi": doi,
            "title": title,
            "authors": "; ".join(authors),
            "publication_year": paper.get("publication_year") or "",
            "publication_date": paper.get("publication_date") or "",
            "journal": journal,
            "cited_by_count": paper.get("cited_by_count", 0),
            "is_open_access": is_oa,
            "oa_status": oa_status,  # gold, green, hybrid, bronze, closed
            "oa_url": oa_url,
            "abstract": abstract,
            "concepts": "; ".join(concepts),
            "openalex_url": paper.get("id", "") or ""
        }
        
        return metadata
    except Exception as e:
        # Silent fail, return None
        return None

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    METADATA_DIR.mkdir(exist_ok=True)
    
    print("=" * 70)
    print("OpenAlex Metadata Collector")
    print("=" * 70)
    
    for topic, queries in TOPIC_QUERIES.items():
        print(f"\n{'='*70}")
        print(f"📚 {topic.upper()}")
        print(f"{'='*70}")
        
        topic_papers = []
        
        for query in queries:
            print(f"\n🔎 Query: {query}")
            papers = get_all_papers_for_query(query, YEARS_FROM, YEARS_TO, MAX_RESULTS_PER_QUERY)
            print(f"  ✅ Found {len(papers)} papers")
            topic_papers.extend(papers)
        
        # Deduplicate by ID
        unique_papers = {}
        for paper in topic_papers:
            paper_id = paper.get("id", "")
            if paper_id and paper_id not in unique_papers:
                unique_papers[paper_id] = paper
        
        print(f"\n📊 Total unique papers: {len(unique_papers)}")
        
        # Extract metadata
        metadata_list = []
        oa_count = 0
        
        for paper in unique_papers.values():
            metadata = extract_paper_metadata(paper)
            if metadata:
                metadata_list.append(metadata)
                if metadata["is_open_access"]:
                    oa_count += 1
        
        if metadata_list:
            df = pd.DataFrame(metadata_list)
            
            # Save CSV
            csv_path = METADATA_DIR / f"{topic}_papers.csv"
            df.to_csv(csv_path, index=False)
            
            # Save JSON
            json_path = METADATA_DIR / f"{topic}_papers.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(metadata_list, f, indent=2, ensure_ascii=False)
            
            print(f"\n✅ Saved {len(metadata_list)} papers")
            print(f"📄 CSV: {csv_path}")
            print(f"📂 JSON: {json_path}")
            print(f"🔓 Open access: {oa_count}/{len(metadata_list)} ({oa_count*100//len(metadata_list)}%)")
    
    print("\n" + "=" * 70)
    print("✅ Metadata collection complete!")
    print(f"Location: {METADATA_DIR}")
    print("=" * 70)

if __name__ == "__main__":
    main()
