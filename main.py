import os
import json
from pathlib import Path
from src.parser.pipeline import ParserPipeline
from src.chunker.legal_chunker import LegalChunker


def run_pipeline():
    base_data_dir = Path("data/raw")
    years = ["1962", "2026"]
    
    total_docs = 0
    total_chunks = 0
    total_tables = 0
    total_footnotes = 0
    
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("STARTING TEMPORAL-RAG PARSING & CHUNKING PIPELINE")
    print("=" * 80)

    for year in years:
        year_dir = base_data_dir / year
        if not year_dir.exists():
            continue
            
        print(f"\n--- Processing {year} Rules ---")
        docs = ParserPipeline.parse_directory(str(year_dir))
        
        all_year_chunks = []
        for doc in docs:
            chunks = LegalChunker.chunk_document(doc)
            all_year_chunks.extend(chunks)
            total_docs += 1
            total_chunks += len(chunks)
            total_tables += len(doc.extracted_tables)
            total_footnotes += len(doc.footnotes)
            
            print(f"Rule {doc.rule_id:8} | Title: {doc.title[:35]:35} | Nodes: {len(doc.root_nodes):2} | Chunks: {len(chunks):2} | Tables: {len(doc.extracted_tables)}")

        # Save processed chunks to JSON
        chunks_file = output_dir / f"chunks_{year}.json"
        with open(chunks_file, "w", encoding="utf-8") as f:
            json.dump([c.model_dump() for c in all_year_chunks], f, indent=2)
        print(f"Saved {len(all_year_chunks)} chunks to {chunks_file}")

    print("\n" + "=" * 80)
    print(f"PIPELINE COMPLETE: {total_docs} Rules Processed, {total_chunks} Chunks Created, {total_tables} Tables Extracted, {total_footnotes} Footnotes Cataloged.")
    print("=" * 80)


if __name__ == "__main__":
    run_pipeline()
