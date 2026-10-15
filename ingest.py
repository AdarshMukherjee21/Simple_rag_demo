import os
import json
import time
from matplotlib.pylab import rint
import pymupdf4llm
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

def process_document(pdf_path: str):
    total_start_time = time.time()
    
    # Path extraction
    pdf_dir = os.path.dirname(pdf_path)
    file_name = os.path.splitext(os.path.basename(pdf_path))[0]
    
    os.makedirs(pdf_dir, exist_ok=True)
    print(f"[{file_name}] Started processing document: {pdf_path}")

    # 1. PDF to Markdown Conversion
    print(f"[{file_name}] Converting PDF to Markdown...")
    t0 = time.time()
    md_text = pymupdf4llm.to_markdown(pdf_path)
    md_path = os.path.join(pdf_dir, f"{file_name}.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_text)
    t_markdown = round(time.time() - t0, 3)
    print(f"[{file_name}] Saved Markdown to '{md_path}' ({t_markdown}s)")

    # 2. Structural Chunking
    print(f"[{file_name}] Performing structural chunking based on Markdown headers...")
    t0 = time.time()
    
    # Define header levels to split on
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    
    # Step A: Split strictly by structural headers
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on, 
        strip_headers=False
    )
    header_splits = markdown_splitter.split_text(md_text)

    # Step B: Sub-chunk oversized sections while preserving header metadata
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=150
    )
    docs = text_splitter.split_documents(header_splits)
    
    # Inject source file metadata into each chunk
    for doc in docs:
        doc.metadata["source"] = file_name

    t_chunking = round(time.time() - t0, 3)
    print(f"[{file_name}] Created {len(docs)} structural chunks ({t_chunking}s)")

    # 3. Generate Embeddings & Build FAISS Index
    print(f"[{file_name}] Generating embeddings with Ollama (qwen3-embedding:4b)...")
    t0 = time.time()
    embeddings = OllamaEmbeddings(model="qwen3-embedding:4b")
    vectorstore = FAISS.from_documents(docs, embeddings)
    
    faiss_dir = os.path.join(pdf_dir, "faiss_folder")
    vectorstore.save_local(faiss_dir)
    t_embedding = round(time.time() - t0, 3)
    print(f"[{file_name}] FAISS index created & saved to '{faiss_dir}' ({t_embedding}s)")

    # 4. Total Time & Metadata Export
    total_time = round(time.time() - total_start_time, 3)
    
    meta = {
        "file_name": os.path.basename(pdf_path),
        "file_size_bytes": os.path.getsize(pdf_path),
        "chunks_created": len(docs),
        "timings_seconds": {
            "pdf_to_markdown": t_markdown,
            "text_chunking": t_chunking,
            "embedding_and_indexing": t_embedding,
            "total_processing_time": total_time
        }
    }
    
    meta_path = os.path.join(pdf_dir, "meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=4)
        
    print(f"[{file_name}] Saved execution metadata to '{meta_path}'")
    print(f"[{file_name}] Processing completed successfully in {total_time}s\n")

    return meta