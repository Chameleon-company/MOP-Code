import hashlib
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
import tiktoken
import nltk
from nltk.tokenize import sent_tokenize
nltk.download('punkt', quiet=True)

_TOKENIZER = tiktoken.get_encoding("o200k_base") # Commonly use in OpenAI models
def count_tokens(text: str) -> int:
    """Return the exact token count for *text* using tiktoken."""
    return len(_TOKENIZER.encode(text))

# Each chunk will be an object of Chunk class
@dataclass
class Chunk:
    """
    chunk_id: Stable SHA-256 fingerprint. Used for primary key in a vector store.
            Same id will be use for identical text, so re-indexing a document is idempotent (unchanged)
    
    doc_title: Document title derive from document filename
    
    source_file: Absolute or relative path
    
    section_path: Breadcrumb of heading. Empty string if no heading.
                    E.g.
                    ```
                    # Chapter 3
                    ## Retrieval
                    ### Dense Retrieval
                    ```
                    It will be "Chapter 3 > Retrieval > Dense Retrieval"
                    
    prev_heading: The parent heading one level up from the current section. Useful for adding context.
                
    chunk_type: One of: "paragraph", "bullet_list", "table", "code_block",
                "mixed", "empty".
    
    chunk_text: Raw chunk text
    
    embedding_text: chunk_text after processed with metadata. This is fed to embedding model.
    
    token_count: Cached token count of embedding_text, so callers don't need to re-tokenize when building batches. 
    """
    chunk_id: str
    doc_title: str
    source_file: str
    section_path: str
    prev_heading: Optional[str]
    chunk_type: str
    chunk_text: str
    embedding_text: str
    token_count: int = field(default=0)
    
# Matches any markdown heading: # … through ###### …
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$")

# Matches a line that starts a bullet or numbered list item
LIST_ITEM_RE = re.compile(r"^\s*([-*+]|\d+\.)\s+")

# Matches a markdown table row (contains at least one pipe character)
TABLE_ROW_RE = re.compile(r"^\|.+\|")


def parse_markdown_sections(markdown_text: str) -> List[dict]:
    """
    Split a markdown document into a list of sections
    
    Each section and its content the falls under a particular heading. 
    The function tracks:
        - section_path: breadcrumb heading path
        - prev_heading: Immediate closing parent heading
    
    Returns a list of dicts each with keys: section_path, prev_heading, content
    """
    
    lines = markdown_text.splitlines()
    sections: List[dict] = []
    
    # Track hierarchy of current heading. 
    # e.g., [(1, "Chapter 1"), (2, "Section A")] = "Chapter 1 > Section A"
    heading_stack: List[tuple] = []

    # Accumulate lines until we hit the next heading
    current_lines: List[str] = []
    
    def flush_section() -> None:
        """Push the accumulated lines as a completed section."""
        nonlocal current_lines
        
        # Skip empty section
        content = "\n".join(current_lines).strip()
        if not content:
            current_lines = []
            return

        # Build the breadcrumb path from the heading stack
        section_path = " > ".join(h[1] for h in heading_stack)
        
        # Extract parent heading (one level up in hierarchy)
        # Useful when current heading is vague like "Overview" or "Introduction"
        prev_heading = heading_stack[-2][1] if len(heading_stack) >= 2 else None

        sections.append(
            {
                "section_path": section_path,
                "prev_heading": prev_heading,
                "content": content,
            }
        )
        
        current_lines = []
        
    # Process each line
    for line in lines:
        
        # Check for markdown headings like # or ## or ###
        heading_match = HEADING_RE.match(line)
        if heading_match:
            # New heading, finalize the previous section first.
            flush_section()

            # Extract the heading level and text
            level = len(heading_match.group(1))  
            heading_text = heading_match.group(2).strip()
            
            # Update the hierarchy: pop headings at same/deeper level
            # This ensures heading_stack is always a valid ancestor chain
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()

            heading_stack.append((level, heading_text))
        else:
            current_lines.append(line)
    
    flush_section()
    
    return sections


def split_into_blocks(section_text: str) -> List[str]:
    """
    Each section is split into sematic blocks. 
    A block is group of lines separated from its neighbor by one or more blank lines.
    Except fenced code blocks, we use ``` ... ```.
    
    Returns a list of non-empty block strings
    """
    lines = section_text.splitlines()
    blocks: List[str] = []
    current_block: List[str] = []
    in_code_fence = False  # True while we're inside a ``` … ``` pair
    
    for line in lines:
        # Code-fence flag on for code block
        if line.strip().startswith("```"):
            in_code_fence = not in_code_fence
            current_block.append(line)
            continue
        
        if not in_code_fence and not line.strip():
            # Blank line and not code block -> end of current block
            if current_block:
                blocks.append("\n".join(current_block).strip())
                current_block = []
        else:
            current_block.append(line)
        
    # Flush the last block
    if current_block:
        blocks.append("\n".join(current_block).strip())

    non_empty = [b for b in blocks if b.strip()]
    return non_empty

def normalize_whitespace(text: str) -> str:
    """
    This function cleans blank lines in blocks by
    collapsing runs of 3+ consecutive blank lines down to exactly two.

    This keeps intentional paragraph breaks without letting runaway blank
    lines bloat token counts.
    """
    return re.sub(r"\n{3,}", "\n\n", text).strip()

def detect_chunk_type(block: str) -> str:
    """
    Classify a block into one of five sematic types.
    
    "code_block" - fenced code (``` … ```)
    "table"- markdown table rows
    "bullet_list" - bullet or numbered list items
    "paragraph" - continuous prose
    "mixed" - block contains more than one of the above
    "empty" - nothing here
    """
    stripped = block.strip()

    if not stripped:
        return "empty"
    
    # Code block
    lines = stripped.splitlines()
    if lines[0].strip().startswith("```") and lines[-1].strip().startswith("```"):
        return "code_block"
    
    # Dictionary counting types
    type_counts: dict[str, int] = {
        "table": 0,
        "list": 0,
        "prose": 0,
    }
    
    for line in lines:
        if not line.strip():
            continue  # blank separators don't count
        if TABLE_ROW_RE.match(line):
            type_counts["table"] += 1
        elif LIST_ITEM_RE.match(line):
            type_counts["list"] += 1
        else:
            type_counts["prose"] += 1
            
    # Determine how many distinct types appear
    types_count = [t for t, n in type_counts.items() if n > 0]
    
    if len(types_count) > 1:
        return "mixed"
    
    if type_counts["table"] > 0:
        return "table"
    if type_counts["list"] > 0:
        return "bullet_list"

    return "paragraph"

def split_block_by_tokens(
    block: str,
    max_tokens: int = 4000,
    overlap_tokens: int = 40,
) -> List[str]:
    """
    Further split a block if it exceeds max_token into smaller chunks
    using a sentence-level sliding window. I am using gpt5 nano model with
    400,000 token context window, so I decide to set the max_token to high.
    The focus is not constrain but retrieval quality.
    
    Args:
        block         : The text to split.
        max_tokens    : Hard upper bound (in tokens) for each output chunk.
        overlap_tokens: How many tokens of the previous chunk to prepend to the next one. 
    """
    if count_tokens(block) <= max_tokens:
        return [block]
    
    # Split on sentence boundaries
    sentences = sent_tokenize(block.strip())
    
    chunks: List[str] = []
    current_sentences: List[str] = []
    current_token_count = 0
    
    for sentence in sentences:
        sentence_tokens = count_tokens(sentence)

        if current_token_count + sentence_tokens <= max_tokens:
            # Sentence fits — add it to the running chunk.
            current_sentences.append(sentence)
            current_token_count += sentence_tokens
        else:
            # Sentence would overflow — seal the current chunk first.
            if current_sentences:
                chunks.append(" ".join(current_sentences).strip())

            # Walk backwards through the sentences of the just-sealed chunk
            # and collect as many as fit within overlap_tokens.
            overlap_sentences: List[str] = []
            overlap_used = 0

            for prev_sentence in reversed(current_sentences):
                prev_tokens = count_tokens(prev_sentence)
                if overlap_used + prev_tokens <= overlap_tokens:
                    overlap_sentences.insert(0, prev_sentence)
                    overlap_used += prev_tokens
                else:
                    break

            # Start the new chunk with the overlap tail + the current sentence.
            current_sentences = overlap_sentences + [sentence]
            current_token_count = sum(count_tokens(s) for s in current_sentences)

    # Seal any remaining sentences as the final chunk.
    if current_sentences:
        chunks.append(" ".join(current_sentences).strip())

    return [c for c in chunks if c.strip()]

def build_embedding_text(
    doc_title: str,
    section_path: str,
    chunk_type: str,
    chunk_text: str,
) -> str:
    """
    Prepend structured metadata to the raw chunk text.

    Embedding models are just language models. They encode meaning by
    telling the model "this is from document X, section Y, and it's a table",
    we push the embedding into a part of the vector space that reflects both
    the content and its metadata. This makes retrieval more precise when
    a query asks about a specific section or document.
    """
    return (
        f"Document: {doc_title}\n"
        f"Section: {section_path or 'Unknown'}\n"
        f"Type: {chunk_type}\n\n"
        f"{chunk_text}"
    )
    
def make_chunk_id(source_file: str, section_path: str, chunk_text: str) -> str:
    """
    Produce a stable, unique identifier for a chunk.

    We hash three fields that together uniquely identify a piece of content:
      - source_file: different files with the same text get different ids
      - section_path: same text appearing in two sections gets different ids
      - chunk_text: the actual content

    SHA-256 gives a 64-character hex string.

    Because the hash is deterministic, re-running the chunker on an unchanged
    document produces the exact same ids, making upserts into a vector store
    safe (overwrite, not duplicate).
    """
    raw = f"{source_file}||{section_path}||{chunk_text}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

def chunk_markdown_document(
    markdown_text: str,
    source_file: str,
    max_tokens: int = 4000,
    overlap_tokens: int = 40,
) -> List[Chunk]:
    """
    Main chunking pipeline. Combine everything above. Markdown text to list of `Chunk` objects.
    
    Args:
        markdown_text: The full content of the markdown file as a string.
        source_file: Path to the source file.
        max_tokens: Target token budget per chunk.  
        overlap_tokens: How many tokens of context to carry over between
                        consecutive sub-chunks of the same block.

    Returns:
        A list of `Chunk` dataclass instances that is ready to embed and store.
    """
    
    if not markdown_text.strip():
        print("Empty document!")
        return []
    
    # Get clean doc title
    doc_title = Path(source_file).stem.replace("_", " ").strip()
    
    sections = parse_markdown_sections(markdown_text)
    if not sections:
        print("No section found")
        return []
    
    all_chunks: List[Chunk] = []
    
    for section in sections:
        section_path = section["section_path"] or doc_title
        prev_heading = section["prev_heading"]
        
        # 1. Split the section content into semantic blocks.
        blocks = split_into_blocks(section["content"])

        for block in blocks:
            block = normalize_whitespace(block)
            chunk_type = detect_chunk_type(block)

            # 2. Further split any block that is over max token.
            sub_chunks = split_block_by_tokens(
                block,
                max_tokens=max_tokens,
                overlap_tokens=overlap_tokens,
            )

            for sub_chunk in sub_chunks:
                # 3. Build the enriched embedding text.
                embedding_text = build_embedding_text(
                    doc_title=doc_title,
                    section_path=section_path,
                    chunk_type=chunk_type,
                    chunk_text=sub_chunk,
                )

                # 4. Generate a stable chunk id.
                chunk_id = make_chunk_id(source_file, section_path, sub_chunk)

                # 5. Cache the token count so callers don't re-tokenize.
                token_count = count_tokens(embedding_text)

                all_chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        doc_title=doc_title,
                        source_file=source_file,
                        section_path=section_path,
                        prev_heading=prev_heading,
                        chunk_type=chunk_type,
                        chunk_text=sub_chunk,
                        embedding_text=embedding_text,
                        token_count=token_count,
                    )
                )

    return all_chunks


# # Test
# file_path = r"D:\Deakin-Data-Science\T2Y3\SIT378 - Team project B\Project\Project 6 - Bridge and Road Crack Detection System\RAG pipeline\processed_documents\Austroads Guide to Bridge Technology Part 7.md"
# with open(file_path, "r", encoding="utf-8") as f:
#     markdown_text = f.read()
    
# chunks = chunk_markdown_document(
#     markdown_text=markdown_text,
#     source_file=file_path,
# )

# print(f"\n{'='*60}")
# print(f"  Total chunks produced: {len(chunks)}")
# print(f"{'='*60}\n")

# # Process first 20 chunks
# for i, chunk in enumerate(chunks[:20], 1):
#     print(f"--- Chunk {i} ---")
#     print(f"  id           : {chunk.chunk_id}")
#     print(f"  section_path : {chunk.section_path}")
#     print(f"  type         : {chunk.chunk_type}")
#     print(f"  tokens       : {chunk.token_count}")
#     print(f"  text preview : {chunk.chunk_text[:80]!r}")
#     print()