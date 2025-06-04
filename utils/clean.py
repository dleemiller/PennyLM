import re
from typing import List, Dict


def extract_penny_journal_segments(
    content: str, min_words: int = 50, max_words: int = 150
) -> List[Dict]:
    """
    Extract high-quality text segments from Irish Penny Journal content.

    Args:
        content: Raw text content from penny journal file
        min_words: Minimum word count per segment (default 50)
        max_words: Maximum word count per segment (default 150)

    Returns:
        List of dictionaries with 'text', 'word_count', and 'char_count' keys
    """
    # Clean the content
    cleaned_content = clean_gutenberg_text(content)

    # Extract segments using sentence-based approach
    raw_segments = sentence_based_segmentation(cleaned_content, min_words, max_words)

    # Apply strict quality filters
    quality_segments = []
    for segment in raw_segments:
        if is_high_quality_prose(segment["text"], min_words):
            quality_segments.append(segment)

    return quality_segments


def clean_gutenberg_text(content: str) -> str:
    """Remove Project Gutenberg headers, footers, and publication metadata"""

    # Remove Project Gutenberg boilerplate
    content = re.sub(r"\*\*\* START OF.*?\*\*\*", "", content, flags=re.DOTALL)
    content = re.sub(r"\*\*\* END OF.*?\*\*\*", "", content, flags=re.DOTALL)
    content = re.sub(r"End of the Project Gutenberg.*$", "", content, flags=re.DOTALL)

    # Remove publication footer
    content = re.sub(
        r"Printed and Published.*?Glasgow\.$", "", content, flags=re.DOTALL
    )

    # Remove leading quote marks
    content = re.sub(r'^[\'"]', "", content)

    # Normalize whitespace
    content = re.sub(r"\n{3,}", "\n\n", content)
    content = re.sub(r"[ \t]+", " ", content)

    return content.strip()


def sentence_based_segmentation(
    content: str, min_words: int, max_words: int
) -> List[Dict]:
    """
    Split content into segments by combining sentences to reach target word counts
    """
    segments = []

    # Split by sentence endings while preserving punctuation
    sentence_parts = re.split(r"([.!?]+\s+)", content)

    current_text = ""
    current_word_count = 0

    for i in range(0, len(sentence_parts), 2):
        sentence = sentence_parts[i] if i < len(sentence_parts) else ""
        punctuation = sentence_parts[i + 1] if i + 1 < len(sentence_parts) else ""

        if sentence.strip():
            sentence_words = len(sentence.split())
            potential_text = current_text + sentence + punctuation
            potential_word_count = current_word_count + sentence_words

            # If adding this sentence would exceed max_words, save current segment first
            if potential_word_count > max_words and current_word_count >= min_words:
                segments.append(create_segment_dict(current_text.strip()))
                current_text = sentence + punctuation
                current_word_count = sentence_words
            else:
                # Add sentence to current segment
                current_text = potential_text
                current_word_count = potential_word_count

                # If we've reached a good size, save the segment
                if current_word_count >= min_words and current_word_count <= max_words:
                    # Look ahead to see if next sentence would fit
                    next_sentence_idx = i + 2
                    if next_sentence_idx < len(sentence_parts):
                        next_sentence = sentence_parts[next_sentence_idx]
                        next_words = (
                            len(next_sentence.split()) if next_sentence.strip() else 0
                        )

                        # If next sentence won't fit, save current segment
                        if current_word_count + next_words > max_words:
                            segments.append(create_segment_dict(current_text.strip()))
                            current_text = ""
                            current_word_count = 0

    # Handle remaining content
    if current_text.strip() and current_word_count >= min_words:
        segments.append(create_segment_dict(current_text.strip()))

    return segments


def create_segment_dict(text: str) -> Dict:
    """Create standardized segment dictionary"""
    return {"text": text, "word_count": len(text.split()), "char_count": len(text)}


def is_high_quality_prose(text: str, min_words: int) -> bool:
    """
    Strict quality filter to ensure only genuine prose content
    """
    if not text or not text.strip():
        return False

    text = text.strip()
    words = text.split()
    word_count = len(words)

    # Must meet minimum word count
    if word_count < min_words:
        return False

    # Skip journal headers and metadata
    if re.search(
        r"IRISH PENNY JOURNAL|VOLUME|NUMBER \d+|SATURDAY|OCTOBER", text, re.IGNORECASE
    ):
        return False

    # Skip illustration references
    if re.search(r"\[Illustration[:\]]", text, re.IGNORECASE):
        return False

    # Skip author bylines
    if re.match(r"^BY [A-Z\s\.]+\.?\s*", text, re.IGNORECASE):
        return False

    # Skip standalone author initials
    if re.match(r"^[A-Z]\.?\s*$", text.strip()):
        return False

    # Skip titles (high ratio of uppercase, short segments)
    uppercase_count = sum(1 for c in text if c.isupper())
    uppercase_ratio = uppercase_count / len(text) if text else 0
    if uppercase_ratio > 0.4 and word_count < 25:
        return False

    # Skip poetry and verse (short lines pattern)
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    if len(lines) > 2:
        short_lines = sum(1 for line in lines if len(line.split()) < 10)
        if short_lines / len(lines) > 0.7:  # More than 70% short lines = likely poetry
            return False

    # Skip quotes and epigraphs
    if re.match(r'^"[^"]*"$', text.strip()) and word_count < 40:
        return False

    # Must be primarily alphabetic (75% letters)
    alpha_count = sum(1 for c in text if c.isalpha())
    alpha_ratio = alpha_count / len(text) if text else 0
    if alpha_ratio < 0.75:
        return False

    # Must contain common English words (indicates actual prose)
    text_lower = text.lower()
    common_words = [
        "the",
        "and",
        "of",
        "to",
        "a",
        "in",
        "is",
        "that",
        "for",
        "with",
        "as",
        "was",
        "his",
        "he",
        "it",
        "on",
        "be",
        "at",
    ]
    common_word_matches = sum(
        1 for word in common_words if f" {word} " in f" {text_lower} "
    )
    if common_word_matches < 5:  # Must have at least 5 common English words
        return False

    # Skip content with too many special characters
    special_chars = re.findall(r'[^a-zA-Z\s\.,;:!?\-\'"()\n]', text)
    special_ratio = len(special_chars) / len(text) if text else 0
    if special_ratio > 0.1:
        return False

    # Skip number-heavy content
    digit_count = sum(1 for c in text if c.isdigit())
    digit_ratio = digit_count / len(text) if text else 0
    if digit_ratio > 0.15:
        return False

    return True


def analyze_segments(segments: List[Dict]) -> Dict:
    """Generate statistics about the extracted segments"""
    if not segments:
        return {"error": "No segments to analyze"}

    word_counts = [seg["word_count"] for seg in segments]
    char_counts = [seg["char_count"] for seg in segments]

    return {
        "total_segments": len(segments),
        "total_words": sum(word_counts),
        "avg_words_per_segment": sum(word_counts) / len(word_counts),
        "min_words": min(word_counts),
        "max_words": max(word_counts),
        "avg_chars_per_segment": sum(char_counts) / len(char_counts),
        "word_count_distribution": {
            "under_50": len([w for w in word_counts if w < 50]),
            "50_to_100": len([w for w in word_counts if 50 <= w < 100]),
            "100_to_150": len([w for w in word_counts if 100 <= w < 150]),
            "over_150": len([w for w in word_counts if w >= 150]),
        },
    }


def save_segments_to_file(
    segments: List[Dict], filename: str = "penny_journal_segments.txt"
):
    """Save extracted segments to a text file"""
    with open(filename, "w", encoding="utf-8") as f:
        f.write("Irish Penny Journal - High Quality Text Segments\n")
        f.write("=" * 50 + "\n")
        f.write(f"Total segments: {len(segments)}\n")
        f.write(f"Extraction settings: 50-150 words per segment\n")
        f.write(f"Quality filters: Applied strict prose validation\n\n")

        for i, segment in enumerate(segments, 1):
            f.write(f"SEGMENT {i:03d}\n")
            f.write(f"Words: {segment['word_count']}\n")
            f.write(f"Characters: {segment['char_count']}\n")
            f.write("-" * 30 + "\n")
            f.write(segment["text"])
            f.write("\n\n")


def print_sample_segments(segments: List[Dict], num_samples: int = 5):
    """Print sample segments for quality inspection"""
    if not segments:
        print("No segments to display")
        return

    print(f"\n📝 Sample segments (showing {min(num_samples, len(segments))}):")

    # Show segments from different parts of the text
    indices = []
    if len(segments) >= num_samples:
        step = len(segments) // num_samples
        indices = [i * step for i in range(num_samples)]
    else:
        indices = list(range(len(segments)))

    for i, idx in enumerate(indices):
        if idx < len(segments):
            seg = segments[idx]
            print(f"\n   Sample {i+1} (Segment {idx+1}) - {seg['word_count']} words:")
            preview = (
                seg["text"][:200] + "..." if len(seg["text"]) > 200 else seg["text"]
            )
            print(f"   {preview}")
