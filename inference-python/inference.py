import sys
import array as arr
from pathlib import Path
from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

ARTICLE_PATH = sys.argv[1]
ARTICLE = Path(ARTICLE_PATH).read_text()

# max input length is 4000?
SEGMENT_LIMIT = 4000

SENTENCES = ARTICLE.split(".")

SEGMENTS = []
SEGMENT = ""

for s in SENTENCES:
    if len(SEGMENT) >= SEGMENT_LIMIT:
        SEGMENTS.append(SEGMENT)
        SEGMENT = ""
    SEGMENT += s
    
for seg in SEGMENTS:
    print(summarizer(seg, max_length=130, min_length=30, do_sample=False))

