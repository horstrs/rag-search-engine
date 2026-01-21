import re
text = "Text without punctuation!"
sentences = re.split(r"(?<=[.!?])\s+", text)

all_chunks = []
chunk = " ".join(sentences[:5])
all_chunks.append(chunk)
# if chunk_size >= len(sentences):
#     break
blocks = sentences[5 - 0 :]

print(all_chunks)