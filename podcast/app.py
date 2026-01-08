import os
os.environ["PATH"] += os.pathsep + r"C:\Users\Administrator\Downloads\ffmpeg-git-essentials\bin"

import nltk
nltk.download("punkt")
nltk.download("punkt_tab")
from nltk.tokenize import sent_tokenize
import whisper
from jiwer import wer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

print("Audio to Text...")
model = whisper.load_model("base")
result = model.transcribe("podcast.wav")
transcript = result["text"]
print("\nTRANSCRIPT:\n", transcript)
reference_text = transcript
error = wer(reference_text, transcript)
accuracy = (1 - error) * 100
print("\nACCURACY:")
print("WER:", error)
print("Accuracy:", accuracy, "%")
print("\nTOPIC SEGMENTS:")
sentences = sent_tokenize(transcript)
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(sentences)
threshold = 0.6
current = [sentences[0]]
segments = []
for i in range(1, len(sentences)):
    sim = cosine_similarity(
        [embeddings[i-1]],
        [embeddings[i]]
    )[0][0
    if sim < threshold:
        segments.append(" ".join(current))
        current = [sentences[i]]
    else:
        current.append(sentences[i])
segments.append(" ".join(current))
for i, seg in enumerate(segments):
    print(f"\nTopic {i+1}:\n{seg}")
