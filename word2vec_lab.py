import torch


# =========================================================
# 1. GPU Setup
# =========================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# =========================================================
# 2. Prepare a Small Corpus
# =========================================================

sentences = [
    "king is a royal man",
    "queen is a royal woman",
    "king rules the kingdom",
    "queen rules the kingdom",
    "king lives in the palace",
    "queen lives in the palace",

    "prince is a young man",
    "princess is a young woman",
    "prince lives in the palace",
    "princess lives in the palace",

    "man is a male person",
    "woman is a female person",
    "boy is a young male",
    "girl is a young female",

    "dog is an animal",
    "cat is an animal",
    "dog is a pet",
    "cat is a pet",
    "puppy is a young dog",
    "kitten is a young cat",

    "apple is a fruit",
    "banana is a fruit",
    "orange is a fruit",
    "apple is sweet",
    "banana is sweet",
    "orange is sweet",

    "car is a vehicle",
    "bus is a vehicle",
    "truck is a vehicle",
    "car moves on road",
    "bus moves on road",
    "truck moves on road",
]


# =========================================================
# 3. Tokenize
# =========================================================

corpus = []

for sentence in sentences:
    tokens = sentence.lower().split()
    corpus.append(tokens)

print("\nFirst 5 tokenized sentences:")

for tokens in corpus[:5]:
    print(tokens)


# =========================================================
# 4. Build Vocabulary
# =========================================================

words = []

for sentence in corpus:
    words.extend(sentence)

vocab = sorted(set(words))

word_to_id = {
    word: i
    for i, word in enumerate(vocab)
}

id_to_word = {
    i: word
    for word, i in word_to_id.items()
}

vocab_size = len(vocab)

print("\nVocabulary size:", vocab_size)
print(word_to_id)

# =========================================================
# 5. Generate Skip-gram Training Pairs
# =========================================================

def create_skipgram_pairs(corpus, window=2):
    pairs = []

    for sentence in corpus:
        for i in range(len(sentence)):
            center_word = sentence[i]

            start = max(0, i - window)
            end = min(
                len(sentence),
                i + window + 1
            )

            for j in range(start, end):
                if i == j:
                    continue

                context_word = sentence[j]

                center_id = word_to_id[
                    center_word
                ]
                context_id = word_to_id[
                    context_word
                ]

                pairs.append(
                    (center_id, context_id)
                )

    return pairs


# 실제 Skip-gram pair 생성
pairs = create_skipgram_pairs(
    corpus,
    window=2
)

print("\nNo of training pairs:", len(pairs))

# 처음 20개 pair 확인
for center_id, context_id in pairs[:20]:
    print(
        id_to_word[center_id],
        "->",
        id_to_word[context_id]
    )


# =========================================================
# 6. Convert Training Data to Tensors
# =========================================================

center_words = torch.tensor(
    [pair[0] for pair in pairs],
    dtype=torch.long
)

context_words = torch.tensor(
    [pair[1] for pair in pairs],
    dtype=torch.long
)

print("\nCenter words shape:", center_words.shape)
print("Context words shape:", context_words.shape)


# =========================================================
# 7. Use DataLoader
# =========================================================

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

dataset = TensorDataset(
    center_words,
    context_words
)

loader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=True
)


# =========================================================
# 8. Build the Skip-gram Neural Network
# =========================================================

import torch.nn as nn


class SkipGramModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim
        )

        self.output = nn.Linear(
            embedding_dim,
            vocab_size
        )

    def forward(self, center_words):
        embedded = self.embedding(
            center_words
        )

        logits = self.output(
            embedded
        )

        return logits

# =========================================================
# 9. Create the Skip-gram Model
# =========================================================

embedding_dim = 50

model = SkipGramModel(
    vocab_size,
    embedding_dim
)

model = model.to(device)

print("\nModel:")
print(model)


# =========================================================
# 10. Loss & Optimizer
# =========================================================

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.01
)


# =========================================================
# 11. Train Word2Vec
# =========================================================

epochs = 200

model.train()

for epoch in range(epochs):

    total_loss = 0

    for centers, contexts in loader:

        # 데이터를 GPU 또는 CPU로 이동
        centers = centers.to(device)
        contexts = contexts.to(device)

        # 이전 step의 gradient 초기화
        optimizer.zero_grad()

        # Forward
        logits = model(centers)

        # Loss 계산
        loss = criterion(
            logits,
            contexts
        )

        # Backpropagation
        loss.backward()

        # Weight 업데이트
        optimizer.step()

        total_loss += loss.item()

    # 20 epoch마다 loss 출력
    if (epoch + 1) % 20 == 0:

        average_loss = (
            total_loss / len(loader)
        )

        print(
            f"Epoch {epoch+1:3d}, "
            f"Loss: {average_loss:.4f}"
        )


# =========================================================
# 12. Extract Learned Embeddings
# =========================================================

embeddings = model.embedding.weight

print("\nEmbedding shape:", embeddings.shape)
print("Embedding device:", embeddings.device)

king_id = word_to_id["king"]

king_vector = embeddings[king_id]

print("\nKing vector:")
print(king_vector)

import torch.nn.functional as F


def similarity(word1, word2):

    id1 = word_to_id[word1]
    id2 = word_to_id[word2]

    v1 = embeddings[id1]
    v2 = embeddings[id2]

    score = F.cosine_similarity(
        v1.unsqueeze(0),
        v2.unsqueeze(0)
    )

    return score.item()

print(
    "king - queen:",
    similarity("king", "queen")
)

print(
    "dog - cat:",
    similarity("dog", "cat")
)

print(
    "apple - banana:",
    similarity("apple", "banana")
)

print(
    "dog - car:",
    similarity("dog", "car")
)

normalized_embeddings = F.normalize(
    embeddings,
    p=2,
    dim=1
)

# =========================================================
# 15. Nearest Neighbor Search
# =========================================================

def most_similar(word, top_k=5):
    word_id = word_to_id[word]

    # 찾고 싶은 단어의 정규화된 embedding
    target = normalized_embeddings[word_id]

    # 모든 단어와 한꺼번에 cosine similarity 계산
    # 이미 모든 벡터의 길이가 1이므로
    # dot product = cosine similarity
    similarities = torch.matmul(
        normalized_embeddings,
        target
    )

    # 자기 자신도 가장 높은 점수로 나오므로
    # top_k + 1개를 먼저 가져옴
    values, indices = torch.topk(
        similarities,
        k=top_k + 1
    )

    results = []

    for score, idx in zip(
        values.detach().cpu().tolist(),
        indices.detach().cpu().tolist()
    ):
        candidate = id_to_word[idx]

        # 자기 자신은 제외
        if candidate != word:
            results.append(
                (candidate, score)
            )

        if len(results) == top_k:
            break

    return results


print("\nMost similar words:")

print(
    "king:",
    most_similar("king", top_k=3)
)

print(
    "dog:",
    most_similar("dog", top_k=3)
)

print(
    "apple:",
    most_similar("apple", top_k=3)
)


# =========================================================
# 16. Prepare Embeddings for PCA
# =========================================================

# GPU Tensor -> CPU NumPy array
embedding_cpu = (
    model.embedding.weight
    .detach()
    .cpu()
    .numpy()
)


# =========================================================
# 17. PCA - Reduce 50D Embeddings to 2D
# =========================================================

from sklearn.decomposition import PCA

pca = PCA(n_components=2)

# 전체 vocabulary의 50차원 벡터를
# 2차원으로 축소
vectors_2d = pca.fit_transform(
    embedding_cpu
)

print(
    "\nPCA result shape:",
    vectors_2d.shape
)


# =========================================================
# 18. Similarity Visualization
# =========================================================

import matplotlib.pyplot as plt


words_to_plot = [
    "king", "queen",
    "prince", "princess",
    "man", "woman",
    "dog", "cat",
    "puppy", "kitten",
    "apple", "banana", "orange",
    "car", "bus", "truck"
]


plt.figure(figsize=(10, 8))

for word in words_to_plot:
    # 해당 단어의 vocabulary ID를 이용해서
    # 올바른 PCA 좌표를 가져옴
    idx = word_to_id[word]

    x = vectors_2d[idx, 0]
    y = vectors_2d[idx, 1]

    plt.scatter(x, y)

    plt.text(
        x + 0.02,
        y + 0.02,
        word,
        fontsize=11
    )

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("Word2Vec Word Similarity")
plt.grid(True)

plt.show()