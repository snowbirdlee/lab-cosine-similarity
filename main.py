from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def main():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    sentence1 = input("Sentence 1: ")
    sentence2 = input("Sentence 2: ")
    
    embedding_vector_1 = model.encode(sentence1)
    embedding_vector_2 = model.encode(sentence2)
    
    similarity = cosine_similarity([embedding_vector_1], [embedding_vector_2])
    similarity_score = similarity[0][0]
    print(f"Cosine similarity: {similarity_score*100:.2f}%")


if __name__ == "__main__":
    main()
