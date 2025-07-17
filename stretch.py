from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def main():
    model = SentenceTransformer('all-MiniLM-L6-v2')
        
    sentence_1 = input("Enter sentence 1: ")
    sentence_2 = input("Enter sentence 2: ")
    sentence_3 = input("Enter sentence 3: ")
    
    emb_1 = model.encode(sentence_1)
    emb_2 = model.encode(sentence_2)
    emb_3 = model.encode(sentence_3)
    
    sim_12 = cosine_similarity([emb_1], [emb_2])
    sim_13 = cosine_similarity([emb_1], [emb_3])
    sim_23 = cosine_similarity([emb_2], [emb_3])
    
    sim_score_12 = sim_12[0][0]
    sim_score_13 = sim_13[0][0]
    sim_score_23 = sim_23[0][0]

    similarities = {
        "Sentence 1 v. 2": sim_score_12,
        "Sentence 1 v. 3": sim_score_13,
        "Sentence 2 v. 3": sim_score_23
    }
    
    similarities = sorted(similarities.items(), key=lambda pair: pair[1], reverse=True)
    
    print("\nSorted scores from highest to lowest:")
    for label, score in similarities:
        print(f"{label}: {score * 100:.2f}%")


if __name__ == "__main__":
    main()
