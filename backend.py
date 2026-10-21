import pandas as pd
import re
import joblib
import faiss
import numpy as np

from lime.lime_text import LimeTextExplainer

from rapidfuzz import fuzz
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from gnews import GNews
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

from datetime import datetime
import os
import csv

# 1. LOAD DATASET
fake = pd.read_csv("Fake.csv")
real = pd.read_csv("True.csv")

#now we createlabel

fake["label"] = 0
real["label"] = 1

#3: Merge datasets

df = pd.concat([fake, real], ignore_index=True)
#4 we shuffle data
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

#after shuffing index num become messy so we need to reset idx 
# Combine title and content

df["content"] = (
    df["title"].fillna("") + " " +
    df["text"].fillna("")
)

# 2. Preprocessing function

def preprocess(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

def extract_keywords(news):

    stop_words = {
        "the","is","a","an","of","to","and","in","on","at",
        "for","with","by","from","has","have","had","was",
        "were","will","would","should","can","could","may",
        "might","this","that","these","those","after","before",
        "today","breaking","news","about","all"
    }

    words = preprocess(news).split()

    keywords = []

    for word in words:

        if len(word) > 3 and word not in stop_words:
            keywords.append(word)

    return " ".join(keywords[:6])

# now we preprocess our dataset--content

df["content"] = df["content"].apply(preprocess)

# it remove all special char,symbol from data

# Features and Labels

X = df["content"]
y = df["label"]

# Train-Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X,y,
    test_size=0.2,
    random_state=42
)
# 4. TF-IDF

vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words='english'
)

X_train_vec = vectorizer.fit_transform(X_train) #Vocabulary learn karega.
X_test_vec = vectorizer.transform(X_test) #Test data ko same vocabulary se vector m convert 

# it convert data into numbers
#Model Training

model = LogisticRegression(max_iter=1000)

model.fit(X_train_vec,y_train)
#it see data again and 
#again to learn a pattern
                          #           LogisticRegression(max_iter=1000)
# we do prediction after model 

# # Prediction

y_pred = model.predict(X_test_vec)

accuracy = accuracy_score(y_test, y_pred)# we predict accuracy of trained model
#by checkin real ans and predicted ans

# Accuracy

print("Accuracy:", accuracy)
# Accuracy: 0.987305122494432

# GNews Initialization

gnews = GNews(
    language="en",
    country="US",
    max_results=5
)

embedding_model = None

def get_embedding_model():

    global embedding_model

    if embedding_model is None:

        print("\nLoading SentenceTransformer...")

        embedding_model = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )

    return embedding_model

# LIME

explainer = LimeTextExplainer(
    class_names=["FAKE", "REAL"]
)

def predict_proba(texts):

    cleaned_texts = [preprocess(text) for text in texts]

    vectors = vectorizer.transform(cleaned_texts)

    return model.predict_proba(vectors)

def generate_reason(words_weights, prediction):

    fake_words = []
    real_words = []

    for word, weight in words_weights:

        if weight < 0:
            fake_words.append(word)

        else:
            real_words.append(word)

    if prediction == "REAL":

        reason = (
            "The Logistic Regression model predicted the news as REAL "
            "because important words such as "
            + ", ".join(real_words[:5])
            + " positively influenced the classification."
        )

    elif prediction == "FAKE":

        reason = (
            "The system predicted the news as FAKE because important words such as "
            + ", ".join(fake_words[:5])
            + " influenced the classification toward the FAKE category."
        )

    else:

        reason = (
            "The final result is UNCERTAIN because the Machine Learning prediction "
            "and retrieved evidence were not fully aligned. "
            "Important influencing words include "
            + ", ".join(real_words[:3] + fake_words[:3])
        )

    return reason

#  RAG KNOWLEDGE BASE


knowledge_base = df["content"] #Entire dataset ko knowledge base banaya.

knowledge_vectors = vectorizer.transform(
    knowledge_base
)

knowledge_vectors_dense = knowledge_vectors.toarray().astype("float32")

dimension = knowledge_vectors_dense.shape[1]

faiss_index = faiss.IndexFlatL2(dimension)

faiss_index.add(knowledge_vectors_dense)

#  RETRIEVAL FUNCTION

# def retrieve_similar_news(news, top_k=3):
def retrieve_similar_news(news, top_k=10):

    clean_news = preprocess(news)

    query_vec = vectorizer.transform([clean_news])

    query_dense = query_vec.toarray().astype("float32")

    distances, indices = faiss_index.search(
        query_dense,
        top_k
    )

    results = []
    unique_evidence = []

    for idx, dist in zip(indices[0], distances[0]):

        similarity = 1 / (1 + dist)
       
        # title = str(df.iloc[idx]["title"])
        
        title = str(df.iloc[idx]["title"]).strip()
        text = str(df.iloc[idx]["text"]).strip()

        if (("http" in title.lower()) or ("www." in title.lower())) and \
           (("http" in text.lower()) or ("www." in text.lower())):
                # evidence = "No readable evidence available in the dataset."
            continue    
              
        elif ("http" in title.lower()) or ("www." in title.lower()):
            evidence = text
            label = df.iloc[idx]["label"]

        else:
            evidence = title
            label = df.iloc[idx]["label"]

        duplicate = False

        for old_evidence in unique_evidence:
            if fuzz.token_set_ratio(evidence, old_evidence) > 85:
                duplicate = True
                break
        
        if duplicate:
             continue
   
        unique_evidence.append(evidence)
      
        results.append({
            "similarity": similarity,
            # "label": df.iloc[idx]["label"],
            "label": label,
            #"title": df.iloc[idx]["title"]
            "evidence": evidence,
             "show_similarity": evidence != "No readable evidence available in the dataset."
            
        })
        if len(results) == 3:
             break
         
    text_count = len(results)

    if text_count == 0:
         results.append({
        "similarity": 0,
        "label": "UNKNOWN",
        "evidence": "No readable evidence available in the dataset.",
        "show_similarity": False
    })
   
    elif text_count < 3:
          results.append({
        "similarity": 0,
        "label": "UNKNOWN",
        "evidence": "No readable evidence available in the dataset.",
        "show_similarity": False
    })
  
    return results

# Compute Article Score

def compute_article_score(news, article_title):

    news = preprocess(news)
    article_title = preprocess(article_title)

    news_words = set(news.split())
    title_words = set(article_title.split())

    # Basic overlap
    overlap = len(news_words & title_words) / (len(news_words) + 1)

    # Phrase boost
    phrase_boost = 0

    for word in news_words:
        if word in article_title:
            phrase_boost += 0.02

    # Important claim words
    important_words = {
        "arrested",
        "custody",
        "embezzlement",
        "scam",
        "fraud",
        "corruption",
        "convicted",
        "murder",
        "attack",
        "resigned",
        "killed",
        "election",
        "won",
        "lost"
    }

    important_score = 0

    for word in important_words:

        if word in news_words:

            if word in title_words:
                important_score += 0.08

            else:
                important_score -= 0.08

    score = overlap + phrase_boost + important_score

    return max(0, min(score, 1))

# Build search query

def build_search_query(news):

    words = preprocess(news).split()

    stop_words = {
        "the","is","are","was","were","a","an","of","to",
        "and","or","in","on","for","with","by","from",
        "today","include","includes","major","news"
    }

    keywords = []

    for word in words:

        if len(word) > 3 and word not in stop_words:
            keywords.append(word)

    return " ".join(keywords[:6])

# Save History

def save_history(news, prediction, confidence):

    history_file = "history.csv"

    new_record = pd.DataFrame([{
        "DateTime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "News": news[:150].replace(",", " ").replace('"', ""),
        "Prediction": prediction,
        "Confidence": confidence
    }])

    if os.path.exists(history_file):

        new_record.to_csv(
        history_file,
        mode="a",
        header=False,
        index=False,
        quoting=csv.QUOTE_ALL
)

    else:

        new_record.to_csv(
        history_file,
        index=False,
        quoting=csv.QUOTE_ALL
)
        
# Analyze news

def analyze_news(news):

    clean_news = preprocess(news)

    # RAG Retrieval
    
    retrieved_docs = retrieve_similar_news(news, top_k=3)

    real_count = 0
    fake_count = 0
    valid_similarities = []
    rag_prediction = "UNKNOWN"
    dataset_strong = False

    for i, doc in enumerate(retrieved_docs, start=1):
    
        title = str(doc["evidence"])

        if doc["label"] == 1:
            label_name = "REAL"
        elif doc["label"] == 0:
            label_name = "FAKE"
        else:
            label_name = "UNKNOWN"
                
        print(f"\nEvidence {i}")
        if title.startswith("http://") or title.startswith("https://"):

            print("Evidence is not found.")

            continue

        print("Title :", title)

        if doc["show_similarity"]:
            print("Similarity :", round(doc["similarity"], 4))
            valid_similarities.append(doc["similarity"])

        print("Label :", label_name)

        if doc["label"] == 1:
            real_count += 1
        elif doc["label"] == 0:
            fake_count += 1

    total = real_count + fake_count

    if total == 0:

        rag_prediction = "UNKNOWN"

        dataset_strong = False

    else:

        rag_real_boost = real_count / total

        rag_fake_boost = fake_count / total

        if real_count > fake_count:
            rag_prediction = "REAL"
        else:
            rag_prediction = "FAKE"
    if len(valid_similarities) > 0:

        best_similarity = max(valid_similarities)

    else:

        best_similarity = 0

    print("Best Similarity :", round(best_similarity, 4))

    # Dataset Strength

    if best_similarity >= 0.80:
        dataset_strong = True
        
    else:
        dataset_strong = False

    # ML Prediction
    
    news_vec = vectorizer.transform([clean_news])

    prediction = model.predict(news_vec)

    probability = model.predict_proba(news_vec)

    ml_prediction = "REAL" if prediction[0] == 1 else "FAKE"

    # GNEWS Implementation
    
    gnews_score = 0.5
    best_article = ""
    best_score = 0
    gnews_articles = []
    verified_news = False      
    verification_reason = "" 

    search_query = build_search_query(news)

    articles = gnews.get_news(search_query)

    embedding_model = get_embedding_model()

    news_embedding = embedding_model.encode([news])

    if len(articles) > 0:

        scores = []

        print(f"Found {min(len(articles),3)} related articles")

        for i, article in enumerate(articles[:3], start=1):

            score = compute_article_score(
                news,
                article["title"]
            )

            # Semantic Similarity

            embedding_model = get_embedding_model()

             # Generate Embeddings

            news_embedding = embedding_model.encode([news])
            
            article_text = ""

            if "title" in article:
                article_text += article["title"] + " "

            if "description" in article:
                article_text += article["description"] + " "

            if "published date" in article:
                article_text += article["published date"]  
            
            article_embedding = embedding_model.encode([article_text])
            
            # Semantic Similarity

            semantic_score = cosine_similarity(
                news_embedding,
                article_embedding
            )[0][0]
            print("Semantic Score :", round(semantic_score, 3))

            # Final GNews Score

            word_score = score
            score = (word_score * 0.3) + (semantic_score * 0.7)
            print("\n----- Score Breakdown -----")
            print("Word Score     :", round(word_score, 3))
            print("Semantic Score :", round(semantic_score, 3))
            print("Final Score    :", round(score, 3))

            if score > best_score:
                best_score = score
                best_article = article["title"]
            
            scores.append(score)

            print(f"\nArticle {i}")
            print("Title :", article["title"])
            print("Score :", round(score, 3))

            if "url" in article:
              print("URL :", article["url"])

            gnews_articles.append({
                "title": article["title"],
                "url": article.get("url", ""),
                "score": float(score)
            })

        gnews_score = best_score

        gnews_verdict = "FOUND"

        # GNews Strength

        if gnews_score >= 0.40:
            gnews_strong = True
        else:
            gnews_strong = False

        print("GNews Score :", round(gnews_score, 2))
        print("GNews Strong :", gnews_strong)
        print("ML Prediction :", ml_prediction)
        print("RAG Prediction :", rag_prediction)
        print("GNews Score :", round(gnews_score, 2))
        print("Best Similarity :", round(best_similarity, 2))

    else:

        print("No related news found")

        gnews_score = 0.3

        gnews_verdict = "NOT_FOUND"
        gnews_strong = False
    
    # Hybrid Prediction

    # Case 0 : No valid RAG evidence
    if rag_prediction == "UNKNOWN":

        if gnews_strong:

            decision = "GNews Verified (RAG unavailable)"
            print("Decision :", decision)
            final_prediction = "REAL"

        else:

            decision = "ML Only"
            print("Decision :", decision)
            print("Reason : No valid evidence was retrieved from the dataset, so RAG was ignored.")
            final_prediction = ml_prediction

    # Case 1 : Dataset Strong + GNews Strong
    elif best_similarity >= 0.80 and gnews_strong:

        if ml_prediction == rag_prediction:

            decision = "ML + RAG + GNews"
            print("Decision :", decision)
            print("Reason : Dataset similarity is high (>=0.80), GNews strongly matched the claim, and ML agrees with retrieved evidence.")

            final_prediction = ml_prediction

        else:

            decision = "UNCERTAIN"
            print("Decision :", decision)
            print("Reason : Dataset and GNews are strong, but ML prediction conflicts with retrieved evidence.")

            final_prediction = "UNCERTAIN"

    # Case 2 : Dataset Strong
    elif best_similarity >= 0.80:

        decision = "ML + RAG"
        print("Decision :", decision)
        print("Reason : Retrieved dataset contains highly similar news (>=0.80), so retrieval evidence is trusted more than GNews.")

        final_prediction = rag_prediction

    # Case 3 : Dataset Weak + GNews Strong
    elif gnews_strong:

        decision = "GNews Verified"
        print("Decision :", decision)
        print("Reason : Dataset does not contain similar news, but Google News found a strong semantic match.")

        final_prediction = "REAL"

    # Case 4 : Dataset Weak + GNews Weak
    else:

        decision = "ML Only"
        print("Decision :", decision)
        print("Reason : Neither dataset retrieval nor Google News produced strong evidence, so prediction relies only on the trained ML model.")

        final_prediction = ml_prediction

    # LIME

    explanation = explainer.explain_instance(
        news,
        predict_proba,
        num_features=10
    )

    lime_words = explanation.as_list()

    reason = generate_reason(
    lime_words,
    ml_prediction
    )

    confidence = round(max(probability[0]) * 100, 2)

    save_history(
        news,
        final_prediction,
        confidence
    )

    return {
        "ml_prediction": ml_prediction,
        "rag_prediction": rag_prediction,
        "final_prediction": final_prediction,
        "confidence": confidence,
        "reason": reason,
        "important_words": lime_words,
        "evidence": retrieved_docs,
        "best_similarity": best_similarity,
        "dataset_strong": dataset_strong,
        "gnews_score": gnews_score,
        "gnews_strong": gnews_strong,
        "best_article": best_article,
        "hybrid_decision": decision,
        "gnews_prediction": gnews_verdict,
        "gnews_articles": gnews_articles,
    }

# call Analyze function
if __name__ == "__main__":

    news = input("\nEnter News: ")

    result = analyze_news(news)

    print("\n========== MODEL RESULT ==========")
    print("ML Prediction :", result["ml_prediction"])
    print("Confidence :", result["confidence"])

    print("\n========== AI REASONING ==========")
    print(result["reason"])

    print("\n========== LIME EXPLANATION ==========")

    for word, weight in result["important_words"]:
        print(word, ":", round(float(weight), 4))

    print("\n========== RAG RESULT ==========")
    print("RAG Prediction :", result["rag_prediction"])
    print("Best Similarity :", result["best_similarity"])
    print("Dataset Strong :", result["dataset_strong"])

    print("\n========== GNEWS ==========")
    print("GNews Score :", result["gnews_score"])
    print("GNews Strong :", result["gnews_strong"])

    print("\n========== FINAL ==========")
    print("Final Prediction :", result["final_prediction"])

    # print(result)