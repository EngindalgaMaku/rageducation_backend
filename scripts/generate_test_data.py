import sys
import os
from pathlib import Path
import random
import json
from datetime import datetime

# Add project root to the Python path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.analytics.database import get_experiment_db
from src.config import DB_PATH

def generate_test_data():
    """
    Populates the database with sample data for the active learning loop.
    """
    print("Starting to generate test data...")
    db = get_experiment_db(DB_PATH)
    
    # --- 1. Define Sample Data ---
    users = ["student_01", "student_02", "student_03"]
    rag_params_list = [
        {"chunk_size": 1000, "top_k": 3, "model": "qwen2.5:14b"},
        {"chunk_size": 512, "top_k": 5, "model": "mistral:7b"},
        {"chunk_size": 1000, "top_k": 4, "model": "qwen2.5:14b"},
    ]
    queries_and_responses = [
        ("Python'da list comprehension nedir?", "List comprehension, Python'da listeleri daha okunaklı ve kısa bir şekilde oluşturmak için kullanılan bir sözdizimidir."),
        ("Lojistik regresyon ne işe yarar?", "Lojistik regresyon, bir veya daha fazla bağımsız değişkene dayanarak kategorik bir sonucun olasılığını modellemek için kullanılan bir istatistiksel yöntemdir."),
        ("RAG ve fine-tuning arasındaki fark nedir?", "RAG, modele dışarıdan bilgi sağlayarak cevap üretmesini sağlarken, fine-tuning modelin kendi iç ağırlıklarını belirli bir veri seti üzerinde günceller."),
        ("SQL'de JOIN türleri nelerdir?", "INNER JOIN, LEFT JOIN, RIGHT JOIN ve FULL OUTER JOIN gibi temel birleştirme türleri vardır."),
    ]
    feedback_types = ["rating", "correction", "categorization"]
    feedback_categories = ["yanlış_bilgi", "eksik_cevap", "alakasız_kaynak", "harika_cevap"]

    # --- 2. Create RAG Configurations ---
    print("\nCreating RAG configurations...")
    config_hashes = []
    for params in rag_params_list:
        config_hash = db.add_or_get_rag_configuration(params)
        config_hashes.append(config_hash)
        print(f"  - RAG Config Hash: {config_hash[:10]}...")

    # --- 3. Generate Interactions and Feedback ---
    print("\nGenerating interactions and feedback...")
    for i in range(15): # Create 15 interactions
        user = random.choice(users)
        config_hash = random.choice(config_hashes)
        query, response = random.choice(queries_and_responses)
        
        interaction_id = db.add_interaction(
            user_id=user,
            query=f"{query} (rand={random.randint(1,100)})", # Make query unique
            response=response,
            retrieved_context=[{"source": "doc1.pdf", "page": i}],
            rag_config_hash=config_hash,
            uncertainty_score=random.random(),
            feedback_requested=random.choice([True, False]),
            processing_time_ms=random.uniform(500, 3000)
        )
        print(f"  - Created interaction {interaction_id} for user {user}.")

        # Add feedback to about 70% of interactions
        if random.random() < 0.7:
            rating = random.randint(1, 5)
            feedback_id = db.add_feedback(
                interaction_id=interaction_id,
                user_id=user,
                feedback_type="rating",
                rating=rating,
                feedback_category=random.choice(feedback_categories) if rating < 4 else "harika_cevap",
                notes="Bu cevap oldukça faydalıydı." if rating >= 4 else "Cevap daha iyi olabilirdi."
            )
            print(f"    - Added feedback {feedback_id} with rating {rating} for interaction {interaction_id}.")

    # --- 4. Verify Data ---
    print("\nVerifying generated data...")
    with db.get_connection() as conn:
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM interactions")
        interaction_count = cursor.fetchone()
        print(f"  - Total interactions: {interaction_count}")

        cursor.execute("SELECT COUNT(*) FROM feedback")
        feedback_count = cursor.fetchone()
        print(f"  - Total feedback entries: {feedback_count}")

        cursor.execute("SELECT COUNT(*) FROM student_profiles")
        profile_count = cursor.fetchone()
        print(f"  - Total student profiles: {profile_count}")
        
        cursor.execute("SELECT user_id, total_queries, avg_feedback_score FROM student_profiles LIMIT 3")
        profiles = cursor.fetchall()
        for profile in profiles:
            print(f"    - Profile for {profile['user_id']}: Queries={profile['total_queries']}, Avg Score={profile['avg_feedback_score']:.2f}")

    print("\nTest data generation completed successfully!")

if __name__ == "__main__":
    generate_test_data()