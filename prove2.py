import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix

# Load and preprocess data
full_rating = pd.read_csv('ratings.csv')
full_rating = full_rating[:511]

# Filter sparse users and items
min_user_ratings = 3
min_food_ratings = 5

user_counts = full_rating['User_ID'].value_counts()
food_counts = full_rating['Food_ID'].value_counts()

filtered_rating = full_rating[
    full_rating['User_ID'].isin(user_counts[user_counts >= min_user_ratings].index) &
    full_rating['Food_ID'].isin(food_counts[food_counts >= min_food_ratings].index)
]

# Train-test split
train_data, test_data = train_test_split(filtered_rating, test_size=0.2, random_state=42)

# Build item-user rating matrix
rating_matrix = train_data.pivot_table(index='Food_ID', columns='User_ID', values='Rating').fillna(0)
csr_rating_matrix = csr_matrix(rating_matrix.values)

# Fit Nearest Neighbors model
recommender = NearestNeighbors(metric='cosine')
recommender.fit(csr_rating_matrix)

# Improved Recommendation Function
def Get_Recommendations(user_id, top_n_seeds=5, top_k=10):
    user_data = train_data[train_data['User_ID'] == user_id]
    if user_data.empty:
        return pd.DataFrame({'Food_ID': []})

    # Get top-N rated items
    top_rated_items = user_data.sort_values(by='Rating', ascending=False).head(top_n_seeds)

    similarity_scores = {}
    for _, row in top_rated_items.iterrows():
        food_id = row['Food_ID']
        try:
            food_index = np.where(rating_matrix.index == food_id)[0][0]
        except IndexError:
            continue

        distances, indices = recommender.kneighbors(rating_matrix.iloc[food_index].values.reshape(1, -1), n_neighbors=11)

        for dist, idx in zip(distances[0][1:], indices[0][1:]):  # skip self
            neighbor_id = rating_matrix.index[idx]
            similarity_scores[neighbor_id] = similarity_scores.get(neighbor_id, []) + [1 - dist]

    # Average the similarity scores
    avg_scores = {
        food_id: np.mean(scores)
        for food_id, scores in similarity_scores.items()
    }

    # Exclude already-rated foods
    rated_food_ids = set(user_data['Food_ID'])
    final_candidates = {
        food_id: score
        for food_id, score in avg_scores.items()
        if food_id not in rated_food_ids
    }

    # Get top-K recommendations
    top_recommended = sorted(final_candidates.items(), key=lambda x: x[1], reverse=True)[:top_k]
    result_df = pd.DataFrame({'Food_ID': [fid for fid, _ in top_recommended]})
    return result_df

# Evaluation: Precision@10
def evaluate_precision_at_10():
    users = test_data['User_ID'].unique()
    hit = 0
    total_recommendations = 0

    for user_id in users:
        actual_items = set(test_data[test_data['User_ID'] == user_id]['Food_ID'])
        recommended_df = Get_Recommendations(user_id)
        recommended_items = set(recommended_df['Food_ID'].tolist())

        if not recommended_items:
            continue

        hit += len(actual_items & recommended_items)
        total_recommendations += len(recommended_items)

    if total_recommendations == 0:
        print("No recommendations made.")
        return 0.0

    precision = hit / total_recommendations
    print(f'🚀 Enhanced Precision@10: {precision:.4f}')
    return precision

# Run evaluation
if __name__ == "__main__":
    evaluate_precision_at_10()
