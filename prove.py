import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix

# Load full dataset
full_rating = pd.read_csv('ratings.csv')
full_rating = full_rating[:511]  # Optional trimming

# Filter out users and foods with too few ratings
min_user_ratings = 3
min_food_ratings = 5

user_counts = full_rating['User_ID'].value_counts()
food_counts = full_rating['Food_ID'].value_counts()

filtered_rating = full_rating[
    full_rating['User_ID'].isin(user_counts[user_counts >= min_user_ratings].index) &
    full_rating['Food_ID'].isin(food_counts[food_counts >= min_food_ratings].index)
]

# Split train/test
train_data, test_data = train_test_split(filtered_rating, test_size=0.2, random_state=42)

# Build pivot table from training data
rating_matrix = train_data.pivot_table(index='Food_ID', columns='User_ID', values='Rating').fillna(0)
csr_rating_matrix = csr_matrix(rating_matrix.values)

# Train item-based NearestNeighbors
recommender = NearestNeighbors(metric='cosine')
recommender.fit(csr_rating_matrix)

def Get_ColdStart_Recommendations(top_k=10):
    popular = (
        train_data.groupby('Food_ID')['Rating']
        .count()
        .sort_values(ascending=False)
        .head(top_k)
        .index
        .tolist()
    )
    return pd.DataFrame({'Food_ID': popular})


# Recommend top 10 items for a user using top-N rated seeds
def Get_Recommendations(user_id, top_n_seeds=3, top_k=10):
    user_data = train_data[train_data['User_ID'] == user_id]
    if user_data.empty:
        return Get_ColdStart_Recommendations(top_k=10)

    top_rated_items = user_data.sort_values(by='Rating', ascending=False).head(top_n_seeds)
    recommended_set = set()

    for _, row in top_rated_items.iterrows():
        food_id = row['Food_ID']
        try:
            food_index = np.where(rating_matrix.index == food_id)[0][0]
        except IndexError:
            continue

        distances, indices = recommender.kneighbors(rating_matrix.iloc[food_index].values.reshape(1, -1), n_neighbors=top_k + 1)
        similar_food_ids = rating_matrix.iloc[indices[0][1:]].index.tolist()
        recommended_set.update(similar_food_ids)

    result_df = pd.DataFrame({'Food_ID': list(recommended_set)[:top_k]})
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
    print(f'Improved Precision@10: {precision:.4f}')
    return precision

# Run evaluation
if __name__ == "__main__":
    evaluate_precision_at_10()
