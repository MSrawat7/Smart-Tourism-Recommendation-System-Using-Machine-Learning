import random
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

# Random seed
random.seed(42)
np.random.seed(42)

# Categories & places
categories = ["beach", "forest", "mountain", "city", "river"]
uttarakhand_places = [
    "Dehradun", "Mussoorie", "Nainital", "Rishikesh", "Haridwar",
    "Auli", "Jim Corbett", "Almora", "Kausani", "Bageshwar",
    "Chopta", "Valley of Flowers", "Lansdowne", "Mukteshwar"
]

# Best months
category_best_months = {
    "beach": [3,4,5,6],
    "forest": [3,4,5,9,10],
    "mountain": [4,5,6,9,10],
    "city": list(range(1,13)),
    "river": [3,4,5,6,9]
}

# Activities
activities_by_cat = {
    "beach": ["sunbathing","river-side relax","picnic","photography"],
    "forest": ["hiking","wildlife","birdwatching","trekking"],
    "mountain": ["hiking","mountain-climbing","scenic drives","photography"],
    "city": ["museums","food-tour","shopping","heritage walk"],
    "river": ["rafting","kayaking","swimming","fishing"]
}

# Helper for random place names
def random_place_name(cat, i):
    return f"{cat.capitalize()} Spot {i}"

# Generate dataset
rows = []
n = 150
for i in range(n):
    cat = random.choice(categories)
    name = random_place_name(cat, i+1)
    city = random.choice(uttarakhand_places)
    rating = round(random.uniform(3.0, 5.0), 1)
    avg_cost = int(random.uniform(20, 200))
    best_months = sorted(list(set(
        random.sample(category_best_months[cat],
                      k=max(1, int(len(category_best_months[cat]) * random.uniform(0.5, 1.0))))
    )))
    activities = random.sample(activities_by_cat[cat], k=2)
    desc = f"{name} in {city}: a {cat} destination great for {activities[0]} and {activities[1]}. Best months: {', '.join([datetime(2000,m,1).strftime('%b') for m in best_months])}."
    lat = round(random.uniform(28.5, 31.5), 3)
    lon = round(random.uniform(77.5, 81), 3)
    rows.append({
        "id": i+1,
        "name": name,
        "category": cat,
        "city": city,
        "rating": rating,
        "avg_cost_inr": avg_cost,
        "best_months": ",".join(str(m) for m in best_months),
        "activities": ";".join(activities),
        "description": desc,
        "lat": lat,
        "lon": lon
    })

df = pd.DataFrame(rows)

# Generate user_score
def generate_user_score(row):
    score = 0.4 * row['rating'] + (200 - row['avg_cost_inr'])/200 * 0.3 + random.uniform(0,1) * 0.3
    return min(round(score*5,1), 5.0)

df['user_score'] = df.apply(generate_user_score, axis=1)

# Encoding
ohe = OneHotEncoder(sparse_output=False)
category_encoded = ohe.fit_transform(df[['category']])
category_df = pd.DataFrame(category_encoded, columns=ohe.get_feature_names_out(['category']))

scaler = StandardScaler()
numeric_features = scaler.fit_transform(df[['rating','avg_cost_inr','lat','lon']])
numeric_df = pd.DataFrame(numeric_features, columns=['rating','avg_cost_inr','lat','lon'])

X = pd.concat([category_df, numeric_df], axis=1)
y = df['user_score']

# Train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("Test RMSE:", round(rmse,3))

# TF-IDF
tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1,2))
tfidf.fit(df['description'])

# Save models
joblib.dump(model, "uttarakhand_rf_model.joblib")
joblib.dump(ohe, "category_encoder.joblib")
joblib.dump(scaler, "scaler.joblib")
joblib.dump(tfidf, "tfidf_vectorizer_uttarakhand.joblib")
df.to_csv("uttarakhand_tourism_dataset.csv", index=False)

print("✅ Dataset and ML model saved successfully.")
