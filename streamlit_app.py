import streamlit as st
import pandas as pd
import joblib
from datetime import datetime
from sklearn.metrics.pairwise import linear_kernel

# ✅ Must be the first Streamlit command
st.set_page_config(page_title="Smart Tourism Recommender", layout="centered")

@st.cache_data
def load_data():
    df = pd.read_csv("uttarakhand_tourism_dataset.csv")
    model = joblib.load("uttarakhand_rf_model.joblib")
    ohe = joblib.load("category_encoder.joblib")
    scaler = joblib.load("scaler.joblib")
    tfidf = joblib.load("tfidf_vectorizer_uttarakhand.joblib")
    return df, model, ohe, scaler, tfidf

df, model, ohe, scaler, tfidf = load_data()

st.title("🌄 Smart Tourism Recommendation System")

category = st.selectbox("Preferred category", options=["any"] + sorted(df['category'].unique().tolist()))
month = st.selectbox("Travel month", options=["Any"] + [datetime(2000,m,1).strftime("%B") for m in range(1,13)])
min_rating = st.slider("Minimum rating", 3.0, 5.0, 4.0)
max_cost = st.number_input("Max avg daily cost (INR, 0 = no limit)", min_value=0, value=0)

def get_ml_predictions(df, model, ohe, scaler):
    cat_encoded = ohe.transform(df[['category']])
    cat_df = pd.DataFrame(cat_encoded, columns=ohe.get_feature_names_out(['category']))
    num_scaled = scaler.transform(df[['rating','avg_cost_inr','lat','lon']])
    num_df = pd.DataFrame(num_scaled, columns=['rating','avg_cost_inr','lat','lon'])
    X = pd.concat([cat_df, num_df], axis=1)
    return model.predict(X)

if st.button("Recommend"):
    df2 = df.copy()
    df2['ml_score'] = get_ml_predictions(df2, model, ohe, scaler)

    if category.lower() != "any":
        df2 = df2[df2['category'] == category]
    if month != "Any":
        m = datetime.strptime(month, "%B").month
        df2 = df2[df2['best_months'].str.contains(str(m))]
    df2 = df2[df2['rating'] >= min_rating]
    if max_cost > 0:
        df2 = df2[df2['avg_cost_inr'] <= max_cost]

    res = df2.sort_values("ml_score", ascending=False).head(10)

    if res.empty:
        st.warning("No matches found — try changing filters.")
    else:
        for _, r in res.iterrows():
            st.subheader(f"{r['name']} — {r['city']} ({r['category']})")
            st.write(f"Predicted Score: {round(r['ml_score'],2)} ⭐")
            st.write(f"Rating: {r['rating']}  —  Avg cost: ₹{r['avg_cost_inr']}/day")
            bm = ", ".join([datetime(2000,int(m),1).strftime("%b") for m in str(r['best_months']).split(",") if m])
            st.write(f"Best months: {bm}")
            st.write("Activities:", r['activities'].replace(";", ", "))
            st.write(r['description'])
            st.write("---")

        choice = st.selectbox("See similar to:", options=res['name'].tolist())
        if choice:
            idx = int(df[df['name']==choice].index[0])
            tfidf_matrix = tfidf.transform(df['description'].values.astype('U'))
            cos = linear_kernel(tfidf_matrix[idx:idx+1], tfidf_matrix).flatten()
            sim_idx = cos.argsort()[-6:-1][::-1]
            st.write("Similar places:")
            for i in sim_idx:
                rr = df.iloc[i]
                st.write(f"- {rr['name']} ({rr['category']}) — Rating {rr['rating']}, Avg cost ₹{rr['avg_cost_inr']}")
