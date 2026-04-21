# =========================================
# IMPORTS
# =========================================
import pandas as pd
import joblib
import re
from supabase import create_client

# =========================================
# CONFIG
# =========================================
SUPABASE_URL = 'https://eyljduxsumaonuxocfhk.supabase.co'
SUPABASE_KEY = 'sb_publishable_NLe_sC6FF5HRSvs5XCpu7w_1xHf-lB4'

MODEL_PATH = "category_model.pkl"
OUTPUT_FILE = "real_world_predictions.csv"

# =========================================
# CONNECT
# =========================================
def connect_db():
    return create_client(SUPABASE_URL, SUPABASE_KEY)

# =========================================
# FETCH DATA
# =========================================
def fetch_data(supabase):
    products = supabase.table("products").select("*").execute().data
    ingredients = supabase.table("ingredients").select("*").execute().data
    product_ingredients = supabase.table("product_ingredients").select("*").execute().data

    return (
        pd.DataFrame(products),
        pd.DataFrame(ingredients),
        pd.DataFrame(product_ingredients)
    )

# =========================================
# MERGE DATA
# =========================================
def merge_data(df_products, df_ingredients, df_pi):
    df_merge = df_pi.merge(
        df_ingredients,
        left_on="ingredient_id",
        right_on="id",
        how="left"
    )

    df_ing = df_merge.groupby("product_id")["ingredient_name"] \
        .apply(lambda x: " ".join(x.dropna())) \
        .reset_index()

    df_grouped = df_products.merge(
        df_ing,
        left_on="id",
        right_on="product_id",
        how="left"
    )

    df_grouped.rename(columns={
        "id": "product_id",
        "ingredient_name": "ingredients"
    }, inplace=True)

    df_grouped["ingredients"] = df_grouped["ingredients"].fillna("")

    return df_grouped

# =========================================
# PREPROCESSING
# =========================================
def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def remove_noise(text):
    return text.replace("toner", "") if isinstance(text, str) else ""

def add_keyword_features(text):
    text = text.lower()
    features = []

    if any(x in text for x in ["spf", "sun", "uv"]):
        features.append("KEY_SUNSCREEN")
    if any(x in text for x in ["cleanser", "face wash", "cleansing"]):
        features.append("KEY_CLEANSER")
    if any(x in text for x in ["serum", "retinol", "niacinamide"]):
        features.append("KEY_SERUM")
    if any(x in text for x in ["cream", "moisturizer", "lotion"]):
        features.append("KEY_MOISTURIZER")

    return " ".join(features)

# =========================================
# RULE BASED
# =========================================
def rule_based(text):
    t = text.lower()

    if "spf" in t or "sunscreen" in t:
        return "Sunscreen"

    if "cleanser" in t or "face wash" in t or "cleansing" in t:
        return "Cleanser"

    if "serum" in t:
        return "Serum"

    if any(x in t for x in ["retinol", "niacinamide", "vitamin c", "salicylic"]):
        return "Serum"

    if any(x in t for x in ["cream", "moisturizer", "lotion"]):
        return "Moisturizer"

    return None

# =========================================
# MAIN PIPELINE
# =========================================
def run_pipeline():
    print("✅ Connecting to Supabase...")
    supabase = connect_db()

    print("✅ Fetching data...")
    df_products, df_ingredients, df_pi = fetch_data(supabase)

    print("✅ Merging data...")
    df = merge_data(df_products, df_ingredients, df_pi)

    print("✅ Loading model...")
    model = joblib.load(MODEL_PATH)

    print("✅ Preprocessing...")
    df['product_name'] = df['product_name'].apply(clean_text)
    df['ingredients'] = df['ingredients'].apply(clean_text)

    df['text'] = (
        df['product_name'].apply(remove_noise) + " " +
        df['ingredients'].apply(remove_noise) + " " +
        df['product_name'].apply(add_keyword_features)
    )

    print("✅ Predicting...")
    predictions = []

    for text in df['text']:
        r = rule_based(text)
        if r:
            predictions.append(r)
        else:
            predictions.append(model.predict([text])[0])

    df['predicted_category'] = predictions

    print("✅ Saving results...")
    df[['product_id', 'product_name', 'ingredients', 'predicted_category']] \
        .to_csv(OUTPUT_FILE, index=False)

    print(f"🎉 Done! Saved to {OUTPUT_FILE}")

    return df

# =========================================
# RUN
# =========================================
if __name__ == "__main__":
    run_pipeline()