import streamlit as st
import pandas as pd
import plotly.express as px
import json
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from nltk.stem import WordNetLemmatizer
import joblib
import re
import nltk

nltk.download('wordnet', quiet=True)

st.set_page_config(page_title="Finance App", layout="wide")

category_file = "categories.json"
model_file = "category_model.pkl"
budget_file = "budgets.json"

if "categories" not in st.session_state:
    st.session_state.categories = {"Uncategorized": []}

if "debit_df" not in st.session_state:
    st.session_state.debit_df = None

if "budgets" not in st.session_state:
    st.session_state.budgets = {}

if os.path.exists(category_file):
    with open(category_file, "r") as f:
        st.session_state.categories = json.load(f)

if os.path.exists(budget_file):
    with open(budget_file, "r") as f:
        st.session_state.budgets = json.load(f)

lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    tokens = text.split()
    return ' '.join([lemmatizer.lemmatize(word) for word in tokens])

def load_transactions(file):
    try:
        df = pd.read_csv(file)
        df.columns = [col.strip() for col in df.columns]
        df["Amount"] = df["Amount"].astype(str).str.replace(',', '').astype(float)
        df["Date"] = pd.to_datetime(df["Date"], format='%d %b %Y', dayfirst=True)
        df["Details"] = df["Details"].str.strip().str.lower()
        df["Category"] = "Uncategorized"
        return df
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return None

def categorize_transaction(df):
    for category, keywords in st.session_state.categories.items():
        if category == "Uncategorized" or not keywords:
            continue
        lowered_keywords = [k.lower().strip() for k in keywords]
        df.loc[df["Details"].isin(lowered_keywords), "Category"] = category
    return df

def train_category_model(df):
    labeled = df[df["Category"] != "Uncategorized"]
    if len(labeled) < 5:
        return None
    X = labeled["Details"].apply(preprocess_text)
    y = labeled["Category"]
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("clf", LinearSVC())
    ])
    pipeline.fit(X, y)
    joblib.dump(pipeline, model_file)
    return pipeline

def load_category_model():
    return joblib.load(model_file) if os.path.exists(model_file) else None

def apply_ml_predictions(df, model):
    auto_df = df.copy()
    unlabeled = auto_df[auto_df["Category"] == "Uncategorized"]
    if len(unlabeled) == 0:
        return auto_df
    unlabeled["Details_clean"] = unlabeled["Details"].apply(preprocess_text)
    preds = model.predict(unlabeled["Details_clean"])
    auto_df.loc[unlabeled.index, "Category"] = preds
    return auto_df

def save_categories():
    with open(category_file, "w") as f:
        json.dump(st.session_state.categories, f)

def save_budgets():
    with open(budget_file, "w") as f:
        json.dump(st.session_state.budgets, f)

def add_keyword(category, keyword):
    keyword = keyword.strip().lower()
    if keyword and keyword not in st.session_state.categories.get(category, []):
        st.session_state.categories[category].append(keyword)
        save_categories()
        return True
    return False

def main():
    st.title("Personal Finance Dashboard")

    tab_dashboard, tab_transactions, tab_categories, tab_settings = st.tabs([
        "Dashboard", "Transactions", "Categories", "Settings"
    ])

    with tab_dashboard:
        st.header("Overview")
        if st.session_state.debit_df is not None:
            debit_df = st.session_state.debit_df.copy()

            st.subheader("Budget Alerts")
            debit_df["Category"] = debit_df["Category"].str.lower()
            for category in set(debit_df["Category"]):
                spent = debit_df[debit_df["Category"] == category]["Amount"].sum()
                limit = st.session_state.budgets.get(category, 0)
                if limit:
                    if spent > limit:
                        st.warning(f"Over budget in '{category.title()}': spent {spent:.2f} AED (limit: {limit:.2f} AED)")
                    else:
                        st.info(f"{category.title()}: {spent:.2f} AED of {limit:.2f} AED budget used.")

            category_totals = debit_df.groupby("Category")["Amount"].sum().reset_index()
            fig_pie = px.pie(category_totals, values="Amount", names="Category", title="Spending by Category")
            st.plotly_chart(fig_pie, use_container_width=True)

            monthly = debit_df.copy()
            monthly["Month"] = monthly["Date"].dt.to_period("M").astype(str)
            monthly = monthly.groupby("Month")["Amount"].sum().reset_index()
            fig_line = px.line(monthly, x="Month", y="Amount", title="Monthly Spending Trend")
            st.plotly_chart(fig_line, use_container_width=True)
        else:
            st.info("Upload a CSV file in the Transactions tab.")

    with tab_transactions:
        uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

        if uploaded_file is not None:
            df = load_transactions(uploaded_file)
            if df is not None:
                df = categorize_transaction(df)
                st.session_state.debit_df = df

        if st.session_state.debit_df is not None:
            debit_df = st.session_state.debit_df.copy()

            model = load_category_model()
            if model:
                auto_df = apply_ml_predictions(debit_df, model)
                with st.expander("ML Suggestions for Uncategorized"):
                    st.dataframe(auto_df[["Date", "Details", "Amount", "Category"]])
                    if st.button("Apply ML Predictions"):
                        st.session_state.debit_df["Category"] = auto_df["Category"]
                        st.success("ML predictions applied!")

            edited_df = st.data_editor(
                debit_df[["Date", "Details", "Amount", "Category"]],
                column_config={
                    "Date": st.column_config.DateColumn("Date", format="DD/MM/YYYY"),
                    "Details": st.column_config.TextColumn(),
                    "Amount": st.column_config.NumberColumn("Amount", format="%.2f AED"),
                    "Category": st.column_config.SelectboxColumn(
                        "Category",
                        options=list(st.session_state.categories.keys())
                    )
                },
                hide_index=True,
                use_container_width=True,
                key="transaction_editor"
            )

            if st.button("Save Changes"):
                for idx, row in edited_df.iterrows():
                    old_cat = debit_df.at[idx, "Category"]
                    new_cat = row["Category"]
                    if old_cat != new_cat:
                        add_keyword(new_cat, row["Details"])
                        st.session_state.debit_df.at[idx, "Category"] = new_cat
                st.success("Changes saved and keywords updated.")

            csv = st.session_state.debit_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Categorized CSV", csv, "categorized_transactions.csv", "text/csv")

    with tab_categories:
        st.header("Manage Categories")
        for cat in list(st.session_state.categories.keys()):
            with st.expander(cat):
                keywords = st.session_state.categories[cat]
                st.write("Keywords:", ", ".join(keywords) if keywords else "None")
                new_kw = st.text_input(f"Add keyword for {cat}", key=f"kw_{cat}")
                if st.button(f"Add to {cat}", key=f"btn_{cat}"):
                    if add_keyword(cat, new_kw):
                        st.success(f"Added '{new_kw}' to '{cat}'")
                    else:
                        st.warning("Keyword already exists or invalid input.")

        new_category = st.text_input("New Category")
        if st.button("Add New Category"):
            if new_category and new_category not in st.session_state.categories:
                st.session_state.categories[new_category] = []
                save_categories()
                st.rerun()
            elif new_category in st.session_state.categories:
                st.warning("Category already exists.")

    with tab_settings:
        st.header("Settings")
        st.write("Model Status:", "Loaded" if load_category_model() else "Not trained yet")
        if st.button("Retrain ML Model"):
            if st.session_state.debit_df is not None:
                model = train_category_model(st.session_state.debit_df)
                st.success("Model retrained and saved.")
            else:
                st.warning("No transaction data available to train.")

        st.subheader("Budget Settings")
        for cat in st.session_state.categories.keys():
            current = st.session_state.budgets.get(cat.lower(), 0)
            new_limit = st.number_input(f"Monthly budget for {cat}", min_value=0.0, value=float(current), step=50.0, key=f"budget_{cat}")
            st.session_state.budgets[cat.lower()] = new_limit
        if st.button("Save Budgets"):
            save_budgets()
            st.success("Budgets saved successfully.")

        if st.button("Reset Categories"):
            st.session_state.categories = {"Uncategorized": []}
            save_categories()
            st.rerun()

        if st.button("Clear All Data"):
            st.session_state.debit_df = None
            st.rerun()

if __name__ == "__main__":
    main()
