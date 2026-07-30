"""
📊 Ultimate Auto-ML God Mode (V2.0 MAX)
==========================================================
Instant Data Analytics and Multi-Model AI Showdown!
"""
import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import SVC, SVR
from sklearn.metrics import mean_squared_error, accuracy_score
import numpy as np
import pickle
import base64

st.set_page_config(page_title="Auto-ML God Mode V2.0 MAX", page_icon="📊", layout="wide")

# --- Custom CSS ---
st.markdown("""
<style>
    .main { background-color: #0E1117; }
    h1 {
        background: -webkit-linear-gradient(#00f2fe, #4facfe);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .st-emotion-cache-1c7y2kd { background-color: #1a1a24 !important; border-radius: 10px; }
    .metric-card {
        background: #1a1a24;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        border: 1px solid #4facfe;
    }
</style>
""", unsafe_allow_html=True)

st.title("📊 Ultimate Auto-ML God Mode (V2.0 MAX)")
st.markdown("**Upload a dataset and let the AI build, train, and evaluate multiple models instantly!**")

uploaded_file = st.file_uploader("📂 Upload your CSV Database", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    st.success("✅ Neural Databank successfully loaded!")
    
    tab1, tab2, tab3 = st.tabs(["📋 Deep Data Dive", "📈 Visual Analytics", "🥊 Multi-Model Showdown"])
    
    with tab1:
        st.header("Dataset Overview")
        st.dataframe(df.head(15), use_container_width=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("<div class='metric-card'><h3>Rows</h3><h2>{}</h2></div>".format(df.shape[0]), unsafe_allow_html=True)
        with col2:
            st.markdown("<div class='metric-card'><h3>Features</h3><h2>{}</h2></div>".format(df.shape[1]), unsafe_allow_html=True)
        with col3:
            st.markdown("<div class='metric-card'><h3>Missing Cells</h3><h2>{}</h2></div>".format(df.isnull().sum().sum()), unsafe_allow_html=True)
            
        st.subheader("Statistical Brain Scan")
        st.dataframe(df.describe(), use_container_width=True)
        
    with tab2:
        st.header("Interactive Cyber Analytics")
        numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
        
        if len(numeric_columns) >= 2:
            col_x, col_y = st.columns(2)
            with col_x:
                x_axis = st.selectbox("X-Axis (Feature 1)", numeric_columns)
            with col_y:
                y_axis = st.selectbox("Y-Axis (Feature 2)", numeric_columns, index=1)
            
            fig = px.scatter(df, x=x_axis, y=y_axis, template="plotly_dark", color_discrete_sequence=['#00f2fe'], title="Feature Distribution")
            st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("Neural Correlation Matrix")
            corr = df[numeric_columns].corr()
            fig_heatmap = px.imshow(corr, text_auto=True, aspect="auto", template="plotly_dark", color_continuous_scale="Viridis")
            st.plotly_chart(fig_heatmap, use_container_width=True)
        else:
            st.warning("Not enough numeric columns for plotting.")
            
    with tab3:
        st.header("🥊 Ultimate AI Model Showdown")
        st.markdown("Select your Target and Features. We will pit **Random Forest**, **Gradient Boosting**, and **Support Vector Machines (SVM)** against each other!")
        
        col_t, col_f = st.columns(2)
        with col_t:
            target = st.selectbox("🎯 Target Variable (What to Predict)", df.columns)
        
        df_clean = df.dropna(subset=[target])
        
        with col_f:
            features = st.multiselect("🧬 Feature Variables (Input Data)", [c for c in df_clean.columns if c != target], default=[c for c in df_clean.columns if c != target][:5])
        
        if features and target:
            X = df_clean[features]
            y = df_clean[target]
            
            X = pd.get_dummies(X, drop_first=True)
            X = X.fillna(X.mean(numeric_only=True))
            
            if st.button("🚀 IGNITE AUTO-ML SHOWDOWN", use_container_width=True):
                with st.spinner("Training Neural Architectures... 🧠"):
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                    
                    is_classification = df_clean[target].dtype == 'object' or df_clean[target].nunique() < 10
                    
                    models = {}
                    if is_classification:
                        models = {
                            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
                            "Gradient Boosting": GradientBoostingClassifier(random_state=42),
                            "SVM": SVC(kernel='linear', probability=True)
                        }
                    else:
                        models = {
                            "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
                            "Gradient Boosting": GradientBoostingRegressor(random_state=42),
                            "SVM": SVR(kernel='linear')
                        }
                    
                    results = []
                    best_model = None
                    best_score = -1 if is_classification else float('inf')
                    
                    for name, model in models.items():
                        model.fit(X_train, y_train)
                        preds = model.predict(X_test)
                        if is_classification:
                            score = accuracy_score(y_test, preds)
                            results.append({"Model": name, "Accuracy": score})
                            if score > best_score:
                                best_score = score
                                best_model = model
                        else:
                            score = np.sqrt(mean_squared_error(y_test, preds))
                            results.append({"Model": name, "RMSE": score})
                            if score < best_score:
                                best_score = score
                                best_model = model
                    
                    # Display Leaderboard
                    st.subheader("🏆 Model Leaderboard")
                    results_df = pd.DataFrame(results).sort_values(by="Accuracy" if is_classification else "RMSE", ascending=not is_classification)
                    st.dataframe(results_df, use_container_width=True)
                    
                    st.success(f"👑 Best Model: **{results_df.iloc[0]['Model']}**")
                    
                    # Feature Importance (if applicable)
                    if hasattr(best_model, "feature_importances_"):
                        st.subheader("🧠 Deep Feature Importance (Best Model)")
                        importances = best_model.feature_importances_
                        imp_df = pd.DataFrame({"Feature": X.columns, "Importance": importances}).sort_values(by="Importance", ascending=False).head(10)
                        fig_imp = px.bar(imp_df, x="Importance", y="Feature", orientation='h', template="plotly_dark", color="Importance", color_continuous_scale="Blues")
                        st.plotly_chart(fig_imp, use_container_width=True)
                    
                    # Export Model
                    st.subheader("💾 Export Winning Model")
                    pkl_model = pickle.dumps(best_model)
                    st.download_button(
                        label="Download Best Model (.pkl)",
                        data=pkl_model,
                        file_name="best_ai_model.pkl",
                        mime="application/octet-stream"
                    )
else:
    st.info("💡 Connect your Neural Databank (Upload a CSV) to initiate the Auto-ML sequence.")
