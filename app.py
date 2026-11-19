import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import time

# 1. PAGE SETUP
st.set_page_config(page_title="California House Price Predictor", page_icon="🏠", layout="wide")

# 2. COLOURFUL THEME CSS
st.markdown("""
<style>
.stApp { background: linear-gradient(135deg, #FFF9C4 0%, #FFCCBC 30%, #C8E6C9 60%, #BBDEFB 100%); }
h1 { color: #E65100 !important; text-align: center; font-size: 45px !important; }
h2, h3 { color: #1A237E !important; }
.stButton>button { background: linear-gradient(90deg, #FF6B6B, #FFA726); color: white; font-size: 20px !important; border-radius: 20px; padding: 10px 30px; border: none; }
div[data-testid="stSidebar"] { background: linear-gradient(180deg, #4A148C, #7B1FA2, #E91E63); }
div[data-testid="stSidebar"] * { color: white !important; }
</style>
""", unsafe_allow_html=True)

# 3. HEADER WITH IMAGES
st.title("🏠 California House Price Predictor 🌈")
st.markdown("<h3 style='text-align:center; color:#6A1B9A;'>✨ AI Powered | Random Forest Model | Live Prediction ✨</h3>", unsafe_allow_html=True)

# TOP BANNER IMAGES - 3 Houses
col_img1, col_img2, col_img3 = st.columns(3)
with col_img1:
    st.image("https://images.unsplash.com/photo-1568605114967-8130f3a36994?w=600", caption="🏡 Modern Family House", width="stretch")
with col_img2:
    st.image("https://images.unsplash.com/photo-1600596542815-ffad4c1539a9?w=600", caption="🏠 Luxury Villa in California", width="stretch")
with col_img3:
    st.image("https://images.unsplash.com/photo-1512917774080-9991f1c4c750?w=600", caption="🌴 California Dream Home", width="stretch")

# 4. LOAD DATA
with st.spinner("📥 Loading Dataset..."):
    time.sleep(1)
    df = pd.read_csv("housing.csv", encoding='latin1', on_bad_lines='skip')
    df = df.dropna()

st.success(f"✅ Dataset Loaded Successfully! Total Rows: {len(df)} | Columns: {len(df.columns)}")

# 5. SIDEBAR - USER INPUTS
st.sidebar.image("https://images.unsplash.com/photo-1560518883-ce09059eeffa?w=400", caption="Find Your Dream House")
st.sidebar.header("🎛️ House Details - Sliders")
st.sidebar.markdown("Move the sliders to see the price instantly!")

medinc = st.sidebar.slider("💰 Median Income (in $10k)", 0.5, 15.0, 3.5)
house_age = st.sidebar.slider("🏡 Housing Median Age", 1, 52, 28)
total_rooms = st.sidebar.slider("🛏️ Total Rooms", 100, 10000, 2000)
total_bedrooms = st.sidebar.slider("🛌 Total Bedrooms", 10, 2000, 400)
population = st.sidebar.slider("👨‍👩‍👧 Population", 50, 30000, 1000)
households = st.sidebar.slider("👪 Households", 10, 5000, 400)
latitude = st.sidebar.slider("📍 Latitude", 32.5, 42.0, 34.0)
longitude = st.sidebar.slider("📍 Longitude", -124.0, -114.0, -118.0)
ocean = st.sidebar.selectbox("🌊 Ocean Proximity", ["NEAR BAY", "INLAND", "NEAR OCEAN", "<1H OCEAN", "ISLAND"])

# 6. TABS
tab1, tab2, tab3, tab4, tab5 = st.tabs(["🔮 PREDICT PRICE", "📊 DATA ANALYSIS", "🤖 MODEL TRAINING", "📈 GRAPHS", "🖼️ HOUSE GALLERY"])

with tab1:
    st.subheader("🔮 Live Prediction")
    st.image("https://images.unsplash.com/photo-1600585154340-be6161a56a0c?w=800", caption="Your Future Home - Predict Its Price!", width="stretch")
    if st.button("✨ Predict Price ✨"):
        with st.spinner("🤖 AI Model Training & Predicting..."):
            progress = st.progress(0)
            for i in range(100):
                time.sleep(0.01)
                progress.progress(i+1)
            X = df.drop(['median_house_value','ocean_proximity'], axis=1)
            y = df['median_house_value']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = RandomForestRegressor(n_estimators=50, random_state=42)
            model.fit(X_train, y_train)
            pred_test = model.predict(X_test)
            acc = r2_score(y_test, pred_test)
            input_data = pd.DataFrame([[medinc, house_age, total_rooms, total_bedrooms, population, households, latitude, longitude]], columns=X.columns)
            price = model.predict(input_data)[0]
            st.balloons()
            col1, col2, col3 = st.columns(3)
            col1.metric("💰 Predicted Price", f"${price:,.0f}")
            col2.metric("🎯 Model Accuracy (R2)", f"{acc*100:.2f}%")
            col3.metric("📦 Rooms per Household", f"{total_rooms/households:.1f}")
            st.success(f"🎉 Your House Price: **${price:,.2f}**")
            st.info(f"Ocean Proximity: **{ocean}** | Location: **{latitude}, {longitude}**")
            if price > 300000:
                st.image("https://images.unsplash.com/photo-1600047509807-ba8f99d2cdde?w=600", caption="Premium House - High Value!")
            elif price > 150000:
                st.image("https://images.unsplash.com/photo-1570129477492-45c003edd2be?w=600", caption="Good Value Family House")
            else:
                st.image("https://images.unsplash.com/photo-1580587771525-78b9dba3b914?w=600", caption="Budget Friendly House")

with tab2:
    st.subheader("📊 Dataset Preview & Stats")
    st.image("https://images.unsplash.com/photo-1460925895917-afdab827c52f?w=800", caption="Data Analysis Dashboard", width="stretch")
    st.write("First 10 Rows:")
    st.dataframe(df.head(10), width="stretch")
    col1, col2 = st.columns(2)
    col1.write("**Dataset Info:**")
    col1.write(df.describe())
    col2.write("**Missing Values:**")
    col2.write(df.isnull().sum())
    st.write(f"**Average House Price:** ${df['median_house_value'].mean():,.0f} | **Max Price:** ${df['median_house_value'].max():,.0f}")

with tab3:
    st.subheader("🤖 How Model Works? - 5 Steps")
    st.image("https://images.unsplash.com/photo-1555255707-c07966088b7b?w=800", caption="Artificial Intelligence - How It Learns", width="stretch")
    st.markdown("""
    1.  **📥 Data Collection:** California Housing Dataset (20,640 houses)
    2.  **🧹 Data Cleaning:** Missing values removed
    3.  **✂️ Train-Test Split:** 80% Training, 20% Testing
    4.  **🌲 Random Forest:** 50 Trees used for training
    5.  **📏 Evaluation:** RMSE & R2 Score checked for accuracy
    """)
    if st.button("🚀 Train Model Now"):
        X = df.drop(['median_house_value','ocean_proximity'], axis=1)
        y = df['median_house_value']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestRegressor(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, pred))
        r2 = r2_score(y_test, pred)
        st.success(f"✅ Training Done! RMSE: {rmse:.2f} | R2 Score: {r2:.4f}")
        imp = pd.DataFrame({"Feature": X.columns, "Importance": model.feature_importances_}).sort_values("Importance", ascending=False)
        st.bar_chart(imp.set_index("Feature"))

with tab4:
    st.subheader("📈 Beautiful Graphs")
    c1, c2 = st.columns(2)
    with c1:
        st.write("**Income vs House Price**")
        fig, ax = plt.subplots()
        sns.scatterplot(data=df.sample(1000), x="median_income", y="median_house_value", hue="ocean_proximity", ax=ax)
        st.pyplot(fig)
    with c2:
        st.write("**House Price Distribution**")
        fig2, ax2 = plt.subplots()
        sns.histplot(df['median_house_value'], bins=30, kde=True, color="purple", ax=ax2)
        st.pyplot(fig2)
    st.write("**Correlation Heatmap**")
    fig3, ax3 = plt.subplots(figsize=(8,5))
    sns.heatmap(df.select_dtypes(include=np.number).corr(), annot=True, cmap="coolwarm", ax=ax3)
    st.pyplot(fig3)

with tab5:
    st.subheader("🖼️ California House Gallery - Choose Your Style")
    st.markdown("**Which house do you like? Imagine its price!**")
    g1, g2, g3 = st.columns(3)
    with g1:
        st.image("https://images.unsplash.com/photo-1583608205776-bfd35f0d9f83?w=500", caption="🏡 Classic House")
        st.image("https://images.unsplash.com/photo-1564013799919-ab600027ffc6?w=500", caption="🌙 Night View Luxury House")
    with g2:
        st.image("https://images.unsplash.com/photo-1605276374104-dee2a0ed3cd6?w=500", caption="🏠 Modern White House")
        st.image("https://images.unsplash.com/photo-1592595896551-12b371d546d5?w=500", caption="🏖️ Beach House")
    with g3:
        st.image("https://images.unsplash.com/photo-1605146769289-440113cc3d00?w=500", caption="🏘️ Suburban House")
        st.image("https://images.unsplash.com/photo-1523217582562-09d0def993a6?w=500", caption="🌳 Garden House")

st.markdown("---")
st.markdown("<p style='text-align:center; color:#E65100; font-weight:bold;'>Built with ❤️ using Streamlit | Model: Random Forest | Dataset: California Housing</p>", unsafe_allow_html=True)