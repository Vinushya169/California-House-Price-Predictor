# 🏠 California House Price Predictor

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30-red)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3-orange)
![Status](https://img.shields.io/badge/Status-Live-success)

> An AI-powered web application that predicts California house prices in real-time using Machine Learning.

### 🚀 Live Demo
**👉 Click Here to Try the App:** https://california-house-price-predictor-vinushya.streamlit.app

**GitHub Repository:** https://github.com/Vinushya169/California-House-Price-Predictor

---

## 📋 Table of Contents
- [About The Project](#-about-the-project)
- [Key Features](#-key-features)
- [Dataset](#-dataset)
- [Machine Learning Model](#-machine-learning-model)
- [Tech Stack](#️-tech-stack)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [How To Use](#-how-to-use)
- [Model Performance](#-model-performance)
- [Future Improvements](#-future-improvements)
- [Author](#-author)

---

## 📖 About The Project

Buying a house is one of the biggest decisions in life. The price depends on many factors like location, income level of the area, age of the house, and number of rooms.

This project uses **Machine Learning** to solve this problem. By analyzing 20,640 California housing records, our model learns the pattern between house features and its price, and then predicts the price for any new house instantly.

This app is built for:
* **Buyers** - To check if a house is overpriced or fairly priced
* **Sellers** - To set a correct selling price
* **Real Estate Agents** - To give quick estimates to clients
* **Students & Developers** - To learn how ML models are deployed as web apps

## ✨ Key Features

* **Real-Time Prediction:** Change any slider and get the price instantly.
* **Interactive UI:** User-friendly sliders for all inputs like Median Income, House Age, Avg Rooms, Population, Latitude & Longitude.
* **Smart Visualization:** Price displayed with beautiful metrics and charts.
* **Location Aware:** Uses Latitude & Longitude to understand location-based pricing.
* **Fast & Lightweight:** Loads in seconds, works on Mobile and Desktop.
* **Accurate Model:** Trained on 20,000+ real housing data points.

## 📊 Dataset

**Source:** California Housing Dataset (1990 US Census Data)

**Total Records:** 20,640 houses

**Features Used:**

| Feature | Description |
| :--- | :--- |
| `MedInc` | Median Income in block group |
| `HouseAge` | Median house age in block group |
| `AveRooms` | Average number of rooms per household |
| `AveBedrms` | Average number of bedrooms per household |
| `Population` | Block group population |
| `AveOccup` | Average number of household members |
| `Latitude` | Latitude of the house location |
| `Longitude` | Longitude of the house location |
| `MedHouseVal` | **Target:** Median house value (Price to Predict) |

## 🤖 Machine Learning Model

**Algorithm Used:** `Random Forest Regressor`

**Why Random Forest?**
We tested Linear Regression, Decision Tree, and Random Forest. Random Forest gave the best accuracy because it can handle non-linear relationships (like how price changes with location) and it avoids overfitting by averaging 100+ decision trees.

**Training Process:**
1.  Data Cleaning - Removed missing values and handled outliers
2.  Feature Engineering - Selected 8 most important features
3.  Train-Test Split - 80% for Training, 20% for Testing
4.  Model Training - Trained Random Forest with 100 estimators
5.  Evaluation - Tested with R2 Score and RMSE

## 🛠️ Tech Stack

* **Programming Language:** Python 3.11
* **Web Framework:** Streamlit
* **Machine Learning:** Scikit-Learn
* **Data Processing:** Pandas, NumPy
* **Visualization:** Matplotlib, Plotly
* **Deployment:** Streamlit Cloud
* **Version Control:** Git & GitHub

## 📁 Project Structure

California-House-Price-Predictor/
│
├── http://app.py                  # Main Streamlit Application
├── http://housing.csv             # California Housing Dataset (20,640 rows)
├── http://requirements.txt        # All Python Dependencies
├── http://README.md               # Project Documentation
└── .gitignore              # Git Ignore File

💻 Installation

Follow these steps to run the project on your local machine:

**1. Clone the Repository**
```bash
git clone https://github.com/Vinushya169/California-House-Price-Predictor.git
cd California-House-Price-Predictor
*2. Create Virtual Environment (Optional but Recommended)*
python -m venv venv
venv\Scripts\activate  # For Windows
source venv/bin/activate # For Mac/Linux
*3. Install Dependencies*
pip install -r requirements.txt
*4. Run the App*
streamlit run app.py
The app will open automatically in your browser at `http://localhost:8501`

## 🎯 How To Use

1.  Open the Live App link.
2.  On the left sidebar, adjust the sliders:
    - *Median Income:* Increase income = Price increases
    - *House Age:* Older houses have different pricing
    - *Avg Rooms:* More rooms = Higher price
    - *Location:* Move Latitude / Longitude to change area
3.  Click *Predict Price* button.
4.  See the predicted price instantly on the main screen!

*Example:*
> Input: Income = 5.0, House Age = 20, Rooms = 5, Location = Los Angeles
> Output: Predicted Price = $280,000

 📈 Model Performance
Metric | Score
**R² Score** | 0.82
**Mean Absolute Error** | ~ $35,000
**Training Data** | 16,512 houses
**Testing Data** | 4,128 houses
_The model explains 82% of the price variation, which is excellent for real estate data._

 🔮 Future Improvements

- [ ] Add Map Visualization to show house location on Google Maps
- [ ] Add Comparison Feature - Compare 2 houses side-by-side
- [ ] Add Price History Graph for California
- [ ] Deploy as Mobile App using Flutter
- [ ] Add User Login to save predictions

 👩‍💻 Author

 Vinushya.M
- B.E CSE Student | Aspiring Data Scientist & AI Developer
- Passionate about Machine Learning and Building Real-World Projects

*Connect With Me:*
- GitHub: https://github.com/Vinushya169
- LinkedIn: https://www.linkedin.com/in/vinushya-m-80b92b400

 📄 License

This project is open source and available under the [MIT License](LICENSE).

⭐ Show Your Support

If you like this project, please give it a ⭐ on GitHub!

---
_Made with ❤️ by Vinushya using Python & Streamlit_


