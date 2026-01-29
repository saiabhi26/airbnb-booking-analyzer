# 🏠 Airbnb Listings Analysis & Predictive Dashboard

An interactive Streamlit dashboard for analyzing Airbnb listings data and predicting key metrics using machine learning models. This project explores pricing trends, booking behaviors, and host performance factors across multiple cities.

## 📊 Project Overview

This application provides a comprehensive analysis of Airbnb listing data with 8 different machine learning models to predict various aspects of rental performance. The dashboard enables users to interact with data visualizations and get predictions for key metrics that influence booking success.

### 🔍 Key Features Analyzed:
- **Property Type Classification** - Predict listing type based on features
- **Price Prediction** - Estimate optimal pricing using amenities and location
- **Booking Rate Analysis** - Understand factors affecting booking likelihood
- **Superhost Impact** - Analyze how host status affects performance
- **Review Score Effects** - Correlate ratings with booking rates
- **Neighborhood Trends** - Track listing growth over time
- **Host Verification Impact** - Measure trust factor on occupancy
- **Response Time Effects** - Analyze how responsiveness influences bookings

## 📁 Project Structure

airbnb-booking-analyzer/ :
/app.py # Main Streamlit application
/cleaned_data.csv # Dataset 
/requirements.txt # Dependencies
/README.md # This file


## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

```bash
### Step 1: Clone the Repository
git clone https://github.com/saiabhi26/airbnb-booking-analyzer.git
cd airbnb-booking-analyzer 

### Step 2: Install dependencies
pip install -r requirements.txt

### Step 3: Run the application
streamlit run app.py
