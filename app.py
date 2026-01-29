import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import sqlite3
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import AdaBoostRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import VotingClassifier
# from pmdarima import auto_arima

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('cleaned_data.csv')
    # Convert boolean-like columns to integer
    bool_columns = ['host_is_superhost', 'instant_bookable', 'host_identity_verified']
    for col in bool_columns:
        df[col] = df[col].map({'t': 1, 'f': 0}).astype(int)

    
    # Handle missing values and data transformations if needed
    df['availability_365'] = df['availability_365'].replace(0, np.nan)  # Avoid division by zero
    df['booking_rate'] = df['number_of_reviews'] / df['availability_365']  # Proxy for booking rate
    df['booking_rate'] = df['booking_rate'].fillna(0)  # Handle NaNs caused by unavailable data
    df['date'] = pd.to_datetime(df['last_scraped'])  # Replace 'timestamp_column' with your actual column
    df['listing_count'] = df.groupby('neighbourhood_cleansed')['neighbourhood_cleansed'].transform('count')
    return df

df = load_data()

# Create SQLite database
conn = sqlite3.connect('airbnb.db')
df.to_sql('listings', conn, if_exists='replace', index=False)

# App title
st.title('Airbnb Listings Analysis')



#####################################################################################################################
# Understanding data and visualization.
st.header('Understanding data')
st.subheader('Visualisation - Prices vs Geospatial Location')

fig = px.scatter_mapbox(df, lat='latitude', lon='longitude', color='price',
                        hover_name='name', zoom=10)
fig.update_layout(mapbox_style='open-street-map')
st.plotly_chart(fig)

################################################################################################################
# Model developed using ensemble
st.header('Problem - 1: Property type prediction')
st.subheader('Visualisation')

region = st.selectbox('Select region', df['neighbourhood_cleansed'].unique())

query = f"""
SELECT property_type, COUNT(*) as count, AVG(price) as avg_price
FROM listings
WHERE neighbourhood_cleansed = '{region}'
GROUP BY property_type
ORDER BY count DESC
"""

result = pd.read_sql_query(query, conn)
st.write(result)

fig = px.bar(result, x='property_type', y='count', color='avg_price',
             labels={'count': 'Number of listings', 'property_type': 'Property Type'},
             title=f'Property Types in {region}')
st.plotly_chart(fig)


st.subheader('Property Type Prediction')

# Prepare data
features = ['price', 'accommodates', 'bedrooms', 'bathrooms', 'review_scores_rating', 'neighbourhood_cleansed']
X = df[features]
y = df['property_type']

# Encode categorical variables
le_neighbourhood = LabelEncoder()
le_property_type = LabelEncoder()
X['neighbourhood_cleansed'] = le_neighbourhood.fit_transform(X['neighbourhood_cleansed'])
y_encoded = le_property_type.fit_transform(y)

# Normalize numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

# Define individual models
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
nn_model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42)

# Create ensemble model
ensemble_model = VotingClassifier(
    estimators=[('rf', rf_model), ('gb', gb_model), ('nn', nn_model)],
    voting='soft'
)

# Train the ensemble model
ensemble_model.fit(X_train, y_train)

# Evaluate the model
y_pred = ensemble_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
# st.write(f"Ensemble Model Accuracy: {accuracy:.2f}")

# User input for prediction
price = st.number_input('Price', min_value=0, value=100)
accommodates = st.number_input('Accommodates', min_value=1, value=2)
bedrooms = st.number_input('Bedrooms', min_value=0, value=1)
bathrooms = st.number_input('Bathrooms', min_value=0.0, value=1.0, step=0.5)
review_score = st.slider('Review Score', 0.0, 5.0, 4.5)
neighbourhood = st.selectbox('Neighbourhood', df['neighbourhood_cleansed'].unique())

if st.button('Predict Property Type'):
    input_data = np.array([[price, accommodates, bedrooms, bathrooms, review_score, 
                            le_neighbourhood.transform([neighbourhood])[0]]])
    input_scaled = scaler.transform(input_data)
    prediction = ensemble_model.predict(input_scaled)
    predicted_class = le_property_type.inverse_transform(prediction)[0]
    st.success(f'Predicted Property Type: {predicted_class}')


####################################################################################################################
# Model developed - 1(Using RF)
# Average price of a listing based on the amenities.
st.header('Problem - 2: Price prediction using amenities')
st.subheader('Visualisation')
amenities = st.multiselect('Select amenities', ['Wifi', 'Pool', 'Kitchen', 'Heating', 'Washer', 'Dryer', 'Long term stays allowed', 'Air conditioning'])

query = "SELECT AVG(price) as avg_price, "
for amenity in amenities:
    query += f"SUM(CASE WHEN amenities LIKE '%{amenity}%' THEN 1 ELSE 0 END) as has_{amenity.lower().replace(' ', '_')}, "
query = query.rstrip(', ')
query += " FROM listings"

result = pd.read_sql_query(query, conn)
st.write(result)

# Machine Learning: Price Prediction
st.subheader('Price Prediction Model')

# Prepare data for ML
df['has_wifi'] = df['amenities'].str.contains('Wifi').astype(int)
df['has_pool'] = df['amenities'].str.contains('Pool').astype(int)
df['has_kitchen'] = df['amenities'].str.contains('Kitchen').astype(int)
df['has_heating'] = df['amenities'].str.contains('Heating').astype(int)
df['has_washer'] = df['amenities'].str.contains('Washer').astype(int)
df['has_dryer'] = df['amenities'].str.contains('Dryer').astype(int)
df['has_long_term_stays'] = df['amenities'].str.contains('Long term stays allowed').astype(int)
df['has_air_conditioning'] = df['amenities'].str.contains('Air conditioning').astype(int)

features = ['room_type', 'accommodates', 'bathrooms', 'bedrooms', 'beds', 
            'has_wifi', 'has_pool', 'has_kitchen', 'has_heating', 'has_washer', 'has_dryer',
            'has_long_term_stays', 'has_air_conditioning',
            'neighbourhood_cleansed']
X = df[features]
y = df['price']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline
categorical_features = ['room_type', 'neighbourhood_cleansed']
numeric_features = ['accommodates', 'bathrooms', 'bedrooms', 'beds', 
                    'has_wifi', 'has_pool', 'has_kitchen', 'has_heating', 'has_washer', 'has_dryer',
                    'has_long_term_stays', 'has_air_conditioning']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='median'), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Train the model
model.fit(X_train, y_train)

# User input for prediction
st.subheader('Predict Price for New Listing')
room_type = st.selectbox('Room Type', df['room_type'].unique())
neighbourhood = st.selectbox('Neighbourhood', df['neighbourhood_cleansed'].unique(), key = 'neigh_key')
accommodates = st.number_input('Accommodates', min_value=1, max_value=16, value=2)
bathrooms = st.number_input('Bathrooms', min_value=0.0, max_value=8.0, value=1.0, step=0.5)
bedrooms = st.number_input('Bedrooms', min_value=0, max_value=8, value=1)
beds = st.number_input('Beds', min_value=1, max_value=16, value=1)

st.subheader('Select Amenities')
has_wifi = st.checkbox('Has Wifi')
has_pool = st.checkbox('Has Pool')
has_kitchen = st.checkbox('Has Kitchen')
has_heating = st.checkbox('Has Heating')
has_washer = st.checkbox('Has Washer')
has_dryer = st.checkbox('Has Dryer')
has_long_term_stays = st.checkbox('Long term stays allowed')
has_air_conditioning = st.checkbox('Has Air Conditioning')

if st.button('Predict Price'):
    input_data = pd.DataFrame({
        'room_type': [room_type],
        'neighbourhood_cleansed': [neighbourhood],
        'accommodates': [accommodates],
        'bathrooms': [bathrooms],
        'bedrooms': [bedrooms],
        'beds': [beds],
        'has_wifi': [int(has_wifi)],
        'has_pool': [int(has_pool)],
        'has_kitchen': [int(has_kitchen)],
        'has_heating': [int(has_heating)],
        'has_washer': [int(has_washer)],
        'has_dryer': [int(has_dryer)],
        'has_long_term_stays': [int(has_long_term_stays)],
        'has_air_conditioning': [int(has_air_conditioning)]
    })
    prediction = model.predict(input_data)
    st.success(f'Predicted Price: ${prediction[0]:.2f}')


################################################################################################################
# model developed - 3(Using SVM)
# Problem: Superhost badge effect on booking rates
st.header('3. Booking rate prediction based on host type')
st.subheader('Visualisation')
superhost_query = """
SELECT host_is_superhost, AVG(30 - availability_30) as avg_bookings
FROM listings
GROUP BY host_is_superhost
"""
superhost_result = pd.read_sql_query(superhost_query, conn)
st.write(superhost_result)
fig = px.bar(superhost_result, x='host_is_superhost', y='avg_bookings', 
             title='Average Bookings by Superhost Status')
st.plotly_chart(fig)

st.subheader('Impact of super host on booking rate')
# Prepare data for ML
features_superhost = ['review_scores_rating', 'number_of_reviews', 'review_scores_accuracy', 
                      'review_scores_cleanliness', 'review_scores_checkin', 'review_scores_communication',
                      'review_scores_location', 'review_scores_value', 'instant_bookable']
X_superhost = df[features_superhost]
y_superhost = df['host_is_superhost']

# Split the data
X_train_superhost, X_test_superhost, y_train_superhost, y_test_superhost = train_test_split(X_superhost, y_superhost, test_size=0.2, random_state=42)

# Create a pipeline with SVM
superhost_model = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('classifier', SVC(kernel='rbf', probability=True, random_state=42))
])

# Train the model
superhost_model.fit(X_train_superhost, y_train_superhost)

# User input for prediction
st.subheader('Predict Superhost Status')
review_scores_rating = st.slider('Review Scores Rating', 0.0, 5.0, 4.5)
number_of_reviews = st.number_input('Number of Reviews', min_value=0, value=10)
review_scores_accuracy = st.slider('Review Scores Accuracy', 0.0, 5.0, 4.5)
review_scores_cleanliness = st.slider('Review Scores Cleanliness', 0.0, 5.0, 4.5)
review_scores_checkin = st.slider('Review Scores Check-in', 0.0, 5.0, 4.5)
review_scores_communication = st.slider('Review Scores Communication', 0.0, 5.0, 4.5)
review_scores_location = st.slider('Review Scores Location', 0.0, 5.0, 4.5)
review_scores_value = st.slider('Review Scores Value', 0.0, 5.0, 4.5)
instant_bookable = st.checkbox('Instant Bookable')

if st.button('Predict Superhost Status'):
    input_data_superhost = pd.DataFrame({
        'review_scores_rating': [review_scores_rating],
        'number_of_reviews': [number_of_reviews],
        'review_scores_accuracy': [review_scores_accuracy],
        'review_scores_cleanliness': [review_scores_cleanliness],
        'review_scores_checkin': [review_scores_checkin],
        'review_scores_communication': [review_scores_communication],
        'review_scores_location': [review_scores_location],
        'review_scores_value': [review_scores_value],
        'instant_bookable': [int(instant_bookable)]
    })
    prediction_superhost = superhost_model.predict(input_data_superhost)
    probability_superhost = superhost_model.predict_proba(input_data_superhost)[0][1]
    st.success(f'Predicted Superhost Status: {"Yes" if prediction_superhost[0] else "No"}')
    st.info(f'Probability of being a Superhost: {probability_superhost:.2f}')


#####################################################################################################################
# model developed - 4(Using KNN)
# Problem : Review score rating effect on bookings
st.header('4. Booking prediction based on review score')
st.subheader('Visualisation')
review_score_query = """
SELECT 
    CASE 
        WHEN review_scores_rating BETWEEN 0 AND 20 THEN '0-20'
        WHEN review_scores_rating BETWEEN 21 AND 40 THEN '21-40'
        WHEN review_scores_rating BETWEEN 41 AND 60 THEN '41-60'
        WHEN review_scores_rating BETWEEN 61 AND 80 THEN '61-80'
        WHEN review_scores_rating BETWEEN 81 AND 100 THEN '81-100'
    END as rating_range,
    AVG(30 - availability_30) as avg_bookings
FROM listings
WHERE review_scores_rating IS NOT NULL
GROUP BY rating_range
ORDER BY rating_range
"""
# review_score_result = pd.read_sql_query(review_score_query, conn)
# st.write(review_score_result)
# fig = px.line(review_score_result, x='rating_range', y='avg_bookings', 
#               title='Average Bookings by Review Score Rating')
# st.plotly_chart(fig)

st.subheader('Booking prediction using host type')

features_booking = [
    'price', 'review_scores_rating', 'host_is_superhost', 
    'instant_bookable', 'accommodates', 'bedrooms', 
    'bathrooms', 'number_of_reviews'
]
target_booking = 'booking_rate'

# Define the data
X_booking = df[features_booking]
y_booking = df[target_booking]

# Split Data
X_train_booking, X_test_booking, y_train_booking, y_test_booking = train_test_split(
    X_booking, y_booking, test_size=0.2, random_state=42
)

# KNN Model Pipeline
knn_booking_model = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),  # Handle missing values
    ('scaler', StandardScaler()),                   # Standardize numerical features
    ('regressor', KNeighborsRegressor(n_neighbors=5))  # KNN Model
])

# Train the Model
knn_booking_model.fit(X_train_booking, y_train_booking)

# Sidebar Input for Prediction
price = st.number_input('Price (per night)', min_value=0, value=100, key="price_input")
review_scores_rating = st.slider('Review Scores Rating', 0.0, 5.0, 4.5, key="review_score_slider")
host_is_superhost = st.checkbox('Is Superhost', key="superhost_checkbox")
instant_bookable = st.checkbox('Instant Bookable', key="instant_bookable_checkbox")
accommodates = st.number_input('Accommodates', min_value=1, value=2, key="accommodates_input")
bedrooms = st.number_input('Bedrooms', min_value=0, value=1, key="bedrooms_input")
bathrooms = st.number_input('Bathrooms', min_value=0.0, value=1.0, step=0.5, key="bathrooms_input")
number_of_reviews = st.number_input('Number of Reviews', min_value=0, value=10, key="reviews_input")

# Predict Booking Rate
if st.button('Predict Booking Rate', key="predict_button"):
    input_data_booking = pd.DataFrame({
        'price': [price],
        'review_scores_rating': [review_scores_rating],
        'host_is_superhost': [int(host_is_superhost)],
        'instant_bookable': [int(instant_bookable)],
        'accommodates': [accommodates],
        'bedrooms': [bedrooms],
        'bathrooms': [bathrooms],
        'number_of_reviews': [number_of_reviews]
    })

    prediction_booking = knn_booking_model.predict(input_data_booking)
    st.success(f'Predicted Booking Rate: {prediction_booking[0]:.2f}')



####################################################################################################################
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# Problem 5: New listings over time by neighborhood
st.header('5. New Listings Over Time by Neighborhood')
neighborhood = st.selectbox('Select Neighborhood', df['neighbourhood_cleansed'].unique())
neighborhood_query = f"""
SELECT strftime('%Y', host_since) as year, COUNT(*) as new_listings
FROM listings
WHERE neighbourhood_cleansed = '{neighborhood}'
GROUP BY year
ORDER BY year
"""
neighborhood_result = pd.read_sql_query(neighborhood_query, conn)
st.write(neighborhood_result)
fig = px.line(neighborhood_result, x='year', y='new_listings', 
              title=f'New Listings Over Time in {neighborhood}')
st.plotly_chart(fig)

st.subheader('5. New listings prediction based on neighborhood')

# Prepare data
df['host_since'] = pd.to_datetime(df['host_since'])
df['year'] = df['host_since'].dt.year

# User input
neighborhood = st.selectbox('Select Neighborhood', df['neighbourhood_cleansed'].unique(), key = 'neigh_select')

# Filter data for selected neighborhood
neighborhood_data = df[df['neighbourhood_cleansed'] == neighborhood]
yearly_data = neighborhood_data.groupby('year').size().reset_index(name='new_listings')

# Display raw data
# st.subheader('Raw Data')
# st.write(yearly_data)

# Visualize raw data
fig = px.line(yearly_data, x='year', y='new_listings', 
              title=f'New Listings Over Time in {neighborhood}')
st.plotly_chart(fig)

# Prepare data for Random Forest
X = yearly_data['year'].values.reshape(-1, 1)
y = yearly_data['new_listings'].values

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions
y_pred = rf_model.predict(X_test)

# # Calculate RMSE
# rmse = np.sqrt(mean_squared_error(y_test, y_pred))
# st.write(f"Root Mean Squared Error: {rmse:.2f}")

# Predict future values
future_years = np.arange(yearly_data['year'].max() + 1, yearly_data['year'].max() + 6).reshape(-1, 1)
future_predictions = rf_model.predict(future_years)

# Create DataFrame for future predictions
future_df = pd.DataFrame({'year': future_years.flatten(), 'predicted_new_listings': future_predictions})

# Combine historical data and future predictions
combined_df = pd.concat([yearly_data, future_df])

# Visualize historical data and predictions
fig_pred = px.line(combined_df, x='year', y=['new_listings', 'predicted_new_listings'],
                   title=f'New Listings Forecast for {neighborhood}')
fig_pred.update_layout(yaxis_title='Number of New Listings', xaxis_title='Year')
st.plotly_chart(fig_pred)


####################################################################################################################
# model developed - 6(Using Logistic Regression)
# Problem : Host identity verification effect on reviews and occupancy
st.header('6. Booking prediction on host identity verification')
st.subheader('Visualisation')
identity_query = """
SELECT host_identity_verified, 
       AVG(number_of_reviews) as avg_reviews,
       AVG(30 - availability_30) / 30.0 as avg_occupancy_rate
FROM listings
GROUP BY host_identity_verified
"""
identity_result = pd.read_sql_query(identity_query, conn)
st.write(identity_result)
fig = px.bar(identity_result, x='host_identity_verified', y=['avg_reviews', 'avg_occupancy_rate'], 
             title='Host Identity Verification Effect on Reviews and Occupancy')
st.plotly_chart(fig)
st.subheader('Impact of host identity on bookings')

# Prepare data
X = df[['host_identity_verified', 'price', 'bedrooms', 'bathrooms', 'accommodates']]

# Calculate median values for reviews and occupancy
median_reviews = df['number_of_reviews'].median()
median_occupancy = ((365 - df['availability_365']) / 365).median()

# Create binary target variables
y_reviews = (df['number_of_reviews'] > median_reviews).astype(int)
y_occupancy = ((365 - df['availability_365']) / 365 > median_occupancy).astype(int)

# Split data
X_train, X_test, y_reviews_train, y_reviews_test, y_occupancy_train, y_occupancy_test = train_test_split(
    X, y_reviews, y_occupancy, test_size=0.2, random_state=42)

# Create and train models
review_model = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(random_state=42))
])
review_model.fit(X_train, y_reviews_train)

occupancy_model = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(random_state=42))
])
occupancy_model.fit(X_train, y_occupancy_train)

# User input for prediction
st.subheader('Predict Impact of Host Verification')
is_verified = st.checkbox('Host is Verified')
price = st.number_input('Price', min_value=0, value=100, key = 'price_min')
bedrooms = st.number_input('Bedrooms', min_value=0, value=1, key = 'bedrooms_min')
bathrooms = st.number_input('Bathrooms', min_value=0.0, value=1.0, step=0.5, key = 'bathrooms_min')
accommodates = st.number_input('Accommodates', min_value=1, value=2, key="accomodates_min")

if st.button('Predict Impact'):
    # Ensure the input is correctly processed for prediction
    input_data = pd.DataFrame({
        'host_identity_verified': [int(is_verified)],  # Convert to integer
        'price': [price],
        'bedrooms': [bedrooms],
        'bathrooms': [bathrooms],
        'accommodates': [accommodates]
    })
    
    # Predict using the trained models
    review_prediction = review_model.predict_proba(input_data)[0][1]
    occupancy_prediction = occupancy_model.predict_proba(input_data)[0][1]
    
    # Display the results
    st.success(f'Probability of Above-Average Reviews: {review_prediction:.2f}')
    st.success(f'Probability of Above-Average Occupancy Rate: {occupancy_prediction:.2f}')

    # Interpret the results
    st.write("Interpretation:")
    if review_prediction > 0.5:
        st.write("- This listing is likely to receive above-average number of reviews.")
    else:
        st.write("- This listing is likely to receive below-average number of reviews.")
    
    if occupancy_prediction > 0.5:
        st.write("- This listing is likely to have above-average occupancy rate.")
    else:
        st.write("- This listing is likely to have below-average occupancy rate.")


#####################################################################################################################
# model developed - 7: (Using Gradient Boost)
# Problem : Host response time effect on availability and booking rates
st.header('7. Booking rate based on Host Response Time ')
st.subheader('Visualisation')
response_time_query = """
SELECT host_response_time, 
       AVG(availability_30) as avg_availability,
       AVG(30 - availability_30) as avg_bookings
FROM listings
WHERE host_response_time IS NOT NULL
GROUP BY host_response_time
"""
response_time_result = pd.read_sql_query(response_time_query, conn)
st.write(response_time_result)
fig = px.bar(response_time_result, x='host_response_time', y=['avg_availability', 'avg_bookings'], 
             title='Host Response Time Effect on Availability and Bookings')
st.plotly_chart(fig)
st.subheader('Impact of response time by host in booking rates')

# Prepare data
le = LabelEncoder()
df['host_response_time_encoded'] = le.fit_transform(df['host_response_time'])

features = ['host_response_time_encoded', 'price', 'accommodates', 'number_of_reviews', 'review_scores_rating']
X = df[features]
y_availability = df['availability_365']
y_bookings = df['number_of_reviews']

# Split data
X_train, X_test, y_avail_train, y_avail_test, y_book_train, y_book_test = train_test_split(
    X, y_availability, y_bookings, test_size=0.2, random_state=42)

# Train models
avail_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
avail_model.fit(X_train, y_avail_train)

booking_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
booking_model.fit(X_train, y_book_train)

# User input for prediction
# st.subheader('Predict Availability and Bookings')
response_time = st.selectbox('Host Response Time', df['host_response_time'].unique())
price = st.number_input('price', min_value=0, value=100)
accommodates = st.number_input('accommodates', min_value=1, value=2)
review_score = st.slider('review_score', 0.0, 5.0, 4.5)

if st.button('Predict Impact', key = 'predict_impact_button'):
    input_data = pd.DataFrame({
        'host_response_time_encoded': [le.transform([response_time])[0]],
        'price': [price],
        'accommodates': [accommodates],
        'number_of_reviews': [0],  # Set to 0 for new prediction
        'review_scores_rating': [review_score]
    })
    
    avail_prediction = avail_model.predict(input_data)
    booking_prediction = booking_model.predict(input_data)
    
    st.success(f'Predicted Availability (days/year): {avail_prediction[0]:.0f}')
    st.success(f'Predicted Number of Bookings: {booking_prediction[0]:.0f}')



####################################################################################################################
# Model developed- 8:(Using ADA Boost)
# Problem : Price effect on availability
st.header('8. Price prediction based on availability')
st.subheader('Visualisation')
price_query = """
SELECT 
    CASE 
        WHEN price < 50 THEN '0-50'
        WHEN price BETWEEN 50 AND 100 THEN '50-100'
        WHEN price BETWEEN 101 AND 150 THEN '101-150'
        WHEN price BETWEEN 151 AND 200 THEN '151-200'
        ELSE '200+'
    END as price_range,
    AVG(availability_30) as avg_availability
FROM listings
GROUP BY price_range
ORDER BY price_range
"""
price_result = pd.read_sql_query(price_query, conn)
st.write(price_result)
fig = px.line(price_result, x='price_range', y='avg_availability', 
              title='Average Availability by Price Range')
st.plotly_chart(fig)
st.subheader('Price prediction based on availability')

# Prepare data
features = ['price', 'accommodates', 'number_of_reviews', 'review_scores_rating', 'host_is_superhost', 'room_type']
X = df[features]
y = df['availability_365']

# Create preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', ['price', 'accommodates', 'number_of_reviews', 'review_scores_rating', 'host_is_superhost']),
        ('cat', OneHotEncoder(drop='first', sparse_output=False), ['room_type'])
    ])

# Create pipeline
model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', AdaBoostRegressor(n_estimators=100, random_state=42))
])

# Split data and train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)

# User input for prediction
st.subheader('Predict Availability Based on Price')
price = st.number_input('price', min_value=0, value=100, key='price_impact_p')
accommodates = st.number_input('accommodates', min_value=1, value=2, key='price_impact_A')
review_score = st.slider('review_score', 0.0, 5.0, 4.5, key='price_impact_r')
is_superhost = st.checkbox('is_superhost')
room_type = st.selectbox('room_type', df['room_type'].unique())

if st.button('Predict Availability', key = 'predict_availability_key'):
    input_data = pd.DataFrame({
        'price': [price],
        'accommodates': [accommodates],
        'number_of_reviews': [0],  # Set to 0 for new prediction
        'review_scores_rating': [review_score],
        'host_is_superhost': [int(is_superhost)],
        'room_type': [room_type]
    })
    
    prediction = model.predict(input_data)
    
    st.success(f'Predicted Availability (days/year): {prediction[0]:.0f}')

####################################################################################################################

conn.close()