import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import OneHotEncoder

# Sample dataset for destinations in Northern Pakistan
data = {
    'Destination': ['Hunza', 'Skardu', 'Fairy Meadows', 'Naltar Valley', 'Naran', 'Khunjerab Pass', 'Swat Valley', 'Ratti Gali Lake'],
    'Type': ['Adventure', 'Adventure', 'Nature', 'Nature', 'Adventure', 'Sightseeing', 'Cultural', 'Nature'],
    'Best_Season': ['Summer', 'Summer', 'Spring', 'Summer', 'Spring', 'Summer', 'Autumn', 'Summer'],
    'Budget': ['Medium', 'High', 'Medium', 'Low', 'Medium', 'High', 'Low', 'Medium'],
    'Family_Friendly': [True, True, False, True, True, True, True, True],
    'Girls_Travel_Friendly': [True, True, True, True, True, True, True, True]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# OneHotEncoder to encode 'Type', 'Best_Season', and 'Budget'
encoder = OneHotEncoder()

encoded_features = encoder.fit_transform(df[['Type', 'Best_Season', 'Budget']]).toarray()

# Convert the encoded features to a DataFrame
encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out())

# Merge encoded features back to the original DataFrame
df_encoded = pd.concat([df, encoded_df], axis=1)

# Drop original columns that are now encoded
df_encoded.drop(['Type', 'Best_Season', 'Budget'], axis=1, inplace=True)

# Function to recommend destinations based on user preferences
def recommend_destinations(user_prefs, df, encoded_df):
    # Encode user preferences to match the format of encoded_df
    user_prefs_encoded = encoder.transform([[user_prefs['Type'], user_prefs['Best_Season'], user_prefs['Budget']]]).toarray()

    # Create a DataFrame from the encoded user preferences
    user_vector = pd.DataFrame(user_prefs_encoded, columns=encoder.get_feature_names_out())

    # Add family and girls travel preferences as they are already boolean
    user_vector['Family_Friendly'] = int(user_prefs['Family_Friendly'])
    user_vector['Girls_Travel_Friendly'] = int(user_prefs['Girls_Travel_Friendly'])

    # Calculate cosine similarity between user preferences and all destinations
    similarity_scores = cosine_similarity(df_encoded.drop('Destination', axis=1), user_vector)

    # Add similarity scores to the DataFrame
    df['Similarity_Score'] = similarity_scores.flatten()

    # Sort by similarity score in descending order and return top recommendations
    recommendations = df.sort_values(by='Similarity_Score', ascending=False)

    return recommendations[['Destination', 'Similarity_Score']]

# Streamlit UI elements for user inputs
def streamlit_app():
    st.title("North Pakistan Destination Recommender")

    # Dropdown for destination type
    user_type = st.selectbox(
        "What type of destination are you interested in?",
        options=['Adventure', 'Nature', 'Sightseeing', 'Cultural']
    )

    # Dropdown for best season
    user_season = st.selectbox(
        "What season would you like to travel in?",
        options=['Summer', 'Spring', 'Autumn']
    )

    # Dropdown for budget
    user_budget = st.selectbox(
        "What is your budget range?",
        options=['Low', 'Medium', 'High']
    )

    # Radio button for family-friendly preference
    user_family_friendly = st.radio(
        "Are you traveling with family?",
        options=['Yes', 'No']
    )
    user_family_friendly = True if user_family_friendly == 'Yes' else False

    # Radio button for girls-travel-friendly preference
    user_girls_travel_friendly = st.radio(
        "Do you prefer girls-travel-friendly destinations?",
        options=['Yes', 'No']
    )
    user_girls_travel_friendly = True if user_girls_travel_friendly == 'Yes' else False

    # User preferences stored in a dictionary
    user_preferences = {
        'Type': user_type,
        'Best_Season': user_season,
        'Budget': user_budget,
        'Family_Friendly': user_family_friendly,
        'Girls_Travel_Friendly': user_girls_travel_friendly
    }

    # Button to get recommendations
    if st.button("Get Recommendations"):
        recommendations = recommend_destinations(user_preferences, df, df_encoded)
        st.write("Here are the top destinations recommended for you:")
        st.table(recommendations.head())

# Run the Streamlit app
if __name__ == "__main__":
    streamlit_app()
