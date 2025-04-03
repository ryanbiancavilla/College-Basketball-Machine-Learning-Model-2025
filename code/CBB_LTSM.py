from sportsdataverse.mbb import mbb_loaders
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score

seasons = range(2004, 2026)
all_data = []

for season in seasons:
    try:
        data = mbb_loaders.load_mbb_team_boxscore(seasons=[season], return_as_pandas=True)
        all_data.append(data)
    except Exception as e:
        print(f"Error loading data for season {season}: {e}")

mbb_df = pd.concat(all_data, ignore_index=True)

columns_to_remove = [
    "team_uid", "team_color", "team_alternate_color", "team_logo", "opponent_team_uid", 
    "opponent_team_color", "opponent_team_alternate_color", "opponent_team_logo", "team_slug", 
    "team_name", "team_abbreviation", "team_display_name", "team_short_display_name", "opponent_team_slug", 
    "opponent_team_name", "opponent_team_abbreviation", "opponent_team_display_name", 
    "opponent_team_short_display_name", "opponent_team_id", "opponent_team_location", "opponent_team_score"]

mbb_df = mbb_df.drop(columns=columns_to_remove)

mbb_df['team_home_away'] = mbb_df['team_home_away'].map({'home': 1, 'away': 0})

def concatenate_team_stats(mbb_df):
    concatenated_df = mbb_df.merge(
        mbb_df, 
        on='game_id', 
        suffixes=('', '_opp')
    )
    concatenated_df = concatenated_df[
        concatenated_df['team_location'] != concatenated_df['team_location_opp']
    ]
    new_column_names = {
        col: col if '_opp' not in col else 'opp_' + col.replace('_opp', '')
        for col in concatenated_df.columns
    }
    concatenated_df.rename(columns=new_column_names, inplace=True)
    
    return concatenated_df

concatenated_stats = concatenate_team_stats(mbb_df)
columns_to_remove = ['opp_season_type','opp_season','opp_game_date','opp_game_date_time']
concatenated_stats = concatenated_stats.drop(columns=columns_to_remove)
concatenated_stats = concatenated_stats.sort_values("game_date").reset_index(drop=True)

def add_target(team_location):
    team_location["target"] = team_location["team_winner"].shift(-1)
    return team_location

concatenated_stats = concatenated_stats.groupby("team_location", group_keys=False).apply(add_target)
concatenated_stats['target'] = concatenated_stats['target'].astype('float')
concatenated_stats["target"].fillna(2, inplace=True)
concatenated_stats["target"] = concatenated_stats["target"].astype(int)

nulls = pd.isnull(concatenated_stats).sum()
valid_columns = concatenated_stats.columns[~concatenated_stats.columns.isin(nulls.index)]
concatenated_stats = concatenated_stats[valid_columns].copy()

removed_columns = ['game_id', 'season', 'season_type', 'game_date', 'game_date_time',
                   'team_id', 'team_location', 'team_home_away', 'opp_team_id', 
                   'opp_team_location', 'opp_team_home_away', 'team_winner', 'opp_team_winner', "target"]
selected_columns = concatenated_stats.columns[~concatenated_stats.columns.isin(removed_columns)]

scaler = MinMaxScaler()
concatenated_stats[selected_columns] = scaler.fit_transform(concatenated_stats[selected_columns])

# Preparing data for LSTM
sequence_length = 10  # Number of past games to consider per sample
features = concatenated_stats[selected_columns].values
targets = concatenated_stats["target"].values

generator = TimeseriesGenerator(features, targets, length=sequence_length, batch_size=32)

# Splitting into training and test sets
split = int(len(generator) * 0.8)
train_generator = generator[:split]
test_generator = generator[split:]

# Building LSTM Model
model = Sequential([
    LSTM(50, activation='relu', return_sequences=True, input_shape=(sequence_length, len(selected_columns))),
    Dropout(0.2),
    LSTM(25, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Training the model
model.fit(train_generator, epochs=20, validation_data=test_generator)

# Making predictions
y_pred = model.predict(test_generator)
y_pred = np.round(y_pred).flatten()

test_targets = np.array([targets[i + sequence_length] for i in range(split, len(targets) - sequence_length)])
accuracy = accuracy_score(test_targets, y_pred)
print("LSTM Model Accuracy:", accuracy)
