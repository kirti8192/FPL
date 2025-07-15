import pandas as pd
import matplotlib.pyplot as plt
import os

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, auc

from tensorflow import keras 
from tensorflow.keras import layers, models


def preprocess_values(df_this):
    """
    This function handles NaN values and converts object columns to numeric types.
    """

    # handle NaN values
    df_this = df_this.fillna(0)  # Replace NaNs with 0s
    
    # handle categorical values
    categorical_cols = list(df_this.select_dtypes(include=['object']).columns)

    # one-hot encode categorical columns
    myOneHotEncoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    df_encoded = pd.DataFrame(myOneHotEncoder.fit_transform(df_this[categorical_cols]))

    # Set the column names to the one-hot encoded feature names
    df_encoded.columns = myOneHotEncoder.get_feature_names_out(categorical_cols)
    df_encoded.index = df_this.index  # Ensure the index matches the original DataFrame

    # drop original categorical columns
    df_this = df_this.drop(columns=categorical_cols, axis=1)

    # concatenate the one-hot encoded DataFrame with the original DataFrame
    df_concat = pd.concat([df_encoded, df_this], axis=1)

    return df_concat

def get_target_column(df_this, gw):
    """
    Returns the boolean 
    """
    target_col = f'total_points_gw{gw}'
    return df_this[target_col] > 4  # threshold for next gameweek points to be considered as a good performance

def get_df_for_gw(df_this, gw):
    """
    Returns a DataFrame with all data for the gameweek just before the one that is targeted.
    """
    if gw == 1:
        raise ValueError("Gameweek 1 does not have a previous gameweek to reference.")
    if gw > 38:
        raise ValueError("Gameweek must be between 1 and 38.")
    
    # get the columns for the gameweek just before the one that is targeted
    static_cols_to_keep = [col for col in df_this.columns if "_gw" not in col]
    gw_suffixes = [f"_gw{idx}" for idx in range(1, gw)]
    gw_cols_to_keep = [col for col in df_this.columns for suffix in gw_suffixes if col.endswith(suffix) ]

    # Filter the DataFrame to keep only the desired columns
    cols_to_keep = static_cols_to_keep + gw_cols_to_keep
    df_filtered = df_this[cols_to_keep]

    # get target column
    df_target_col = get_target_column(df_this, gw)

    # merge df_filtered with the target column
    df_filtered = df_filtered.merge(df_target_col.rename('target'), left_index=True, right_index=True)

    return df_filtered

def get_df():
    """
    Returns a DataFrame with all data for the 2022–23 Fantasy Premier League season.
    The DataFrame contains aggregated statistics for each player across all gameweeks.
    """

    # read the data
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../data/raw'))
    csv_path = os.path.join(base_dir, 'vaastav_2022_23.csv')
    df = pd.read_csv(csv_path)

    # extract team and position information
    df_common = df.groupby('element').agg({'team': 'first',
                                             'position': 'first'
                                             }).reset_index()

    # select features
    cols_to_keep = ['minutes',
                    'goals_scored', 
                    'assists', 
                    # 'expected_goals',     # all zeros
                    # 'expected_assists',   # all zeros
                    'clean_sheets',
                    'ict_index',
                    'bps', 
                    'bonus', 
                    'total_points',
                    ]

    # pivot table to get unified dataframe
    df_multigw = df.pivot_table(index='element', 
                    columns = 'GW', 
                    values = cols_to_keep, 
                    aggfunc='sum').reset_index()

    # flatten columns with gameweek suffix
    df_multigw.columns = [f"{col}_gw{int(gw)}" if isinstance(gw, (int,float)) else col for col,gw in df_multigw.columns]

    # merge common information with gamew data
    df_multigw = df_common.merge(df_multigw, on='element', how='left')

    # set element as the index
    df_multigw.set_index('element', inplace=True)

    return df_multigw

if __name__ == '__main__':

    df = get_df()

    # extract just data to predict gw X
    gw = 35
    df_gw = get_df_for_gw(df, gw)

    # value preprocess
    df_gw = preprocess_values(df_gw)

    # extract X and y
    y = df_gw.target
    X = df_gw.drop(columns=['target'], axis=1)

    # train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

    # random forest binary classifier
    forest_model = RandomForestClassifier(n_estimators=100, random_state=0)
    forest_model.fit(X_train, y_train)
    print("Model trained successfully.")

    forest_prediction = forest_model.predict(X_test)
    print("Predictions made successfully.")

    forest_accuracy = forest_model.score(X_test, y_test)
    print(f"Model accuracy: {forest_accuracy:.2f}")

    y_pred_rf = forest_model.predict(X_test)

    print(classification_report(y_test, y_pred_rf))

    cm = confusion_matrix(y_test, y_pred_rf)
    ConfusionMatrixDisplay(cm).plot()

    importances = forest_model.feature_importances_
    feature_names = X_train.columns

    feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)

    plt.figure(figsize=(10,6))
    feat_imp.head(20).plot(kind='barh')
    plt.title("Top 20 Feature Importances - Random Forest")
    plt.show()

    # neural network binary classifier
    nn_model = models.Sequential([
        layers.Input(shape=(X_train.shape[1],)),

        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),

        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),

        layers.Dense(8, activation='relu'),
        layers.Dropout(0.2),

        layers.Dense(1, activation='sigmoid')
    ])

    nn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy'])
    history = nn_model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)
    nn_prediction = nn_model.predict(X_test)
    print("Neural network predictions made successfully.")

    nn_accuracy = nn_model.evaluate(X_test, y_test, verbose=0)[1]
    print(f"Neural network accuracy: {nn_accuracy:.2f}")

    plt.figure(figsize=(12,5))

    # Accuracy
    plt.subplot(1,2,1)
    plt.plot(history.history['binary_accuracy'], label='train_accuracy')
    plt.plot(history.history['val_binary_accuracy'], label='val_accuracy')
    plt.legend()
    plt.title('Training vs Validation Accuracy')

    # Loss
    plt.subplot(1,2,2)
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.legend()
    plt.title('Training vs Validation Loss')

    plt.show()

    # Random Forest probabilities
    y_probs_rf = forest_model.predict_proba(X_test)[:,1]

    # Neural Network probabilities
    y_probs_nn = nn_model.predict(X_test).ravel()

    # ROC Curves
    fpr_rf, tpr_rf, _ = roc_curve(y_test, y_probs_rf)
    fpr_nn, tpr_nn, _ = roc_curve(y_test, y_probs_nn)

    auc_rf = auc(fpr_rf, tpr_rf)
    auc_nn = auc(fpr_nn, tpr_nn)

    plt.figure(figsize=(8,6))
    plt.plot(fpr_rf, tpr_rf, label=f"RF AUC = {auc_rf:.2f}")
    plt.plot(fpr_nn, tpr_nn, label=f"NN AUC = {auc_nn:.2f}")
    plt.plot([0,1], [0,1], 'k--')
    plt.legend()
    plt.title("ROC Curves")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.show()