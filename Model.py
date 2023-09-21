import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow import keras
from keras.layers import Dense
from kerastuner.tuners import RandomSearch

# Load the CSV file
data = pd.read_csv('cheat.csv')

# Split the data into input features and labels
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values
y[y == 'cheat'] = 1.0
y[y == 'non-cheat'] = 0.0
y = y.astype(float)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def build_model(hp):
    model = keras.Sequential()

    # Tune the number of units in the dense layers
    hp_units = hp.Int('units', min_value=64, max_value=256, step=64)
    model.add(Dense(units=hp_units, activation='relu', input_shape=(2,)))

    # Tune the number of hidden layers
    hp_layers = hp.Int('layers', min_value=1, max_value=5, step=1)
    for _ in range(hp_layers):
        model.add(Dense(units=hp_units, activation='relu'))

    model.add(Dense(1, activation='sigmoid'))

    # Compile the model
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model

# Define the search space
tuner = RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=5,  # Number of different models to try
    executions_per_trial=1,  # Number of executions per model
    directory='my_tuner_dir',  # Directory to store the tuning results
    project_name='my_mlp_tuner'  # Name of the tuning project
)

# Start the hyperparameter search
tuner.search(X_train, y_train, validation_data=(X_test, y_test), epochs=10)

# Get the best model
best_model = tuner.get_best_models(num_models=5)[0]

# Evaluate each of the 5 trained models and save them
for i, model in enumerate(tuner.get_best_models(num_models=5)):
    model_name = f'model_{i + 1}.h5'
    
    # Evaluate the model on the test set
    y_pred = model.predict(X_test)
    y_pred_classes = (y_pred > 0.5).astype(int)

    # Get classification report and confusion matrix
    print(f"Classification Report for Model {i + 1}:")
    print(classification_report(y_test, y_pred_classes))

    print(f"Confusion Matrix for Model {i + 1}:")
    print(confusion_matrix(y_test, y_pred_classes))
    
    # Save the model
    model.save(model_name)
    print(f"Model {model_name} saved.")

