import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ML Imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error

# TensorFlow/Keras Imports
import tensorflow as tf
from tensorflow import keras

def main():
    """
    This script demonstrates how to load a CSV file of IMU + potentiometer data,
    select only the necessary columns, and train two regression models (Random Forest
    and a Neural Network) to predict all six pose parameters: x, y, z, roll, pitch, and yaw.
    """

    # ---------------------------
    # 1. LOAD & FILTER THE DATA
    # ---------------------------
    data_path = "simulated_datax.csv"
  
    # Load full dataset to include estimated pose columns for later comparison
    full_df = pd.read_csv(data_path)
    
    # Columns we actually need for sensor fusion - now including position data
    columns_to_use = [
        "IMU roll", "IMU pitch", "IMU yaw",
        "Pot 1 distance", "Pot 2 distance", "Pot 3 distance",
        "Pot 4 distance", "Pot 5 distance", "Pot 6 distance",
        "Actual pose x", "Actual pose y", "Actual pose z",
        "Actual pose roll", "Actual pose pitch", "Actual pose yaw"
    ]

    print("Loading data from:", data_path)
    df = full_df[columns_to_use].copy()

    print("Data loaded. Columns in use:")
    print(df.columns)
    print("First 5 rows:\n", df.head(), "\n")

    # -------------------------------------
    # 2. SPLIT INTO FEATURES & TARGET (y)
    # -------------------------------------
    feature_cols = [
        "IMU roll", "IMU pitch", "IMU yaw",
        "Pot 1 distance", "Pot 2 distance", "Pot 3 distance",
        "Pot 4 distance", "Pot 5 distance", "Pot 6 distance"
    ]
    # Updated to include position parameters
    target_cols = [
        "Actual pose x", "Actual pose y", "Actual pose z",
        "Actual pose roll", "Actual pose pitch", "Actual pose yaw"
    ]
    estimated_cols = [
        "Estimated pose x", "Estimated pose y", "Estimated pose z",
        "Estimated pose roll", "Estimated pose pitch", "Estimated pose yaw"
    ]
    
    X = df[feature_cols].values  # sensor inputs
    y = df[target_cols].values  # pose parameters to predict (position + orientation)
    
    # Create a mapping from original indices to keep track for comparison later
    indices = np.arange(len(df))

    # ---------------------------------------------
    # 3. TRAIN/TEST SPLIT + OPTIONAL SCALING
    # ---------------------------------------------
    X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(
        X, y, indices, test_size=0.2, random_state=42
    )

    # Get estimated pose values for the test set
    y_estimated = full_df[estimated_cols].values[indices_test]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"Train set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")

    # --------------------------
    # 4A. RANDOM FOREST MODEL
    # --------------------------
    print("\n===== RANDOM FOREST REGRESSOR =====")
    # Use MultiOutputRegressor to handle multiple targets
    base_rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model = MultiOutputRegressor(base_rf)
    rf_model.fit(X_train_scaled, y_train)

    y_pred_rf = rf_model.predict(X_test_scaled)
    
    # Calculate MSE for each output dimension
    mse_rf_individual = mean_squared_error(y_test, y_pred_rf, multioutput='raw_values')
    mse_rf_avg = mean_squared_error(y_test, y_pred_rf)
    print(f"Random Forest Test MSE (Average): {mse_rf_avg}")
    # Print individual MSEs for all 6 pose parameters
    parameter_names = ["X", "Y", "Z", "Roll", "Pitch", "Yaw"]
    for i, param in enumerate(parameter_names):
        print(f"Random Forest Test MSE ({param}): {mse_rf_individual[i]}")


    # Optional: Compare predictions vs. actual visually for each parameter
    plt.figure(figsize=(20, 10))
    for i in range(6):  # Now 6 parameters (x, y, z, roll, pitch, yaw)
        plt.subplot(2, 3, i+1)
        plt.scatter(y_test[:, i], y_pred_rf[:, i], alpha=0.5, label=f"RF {parameter_names[i]}")
        plt.plot([y_test[:, i].min(), y_test[:, i].max()], 
                [y_test[:, i].min(), y_test[:, i].max()],
                'r--', label="Ideal")
        plt.xlabel(f"True {parameter_names[i]}")
        plt.ylabel(f"Predicted {parameter_names[i]}")
        plt.title(f"Random Forest - {parameter_names[i]} Predictions")
        plt.legend()
    plt.tight_layout()
    plt.show()

    # --------------------------
    # 4B. NEURAL NETWORK MODEL
    # --------------------------
    print("\n===== NEURAL NETWORK (TensorFlow/Keras) =====")
    nn_model = keras.Sequential([
        keras.layers.Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dense(6)  # six outputs for x, y, z, roll, pitch, yaw
    ])

    nn_model.compile(optimizer='adam', loss='mean_squared_error')
    nn_model.summary()

    # Train the NN
    history = nn_model.fit(
        X_train_scaled, y_train,
        validation_split=0.2,
        epochs=100,
        batch_size=32,
        verbose=1,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            )
        ]
    )

    # Evaluate on test set
    test_loss = nn_model.evaluate(X_test_scaled, y_test, verbose=0)
    print("Neural Network Test MSE (Average):", test_loss)

    # Predictions vs. actual
    y_pred_nn = nn_model.predict(X_test_scaled)
    
    # Calculate MSE for each output dimension for NN
    mse_nn_individual = mean_squared_error(y_test, y_pred_nn, multioutput='raw_values')
    for i, param in enumerate(parameter_names):
        print(f"Neural Network Test MSE ({param}): {mse_nn_individual[i]}")

    # Plot learning curves
    plt.figure(figsize=(10, 4))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.title('Neural Network Learning Curves')
    plt.tight_layout()
    plt.show()

    # Visualize NN predictions
    plt.figure(figsize=(20, 10))
    for i in range(6):  # Now 6 parameters
        plt.subplot(2, 3, i+1)
        plt.scatter(y_test[:, i], y_pred_nn[:, i], alpha=0.5, label=f"NN {parameter_names[i]}")
        plt.plot([y_test[:, i].min(), y_test[:, i].max()], 
                [y_test[:, i].min(), y_test[:, i].max()],
                'r--', label="Ideal")
        plt.xlabel(f"True {parameter_names[i]}")
        plt.ylabel(f"Predicted {parameter_names[i]}")
        plt.title(f"Neural Network - {parameter_names[i]} Predictions")
        plt.legend()
    plt.tight_layout()
    plt.show()
    
    # --------------------------
    # 5. COMPARISON WITH ESTIMATED POSE
    # --------------------------
    print("\n===== COMPARING ML FUSION VS. EXISTING ESTIMATOR =====")

    # Calculate MSE for the built-in estimator on the test set
    mse_estimator = mean_squared_error(y_test, y_estimated, multioutput='raw_values')
    print(f"Built-in Estimator - MSE (Average): {mean_squared_error(y_test, y_estimated)}")
    for i, param in enumerate(parameter_names):
        print(f"Built-in Estimator - MSE ({param}): {mse_estimator[i]}")
    
    # Calculate percentage improvement
    rf_improvement = ((mse_estimator - mse_rf_individual) / mse_estimator) * 100
    nn_improvement = ((mse_estimator - mse_nn_individual) / mse_estimator) * 100

    for i, param in enumerate(parameter_names):
        print(f"Random Forest improvement over built-in estimator ({param}): {rf_improvement[i]:.2f}%")
    for i, param in enumerate(parameter_names):
        print(f"Neural Network improvement over built-in estimator ({param}): {nn_improvement[i]:.2f}%")

    # Visualize the comparison - Bar chart of MSE values
    plt.figure(figsize=(15, 8))
    x = np.arange(len(parameter_names))
    width = 0.25

    plt.bar(x - width, mse_estimator, width, label='Built-in Estimator')
    plt.bar(x, mse_rf_individual, width, label='Random Forest')
    plt.bar(x + width, mse_nn_individual, width, label='Neural Network')

    plt.xticks(x, parameter_names)
    plt.ylabel('Mean Squared Error')
    plt.title('Error Comparison: Built-in Estimator vs. ML Fusion')
    plt.legend()
    plt.yscale('log')  # Log scale often helps visualize error differences
    plt.tight_layout()
    plt.savefig('mse_comparison_all_params.png')  # Save the figure
    plt.show()

    # Separate position and orientation errors for clearer visualization
    plt.figure(figsize=(15, 10))
    
    # Position errors (x, y, z)
    plt.subplot(2, 1, 1)
    x_pos = np.arange(3)
    plt.bar(x_pos - width, mse_estimator[:3], width, label='Built-in Estimator')
    plt.bar(x_pos, mse_rf_individual[:3], width, label='Random Forest')
    plt.bar(x_pos + width, mse_nn_individual[:3], width, label='Neural Network')
    plt.xticks(x_pos, parameter_names[:3])
    plt.ylabel('Mean Squared Error')
    plt.title('Position Error Comparison (X, Y, Z)')
    plt.legend()
    plt.yscale('log')
    
    # Orientation errors (roll, pitch, yaw)
    plt.subplot(2, 1, 2)
    x_ori = np.arange(3)
    plt.bar(x_ori - width, mse_estimator[3:], width, label='Built-in Estimator')
    plt.bar(x_ori, mse_rf_individual[3:], width, label='Random Forest')
    plt.bar(x_ori + width, mse_nn_individual[3:], width, label='Neural Network')
    plt.xticks(x_ori, parameter_names[3:])
    plt.ylabel('Mean Squared Error')
    plt.title('Orientation Error Comparison (Roll, Pitch, Yaw)')
    plt.legend()
    plt.yscale('log')
    
    plt.tight_layout()
    plt.savefig('mse_comparison_position_orientation.png')
    plt.show()

    # Add direct comparison plots for one position parameter (e.g., x) and one orientation parameter (e.g., pitch)
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # For position (x)
    pos_idx = 0  # x position
    plot_size = min(100, len(y_test))
    x_indices = np.arange(plot_size)
    
    axes[0].plot(x_indices, y_test[:plot_size, pos_idx], 'k-', label='Actual')
    axes[0].plot(x_indices, y_estimated[:plot_size, pos_idx], 'r--', label='Built-in Estimator')
    axes[0].plot(x_indices, y_pred_rf[:plot_size, pos_idx], 'b--', label='Random Forest')
    axes[0].plot(x_indices, y_pred_nn[:plot_size, pos_idx], 'g--', label='Neural Network')
    axes[0].set_xlabel('Sample Index')
    axes[0].set_ylabel('X Position')
    axes[0].set_title('Comparison of X Position Predictions')
    axes[0].legend()
    axes[0].grid(True)
    
    # For orientation (pitch)
    angle_idx = 4  # pitch (index 4 after including x,y,z)
    axes[1].plot(x_indices, y_test[:plot_size, angle_idx], 'k-', label='Actual')
    axes[1].plot(x_indices, y_estimated[:plot_size, angle_idx], 'r--', label='Built-in Estimator')
    axes[1].plot(x_indices, y_pred_rf[:plot_size, angle_idx], 'b--', label='Random Forest')
    axes[1].plot(x_indices, y_pred_nn[:plot_size, angle_idx], 'g--', label='Neural Network')
    axes[1].set_xlabel('Sample Index')
    axes[1].set_ylabel('Pitch (degrees)')
    axes[1].set_title('Comparison of Pitch Predictions')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig('position_orientation_comparison.png')
    plt.show()

    # Create scatter plots comparing actual vs. predicted values for all parameters
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, param in enumerate(parameter_names):
        axes[i].scatter(y_test[:, i], y_estimated[:, i], alpha=0.5, label='Built-in Estimator')
        axes[i].scatter(y_test[:, i], y_pred_rf[:, i], alpha=0.5, label='Random Forest')
        axes[i].scatter(y_test[:, i], y_pred_nn[:, i], alpha=0.5, label='Neural Network')
        
        # Add ideal line
        min_val = min(y_test[:, i].min(), y_estimated[:, i].min(), y_pred_rf[:, i].min(), y_pred_nn[:, i].min())
        max_val = max(y_test[:, i].max(), y_estimated[:, i].max(), y_pred_rf[:, i].max(), y_pred_nn[:, i].max())
        axes[i].plot([min_val, max_val], [min_val, max_val], 'k--')
        
        axes[i].set_xlabel(f'Actual {param}')
        axes[i].set_ylabel(f'Predicted {param}')
        axes[i].set_title(f'{param} Predictions vs. Actual')
        axes[i].legend()
        axes[i].grid(True)

    plt.tight_layout()
    plt.savefig('scatter_comparison_all_params.png')
    plt.show()
    
    # Save the models
    import joblib
    joblib.dump(rf_model, 'rf_pose_model.joblib')
    joblib.dump(scaler, 'scaler.joblib')
    nn_model.save('nn_pose_model.keras')
    
    print("Full dataset shape:", df.shape)
    print("\nModels saved! Random Forest and Neural Network models are ready for deployment.")
    print("\nScript complete! Review the plots to see model performance.")


if __name__ == "__main__":

    main()
