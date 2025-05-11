import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt

def create_model(input_shape, num_gears=8):
    """Create a multi-task learning model for racing control."""
    # Input layer
    inputs = Input(shape=input_shape)
    
    # Shared layers
    x = Dense(256, activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    # Regression outputs (Steer, Accel, Brake)
    reg_output = Dense(3, activation='tanh', name='regression_output')(x)
    
    # Classification output (Gear)
    cls_output = Dense(num_gears, activation='softmax', name='classification_output')(x)
    
    # Create model
    model = Model(inputs=inputs, outputs=[reg_output, cls_output])
    
    return model

def train_model():
    # Load preprocessed data
    X_train = np.load("X_train.npy")
    y_reg_train = np.load("y_reg_train.npy")
    y_cls_train = np.load("y_cls_train.npy")
    X_val = np.load("X_val.npy")
    y_reg_val = np.load("y_reg_val.npy")
    y_cls_val = np.load("y_cls_val.npy")
    
    # Create and compile model
    model = create_model(input_shape=(X_train.shape[1],))
    
    # Compile with different loss weights for regression and classification
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss={
            'regression_output': 'mse',
            'classification_output': 'categorical_crossentropy'
        },
        loss_weights={
            'regression_output': 1.0,
            'classification_output': 0.5
        },
        metrics={
            'regression_output': ['mae'],
            'classification_output': ['accuracy']
        }
    )
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            'best_model.h5',
            monitor='val_loss',
            save_best_only=True,
            mode='min'
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )
    ]
    
    # Train model
    history = model.fit(
        X_train,
        [y_reg_train, y_cls_train],
        validation_data=(X_val, [y_reg_val, y_cls_val]),
        epochs=100,
        batch_size=64,
        callbacks=callbacks
    )
    
    # Plot training history
    plot_training_history(history)
    
    return model, history

def plot_training_history(history):
    """Plot training metrics."""
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    
    # Plot regression metrics
    ax1.plot(history.history['regression_output_loss'], label='Training Loss')
    ax1.plot(history.history['val_regression_output_loss'], label='Validation Loss')
    ax1.set_title('Regression Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    # Plot classification metrics
    ax2.plot(history.history['classification_output_accuracy'], label='Training Accuracy')
    ax2.plot(history.history['val_classification_output_accuracy'], label='Validation Accuracy')
    ax2.set_title('Classification Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

def evaluate_model(model):
    """Evaluate model on validation set."""
    X_val = np.load("X_val.npy")
    y_reg_val = np.load("y_reg_val.npy")
    y_cls_val = np.load("y_cls_val.npy")
    
    # Evaluate model
    results = model.evaluate(X_val, [y_reg_val, y_cls_val])
    
    print("\nValidation Results:")
    print(f"Regression Loss: {results[1]:.4f}")
    print(f"Classification Loss: {results[2]:.4f}")
    print(f"Regression MAE: {results[3]:.4f}")
    print(f"Classification Accuracy: {results[4]:.4f}")

if __name__ == "__main__":
    # Train model
    model, history = train_model()
    
    # Evaluate model
    evaluate_model(model) 