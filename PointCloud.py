#%%
"""
Enhanced EventClassifier: A PointNet-based classifier for particle physics events
with improved training performance and generalization for the X-17 boson search experiment.

Fixed version that works without custom F1Score metrics.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, f1_score
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import random

#%%
# Bad Event Generation
def corrupt_event(event_df):
    mode = 'distort'
    distortion_scale = 0.15
    if mode == 'zero':
        """
        Randomly zero out a chunk of rows within an event.
        """
        if len(event_df) <= 5:
            return event_df.copy()  # Skip tiny events

        # Decide how much of the event to zero (e.g., 10–50%)
        # For missing information algorithm, threshold is about 15%
        # Try with distortion instead of missing events next. Once these methods are refined, distort the full dataset and train with different.
        # current distortion algorithm threshold is around 30%
        # try distorction with high event fraction and low distortion scale.
        frac_to_zero = np.random.uniform(0.05, 0.15)
        n_zero = int(len(event_df) * frac_to_zero)

        # Randomly pick which rows to zero out
        rows_to_zero = np.random.choice(event_df.index, size=n_zero, replace=False)

        # Copy to avoid modifying original
        corrupted_df = event_df.copy()

        # Set x, y, z to 0 in selected rows
        corrupted_df.loc[rows_to_zero, ['x', 'y', 'z']] = 0
    if mode == 'distort':
        if len(event_df) <= 5:
            return event_df.copy()

        # Fraction to distort (same logic as before)
        frac_to_distort = np.random.uniform(0.5, 0.55)
        n_distort = int(len(event_df) * frac_to_distort)

        # Randomly select rows to distort
        rows_to_distort = np.random.choice(event_df.index, size=n_distort, replace=False)

        # Copy to avoid modifying original
        corrupted_df = event_df.copy()

        # Apply distortion
        for axis in ['x', 'y', 'z']:
            original_values = corrupted_df.loc[rows_to_distort, axis]
            noise = original_values * np.random.uniform(-distortion_scale, distortion_scale, size=n_distort)
            corrupted_df.loc[rows_to_distort, axis] = original_values + noise

    return corrupted_df
class EnhancedEventClassifier:
    def __init__(self, num_points=1024, num_classes=2):
        """
        Initialize the Enhanced Event Classifier.

        Parameters:
            num_points (int): Number of points to sample from each event
            num_classes (int): Number of classes to predict (2 for binary classification)
        """
        self.num_points = num_points
        self.num_classes = num_classes
        self.model = None
        self.label_encoder = LabelEncoder()
       
        # Define the categories and their corresponding event numbers
        # How do I set the events for my data set since they are all good events?
        # For our data type call everything one class. Category "seagulls". 
        # Add Gaussian function with some width to add random distribution for artificial bad events
        # Take a sample of about 1000 events from the full data set to use as the training set
        # Load and filter the raw dataset
        columns = ["a", "tb", "x", "y", "z", "q"]
        file_path = "/home/jak689/3D-ParticlePlot-Main/CompleteData.txt"
        dfFull = pd.read_csv(file_path, delim_whitespace=True, names=columns, skiprows=1)

        # Clean data
        for col in ["a", "x", "y", "z"]:
            dfFull[col] = pd.to_numeric(dfFull[col], errors='coerce')
        dfFull = dfFull.dropna(subset=["a", "x", "y", "z"])

        # Filter to set number of events
        df = dfFull[dfFull['a'] < 10000].copy()
        df['a'] = df['a'].astype(int)
        self.dfFull = dfFull  # Just in case you want access to full data
        self.df = df           # Original filtered data

        # Generate "bad" events
        unique_bad_events = np.random.choice(df['a'].unique(), size=3000, replace=False)
        self.unique_bad_events = unique_bad_events.astype(int)
        good_events = np.setdiff1d(df['a'].unique(), self.unique_bad_events)

        # Create corrupted versions of bad events
        corrupted_parts = []
        for event_id in self.unique_bad_events:
            event_df = df[df['a'] == event_id]
            corrupted_df = corrupt_event(event_df)
            corrupted_parts.append(corrupted_df)

        # Recombine to get final working dataset
        untouched_df = df[~df['a'].isin(self.unique_bad_events)]
        final_df = pd.concat([untouched_df] + corrupted_parts)
        final_df = final_df.sort_index().reset_index(drop=True)

        # Save to instance variable
        self.final_df = final_df
        self.categories = {
            'seagulls': good_events,           # Good events
            #'triplets': [25, 41, 42, 63, 81, 91, 93],   # Special pattern events
            #'seagullish': [1, 7, 18, 19, 21, 23, 29, 36, 69, 78, 82, 84, 87, 88, 89],  # Similar to seagulls
            #'low_and_double_single': [46, 57, 62, 67, 71, 73, 77]  # Low energy or double/single events
        }
       
        # For binary classification, only seagulls are considered "good"
        self.good_events = set(self.categories['seagulls'])
       
        # Set seeds for reproducibility
        self.set_seeds()

    def set_seeds(self, seed=42):
        """Set random seeds for reproducibility."""
        np.random.seed(seed)
        tf.random.set_seed(seed)
        random.seed(seed)
       
    def load_and_preprocess_data(self, file_path, binary=True, augmentation_factor=10):
        """
        Load and preprocess the event data from file.

        Parameters:
            file_path (str): Path to the data file
            binary (bool): If True, use binary classification (good/bad)
            augmentation_factor (int): Number of augmented versions to create for each good event

        Returns:
            tuple: (point_clouds, labels)
        """
        # Step 1: Load the raw data
        # a: Event number
        # b: not used
        # c: not used
        # xyz: position data
        # tb: timestamps (not currently used)
        # q: not used
        if not hasattr(self, 'final_df'):
            raise ValueError("final_df not found. Make sure corruption was applied in __init__().")

        df = self.final_df.copy()
       
        # Step 2: Clean and convert numeric data
        #numeric_columns = ["a", "x", "y", "z"]
        #for col in numeric_columns:
            #dfFull[col] = pd.to_numeric(dfFull[col], errors='coerce')
       
        # Drop rows with missing values in essential columns
        #dfFull = dfFull.dropna(subset=["a", "x", "y", "z"])

        # Step 3: Assign labels based on classification mode
        if binary:
            # For binary: 'good' if in seagulls category, 'bad' otherwise
            df['label'] = df['a'].apply(lambda x: 'good' if x in self.good_events else 'bad')
        else:
            # For multi-class: assign actual category names
            df['label'] = df['a'].apply(self._get_event_category)
       
        # Step 4: Process each event into a point cloud
        event_point_clouds = []
        event_labels = []
        event_ids = []  # Track original IDs for later reference
       
        # Extract unique event IDs
        unique_events = df['a'].unique()
       
        print(f"Processing {len(unique_events)} unique events...")
       
        # Process each event
        for event_id in unique_events:
            # Get all points for this event
            event_data = df[df['a'] == event_id]
           
            # Get core point coordinates
            points = event_data[['x', 'y', 'z']].values
           
            # Get label for this event
            label = event_data['label'].iloc[0]
           
            # Normalize and sample points
            points_normalized = self._normalize_point_cloud(points)
            points_sampled = self._sample_points(points_normalized)
           
            # Add additional geometric features
            points_with_features = self._compute_point_features(points_sampled)
           
            # Store the processed event
            event_point_clouds.append(points_with_features)
            event_labels.append(label)
            event_ids.append(event_id)
           
            # Data Augmentation: Add augmented versions of good events
            if label == 'good':
                for i in range(augmentation_factor):
                    # Create diverse augmentations
                    augmented_points = self._augment_point_cloud(
                        points_sampled,
                        augmentation_type=random.choice(['rotation', 'jitter', 'scale', 'flip', 'combined'])
                    )
                   
                    # Compute features for augmented points
                    augmented_with_features = self._compute_point_features(augmented_points)
                   
                    event_point_clouds.append(augmented_with_features)
                    event_labels.append(label)
                    event_ids.append(event_id)  # Same ID as original
       
        # Convert string labels to numeric using label encoder
        event_labels = self.label_encoder.fit_transform(event_labels)
       
        # Print class distribution information
        self._print_class_info(event_labels, event_ids)

        # Store event IDs for later reference
        self.event_ids = np.array(event_ids)
       
        # Convert to numpy arrays
        event_point_clouds = np.array(event_point_clouds)
        event_labels = np.array(event_labels)
       
        # Shuffle the data
        indices = np.arange(len(event_point_clouds))
        np.random.shuffle(indices)
        event_point_clouds = event_point_clouds[indices]
        event_labels = event_labels[indices]
        self.event_ids = self.event_ids[indices]
       
        return event_point_clouds, event_labels

    def _print_class_info(self, labels, event_ids):
        """Print information about class distribution and events."""
        # After encoding labels, print the mapping
        print("\nLabel Encoding Mapping:")
        classes = self.label_encoder.classes_
        for i, class_name in enumerate(classes):
            print(f"Class {i}: {class_name}")
       
        # Print class distribution
        unique, counts = np.unique(labels, return_counts=True)
        print("\nClass Distribution:")
        for class_idx, count in zip(unique, counts):
            class_name = self.label_encoder.inverse_transform([class_idx])[0]
            print(f"{class_name}: {count} events")
           
        # Print info about good events specifically
        good_indices = np.where(labels == 1)[0]  # Assuming 'good' is encoded as 1
        good_event_ids = set([event_ids[i] for i in good_indices])
        orig_good_ids = [ev_id for ev_id in good_event_ids if ev_id in self.good_events]
       
        print(f"\nUnique good events found: {len(set(orig_good_ids))}")
        print(f"Original good event IDs in dataset: {sorted(orig_good_ids)}")

    def _normalize_point_cloud(self, points):
        """
        Normalize a point cloud by centering it and scaling to unit sphere.
       
        Parameters:
            points (np.array): Point cloud of shape (n_points, features)
       
        Returns:
            np.array: Normalized point cloud
        """
        centroid = np.mean(points, axis=0)
        points_centered = points - centroid
       
        max_dist = np.max(np.sqrt(np.sum(points_centered**2, axis=1)))
        if max_dist > 0:
            points_normalized = points_centered / max_dist
        else:
            points_normalized = points_centered
           
        return points_normalized

    def _compute_point_features(self, points):
        """
        Compute additional geometric features from point cloud.
       
        Parameters:
            points (np.array): Point cloud
           
        Returns:
            np.array: Points with additional features
        """
        # Calculate distances from centroid
        centroid = np.mean(points, axis=0)
        distances = np.sqrt(np.sum((points - centroid)**2, axis=1))
       
        # Calculate normalized distances
        mean_dist = np.mean(distances)
        std_dist = np.std(distances)
        if std_dist > 0:
            norm_distances = (distances - mean_dist) / std_dist
        else:
            norm_distances = np.zeros_like(distances)
       
        # Stack features as additional columns
        point_features = np.column_stack([
            points,
            distances.reshape(-1, 1),
            norm_distances.reshape(-1, 1)
        ])
       
        return point_features

    def _sample_points(self, points):
        """
        Sample a fixed number of points from the point cloud.
       
        Parameters:
            points (np.array): Point cloud
       
        Returns:
            np.array: Sampled point cloud with exactly num_points points
        """
        if len(points) == self.num_points:
            return points
        elif len(points) > self.num_points:
            # Randomly sample without replacement
            indices = np.random.choice(len(points), self.num_points, replace=False)
            return points[indices]
        else:
            # For fewer points, use a hybrid approach
            # First, include all original points
            sampled_points = points.copy()
           
            # Then, fill the remaining slots with points sampled with noise
            remaining = self.num_points - len(points)
            indices = np.random.choice(len(points), remaining, replace=True)
            selected_points = points[indices].copy()
           
            # Add small noise to these points to create more diversity
            noise = np.random.normal(0, 0.02, selected_points.shape)
            selected_points += noise
           
            # Combine original and noise-added points
            return np.vstack([sampled_points, selected_points])

    def _augment_point_cloud(self, points, augmentation_type='combined'):
        """
        Apply data augmentation to a point cloud with multiple techniques.
       
        Parameters:
            points (np.array): Point cloud
            augmentation_type (str): Type of augmentation to apply
           
        Returns:
            np.array: Augmented point cloud
        """
        # Make a copy to avoid modifying the original
        augmented_points = points.copy()
       
        if augmentation_type in ['rotation', 'combined']:
            # Random rotation around z-axis
            theta = np.random.uniform(0, 2 * np.pi)
            rotation_matrix_z = np.array([
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1]
            ])
           
            # Apply rotation
            augmented_points = np.dot(augmented_points, rotation_matrix_z)
           
            # With some probability, also rotate around other axes
            if np.random.random() < 0.3:
                # Random rotation around x axis
                phi_x = np.random.uniform(-np.pi/4, np.pi/4)
               
                rotation_matrix_x = np.array([
                    [1, 0, 0],
                    [0, np.cos(phi_x), -np.sin(phi_x)],
                    [0, np.sin(phi_x), np.cos(phi_x)]
                ])
               
                # Apply rotation
                augmented_points = np.dot(augmented_points, rotation_matrix_x)
       
        if augmentation_type in ['jitter', 'combined']:
            # Add random jitter (noise)
            jitter_scale = np.random.uniform(0.005, 0.02)
            jitter = np.random.normal(0, jitter_scale, augmented_points.shape)
            augmented_points = augmented_points + jitter

        if augmentation_type in ['scale', 'combined']:
            # Scale the point cloud
            scale_factor = np.random.uniform(0.8, 1.2)
            augmented_points = augmented_points * scale_factor

        if augmentation_type in ['flip', 'combined']:
            # Randomly flip axes
            if np.random.random() < 0.2:
                augmented_points[:, 0] = -augmented_points[:, 0]  # Flip x
            if np.random.random() < 0.2:
                augmented_points[:, 1] = -augmented_points[:, 1]  # Flip y
               
        return augmented_points

    def _get_event_category(self, event_id):
        """
        Get the category for an event ID.
       
        Parameters:
            event_id (int): The event identifier
       
        Returns:
            str: Category name ('seagulls', 'triplets', etc., or 'junk' if not found)
        """
        for category, events in self.categories.items():
            if event_id in events:
                return category
        return 'junk'

    def create_model(self, input_shape):
        """
        Create the PointNet model architecture with regularization.
       
        Parameters:
            input_shape: Shape of input data (n_points, n_features)
           
        Returns:
            keras.Model: Compiled model
        """
        # Define regularization
        l2_reg = keras.regularizers.l2(0.001)
       
        # Input layer
        inputs = keras.layers.Input(shape=input_shape)
       
        # Feature extraction network
        x = keras.layers.Conv1D(64, 1, activation=None, kernel_regularizer=l2_reg)(inputs)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.LeakyReLU(alpha=0.2)(x)
       
        x = keras.layers.Conv1D(128, 1, activation=None, kernel_regularizer=l2_reg)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.LeakyReLU(alpha=0.2)(x)
        x = keras.layers.Dropout(0.3)(x)
       
        # Global feature aggregation
        x_max = keras.layers.GlobalMaxPooling1D()(x)
        x_avg = keras.layers.GlobalAveragePooling1D()(x)
        x = keras.layers.Concatenate()([x_max, x_avg])
       
        # Dense layers with stronger regularization
        x = keras.layers.Dense(256, activation=None, kernel_regularizer=l2_reg)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.LeakyReLU(alpha=0.2)(x)
        x = keras.layers.Dropout(0.5)(x)
       
        x = keras.layers.Dense(128, activation=None, kernel_regularizer=l2_reg)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.LeakyReLU(alpha=0.2)(x)
        x = keras.layers.Dropout(0.5)(x)
       
        # Final classification layer
        if self.num_classes == 2:
            # For binary classification, use sigmoid
            outputs = keras.layers.Dense(1, activation='sigmoid')(x)
        else:
            # For multi-class, use softmax
            outputs = keras.layers.Dense(self.num_classes, activation='softmax')(x)
       
        # Create model
        self.model = keras.Model(inputs=inputs, outputs=outputs)
        return self.model

    def train(self, X, y, epochs=100, batch_size=32, validation_split=0.2):
        """
        Train the model with improved strategies.
       
        Parameters:
            X: Point clouds data
            y: Labels
            epochs: Maximum number of training epochs
            batch_size: Batch size for training
            validation_split: Portion of data to use for validation
           
        Returns:
            history: Training history
        """
        input_shape = (X.shape[1], X.shape[2])
        print(f"Input shape: {input_shape}")
       
        # Split data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42, stratify=y
        )
       
        # Compute balanced class weights
        if self.num_classes == 2:
            # Compute weight based on class frequency
            class_counts = np.bincount(y_train)
            total = len(y_train)
            weight_for_0 = (1 / class_counts[0]) * (total / 2)
            weight_for_1 = (1 / class_counts[1]) * (total / 2)
           
            # Use a more balanced approach with focus on good events
            class_weights = {0: weight_for_0 * 0.75, 1: weight_for_1 * 1.25}
        else:
            class_weights = None
       
        print(f"\nTraining set: {len(X_train)} samples")
        print(f"  - Good events (class 1): {np.sum(y_train == 1)} ({np.sum(y_train == 1)/len(y_train)*100:.1f}%)")
        print(f"  - Bad events (class 0): {np.sum(y_train == 0)} ({np.sum(y_train == 0)/len(y_train)*100:.1f}%)")
        print(f"Validation set: {len(X_val)} samples")
        print(f"  - Good events (class 1): {np.sum(y_val == 1)} ({np.sum(y_val == 1)/len(y_val)*100:.1f}%)")
        print(f"  - Bad events (class 0): {np.sum(y_val == 0)} ({np.sum(y_val == 0)/len(y_val)*100:.1f}%)")
        print(f"\nUsing class weights: {class_weights}")
       
        # Create model
        self.create_model(input_shape)
       
        # Compile with appropriate metrics
        if self.num_classes == 2:
            self.model.compile(
                optimizer='adam',
                loss='binary_crossentropy',
                metrics=[
                    'accuracy',
                    tf.keras.metrics.Precision(name='precision'),
                    tf.keras.metrics.Recall(name='recall'),
                    tf.keras.metrics.AUC(name='auc')
                ]
            )
        else:
            self.model.compile(
                optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
       
        # Define callbacks
        callbacks = [
            # Early stopping based on validation recall with patience
            keras.callbacks.EarlyStopping(
                monitor='val_recall' if self.num_classes == 2 else 'val_loss',
                patience=15,
                restore_best_weights=True,
                mode='max' if self.num_classes == 2 else 'min'
            ),
            # Model checkpoint to save best model
            keras.callbacks.ModelCheckpoint(
                'best_model.keras',
                monitor='val_recall' if self.num_classes == 2 else 'val_loss',
                save_best_only=True,
                mode='max' if self.num_classes == 2 else 'min'
            ),
            # ReduceLROnPlateau to reduce learning rate when performance plateaus
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            )
        ]
       
        # Train the model
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            class_weight=class_weights
        )
       
        return history, (X_val, y_val)

    def find_optimal_threshold(self, X_val, y_val):
        """
        Find the optimal threshold for binary classification.
       
        Parameters:
            X_val: Validation data
            y_val: Validation labels
           
        Returns:
            float: Optimal threshold
        """
        if self.num_classes != 2:
            raise ValueError("Finding optimal threshold only applies to binary classification")
           
        # Get predictions
        y_pred_prob = self.model.predict(X_val).flatten()
       
        # Compute precision-recall curve and find best threshold
        precision, recall, thresholds = precision_recall_curve(y_val, y_pred_prob)
       
        # Calculate F1 score for each threshold
        f1_scores = []
        for t in thresholds:
            y_pred = (y_pred_prob >= t).astype(int)
            f1 = f1_score(y_val, y_pred)
            f1_scores.append(f1)
           
        # Find threshold with best F1 score
        best_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_idx]
        best_f1 = f1_scores[best_idx]
       
        print(f"Found optimal threshold: {best_threshold:.4f} with F1 score: {best_f1:.4f}")
       
        return best_threshold

    def evaluate(self, X_test, y_test, threshold=0.5):
        """
        Evaluate the model's performance with various metrics.
       
        Parameters:
            X_test: Test data
            y_test: Test labels
            threshold: Classification threshold for binary models
           
        Returns:
            dict: Evaluation results
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
           
        # For binary classification
        if self.num_classes == 2:
            # Get predictions
            y_pred_prob = self.model.predict(X_test).flatten()
            y_pred = (y_pred_prob >= threshold).astype(int)
           
            # Print predictions for good events
            good_indices = np.where(y_test == 1)[0]
            if len(good_indices) > 0:
                print(f"\nPredictions for good events (threshold = {threshold}):")
                for i, idx in enumerate(good_indices):
                    status = "CORRECT" if y_pred[idx] == 1 else "WRONG"
                    print(f"  Good event {i+1}: {y_pred_prob[idx]:.4f} -> {status}")
           
            # Print overall evaluation metrics
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred, zero_division=0))
           
            print("\nConfusion Matrix:")
            cm = confusion_matrix(y_test, y_pred)
            print(cm)
           
            # Calculate metrics for good event detection
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
               
                print("\nGood Event Detection Metrics:")
                print(f"  Precision: {precision:.4f}")
                print(f"  Recall: {recall:.4f}")
                print(f"  F1 Score: {f1:.4f}")
               
                # Try different thresholds
                print("\nPerformance at different thresholds:")
                thresholds = np.arange(0.3, 0.7, 0.05)
                best_f1 = 0
                best_threshold = 0.5
               
                for t in thresholds:
                    y_pred_t = (y_pred_prob >= t).astype(int)
                    cm_t = confusion_matrix(y_test, y_pred_t)
                   
                    if cm_t.shape == (2, 2):
                        tn, fp, fn, tp = cm_t.ravel()
                        precision_t = tp / (tp + fp) if (tp + fp) > 0 else 0
                        recall_t = tp / (tp + fn) if (tp + fn) > 0 else 0
                        f1_t = 2 * (precision_t * recall_t) / (precision_t + recall_t) if (precision_t + recall_t) > 0 else 0
                       
                        print(f"  Threshold {t:.2f}: Precision={precision_t:.4f}, Recall={recall_t:.4f}, F1={f1_t:.4f}")
                       
                        if f1_t > best_f1:
                            best_f1 = f1_t
                            best_threshold = t
               
                print(f"\nBest threshold found: {best_threshold:.2f} with F1 score: {best_f1:.4f}")
               
                # Return evaluation results
                results = {
                    "accuracy": np.mean(y_pred == y_test),
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "confusion_matrix": cm,
                    "best_threshold": best_threshold,
                    "best_f1": best_f1,
                    "predictions": y_pred_prob
                }
               
                return results
           
        # For multi-class classification
        else:
            # Get predictions
            y_pred = np.argmax(self.model.predict(X_test), axis=1)
           
            # Print evaluation metrics
            print("\nMulti-class Classification Report:")
            print(classification_report(y_test, y_pred, zero_division=0))
           
            # Return evaluation results
            results = {
                "accuracy": np.mean(y_pred == y_test),
                "confusion_matrix": confusion_matrix(y_test, y_pred),
                "predictions": y_pred
            }
           
            return results

    def plot_training_history(self, history):
        """
        Plot the training history.
       
        Parameters:
            history: Training history from model.fit()
        """
        # Get available metrics
        metrics = list(history.history.keys())
        train_metrics = [m for m in metrics if not m.startswith('val_')]
       
        # Determine how many plots to create
        n_plots = len(train_metrics)
        n_cols = min(3, n_plots)
        n_rows = (n_plots + n_cols - 1) // n_cols
       
        # Create the figure
        plt.figure(figsize=(n_cols*6, n_rows*4))
       
        # Plot each metric
        for i, metric in enumerate(train_metrics, 1):
            plt.subplot(n_rows, n_cols, i)
           
            plt.plot(history.history[metric], label=f'Training {metric}')
            if f'val_{metric}' in metrics:
                plt.plot(history.history[f'val_{metric}'], label=f'Validation {metric}')
               
            plt.title(f'Model {metric.capitalize()}')
            plt.xlabel('Epoch')
            plt.ylabel(metric.capitalize())
            plt.legend()
            plt.grid(True, alpha=0.3)
       
        plt.tight_layout()
        plt.show()

    def visualize_predictions(self, X, y_true, y_pred=None, threshold=0.5, sample_size=5):
        """
        Visualize a sample of predictions in 3D.
       
        Parameters:
            X: Input data
            y_true: True labels
            y_pred: Predicted probabilities (for binary) or classes
            threshold: Classification threshold (for binary)
            sample_size: Number of samples to visualize
        """
        # If no predictions provided, generate them
        if y_pred is None and self.model is not None:
            if self.num_classes == 2:
                y_pred = self.model.predict(X).flatten()
            else:
                y_pred = np.argmax(self.model.predict(X), axis=1)
               
        # Create a figure with subplots
        fig = plt.figure(figsize=(15, sample_size * 3))
       
        # For binary classification
        if self.num_classes == 2:
            # Get indices for different cases
            correct_pos = np.where((y_true == 1) & (y_pred >= threshold))[0]  # True positives
            incorrect_pos = np.where((y_true == 0) & (y_pred >= threshold))[0]  # False positives
            incorrect_neg = np.where((y_true == 1) & (y_pred < threshold))[0]  # False negatives
           
            # Prioritize visualization of interesting cases
            interesting_indices = []
           
            # Add false negatives (missed good events) - these are most important
            n_fn = min(sample_size // 2, len(incorrect_neg))
            if n_fn > 0:
                interesting_indices.extend(np.random.choice(incorrect_neg, n_fn, replace=False))
               
            # Add some false positives
            n_fp = min(sample_size // 4, len(incorrect_pos))
            if n_fp > 0:
                interesting_indices.extend(np.random.choice(incorrect_pos, n_fp, replace=False))
               
            # Fill the rest with correct predictions
            n_tp = sample_size - len(interesting_indices)
            if n_tp > 0 and len(correct_pos) > 0:
                interesting_indices.extend(np.random.choice(correct_pos, min(n_tp, len(correct_pos)), replace=False))
               
            # Visualize each selected sample
            for i, idx in enumerate(interesting_indices[:sample_size]):
                # Get point data (first 3 coords are x, y, z)
                points = X[idx, :, :3]
               
                # Determine point types
                true_label = "Good" if y_true[idx] == 1 else "Bad"
                pred_value = y_pred[idx]
                pred_label = "Good" if pred_value >= threshold else "Bad"
                correct = true_label == pred_label
               
                # Set colors based on prediction correctness
                color = 'green' if correct else 'red'
               
                # Create 3D subplot
                ax = fig.add_subplot(sample_size, 1, i+1, projection='3d')
               
                # Plot points
                ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=color, s=10, alpha=0.7)
               
                # Add event ID if available
                event_id = self.event_ids[idx] if hasattr(self, 'event_ids') else f"Sample {idx}"
               
                # Set title with prediction info
                ax.set_title(f"Event {event_id}: True={true_label}, Pred={pred_label} ({pred_value:.4f})")
               
                # Set labels
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')
       
        plt.tight_layout()
        plt.show()

    def save_model(self, filepath="enhanced_event_classifier.keras"):
        """Save the model and necessary configuration."""
        if self.model is None:
            raise ValueError("No model to save - train a model first")
           
        # Save the model
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
       
        # Save label encoder and other configuration
        config_path = os.path.splitext(filepath)[0] + "_config.npz"
       
        # Prepare configuration data to save
        config_data = {
            'label_classes': self.label_encoder.classes_,
            'num_classes': self.num_classes,
            'num_points': self.num_points
        }
       
        # Save configuration
        np.savez(config_path, **config_data)
        print(f"Configuration saved to {config_path}")

#%%
# Small section for debugging/testing/additional arguments before running program
# Initialize the classifier
"""
print("Initializing Enhanced Event Classifier...")
classifier = EnhancedEventClassifier(num_points=1024, num_classes=2)
# Load and preprocess data
print("Loading and preprocessing data...")
X, y = classifier.load_and_preprocess_data(
    "/home/jak689/3D-ParticlePlot-Main/CompleteData.txt",
    binary=True,
    augmentation_factor=10
)

print(f"Dataset shape: {X.shape}")
print(f"Class distribution: {np.bincount(y)}")
"""
#%%
"""
print("X:", X)
print("lenght of X: ", len(X))
print("y:", y)
testgood = np.arange(0,len(X),10)
print("Test Array: ", testgood)
print(len(testgood))
"""
#%%
# Seperated main function to work on variables without running program first
def main():
    """
    Main function to train and evaluate the enhanced event classifier.
    """
    # Initialize the classifier
    print("Initializing Enhanced Event Classifier...")
    classifier = EnhancedEventClassifier(num_points=1024, num_classes=2)

    # Load and preprocess data
    print("Loading and preprocessing data...")
    X, y = classifier.load_and_preprocess_data(
        "/home/jak689/3D-ParticlePlot-Main/CompleteData.txt",
        binary=True,
        augmentation_factor=0
    )
   
    print(f"Dataset shape: {X.shape}")
    print(f"Class distribution: {np.bincount(y)}")
   
    # Train model
    print("\nTraining model...")
    history, validation_data = classifier.train(
        X, y,
        epochs=100,
        batch_size=32,
        validation_split=0.3
    )
   
    # Find optimal threshold
    X_val, y_val = validation_data
    optimal_threshold = classifier.find_optimal_threshold(X_val, y_val)
   
    # Evaluate model with the optimal threshold
    print("\nEvaluating model with optimal threshold...")
    evaluation_results = classifier.evaluate(X_val, y_val, threshold=optimal_threshold)
   
    # Plot training history
    classifier.plot_training_history(history)
   
    # Visualize predictions
    print("\nVisualizing sample predictions...")
    y_pred = classifier.model.predict(X_val).flatten()
    classifier.visualize_predictions(X_val, y_val, y_pred, threshold=optimal_threshold)
   
    # Save the trained model
    classifier.save_model("enhanced_event_classifier.keras")
   
    print("\nTraining and evaluation complete!")
   

if __name__ == "__main__":
    main()