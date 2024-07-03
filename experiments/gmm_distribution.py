import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report
import tensorflow as tf


# Define UNFIS Layers and Model
class FuzzificationLayer(tf.keras.layers.Layer):
    def __init__(self, num_rules, num_inputs):
        super(FuzzificationLayer, self).__init__()
        self.num_rules = num_rules
        self.num_inputs = num_inputs
        self.centers = self.add_weight(shape=(num_rules, num_inputs), initializer='random_normal', trainable=True)
        self.sigmas = self.add_weight(shape=(num_rules, num_inputs), initializer='random_normal', trainable=True)

    def call(self, inputs):
        return tf.exp(-tf.square(inputs[:, tf.newaxis, :] - self.centers) / (2 * tf.square(self.sigmas)))


class FuzzySelectionLayer(tf.keras.layers.Layer):
    def __init__(self, num_rules, num_inputs):
        super(FuzzySelectionLayer, self).__init__()
        self.num_rules = num_rules
        self.num_inputs = num_inputs
        self.selection_params = self.add_weight(shape=(num_rules, num_inputs), initializer='random_normal',
                                                trainable=True)

    def call(self, fuzzified_inputs):
        selection = tf.sigmoid(self.selection_params)
        return fuzzified_inputs * selection


class FuzzyRulesLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(FuzzyRulesLayer, self).__init__()

    def call(self, selected_inputs):
        return tf.reduce_prod(selected_inputs, axis=-1)


class NormalizationLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(NormalizationLayer, self).__init__()

    def call(self, firing_strengths):
        return firing_strengths / tf.reduce_sum(firing_strengths, axis=-1, keepdims=True)


class ConsequentLayer(tf.keras.layers.Layer):
    def __init__(self, num_rules, num_outputs):
        super(ConsequentLayer, self).__init__()
        self.num_rules = num_rules
        self.num_outputs = num_outputs
        self.consequent_params = self.add_weight(shape=(num_rules, num_outputs), initializer='random_normal',
                                                 trainable=True)

    def call(self, normalized_strengths):
        return tf.reduce_sum(normalized_strengths[:, :, tf.newaxis] * self.consequent_params, axis=1)


def create_unfis_model(num_inputs, num_rules, num_outputs, is_classification=True):
    inputs = tf.keras.Input(shape=(num_inputs,))
    fuzzification_layer = FuzzificationLayer(num_rules, num_inputs)
    fuzzy_selection_layer = FuzzySelectionLayer(num_rules, num_inputs)
    fuzzy_rules_layer = FuzzyRulesLayer()
    normalization_layer = NormalizationLayer()
    consequent_layer = ConsequentLayer(num_rules, num_outputs)

    fuzzified_inputs = fuzzification_layer(inputs)
    selected_inputs = fuzzy_selection_layer(fuzzified_inputs)
    firing_strengths = fuzzy_rules_layer(selected_inputs)
    normalized_strengths = normalization_layer(firing_strengths)
    outputs = consequent_layer(normalized_strengths)

    if is_classification:
        outputs = tf.keras.layers.Dense(1, activation='sigmoid')(outputs)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


# Create dummy binary classification dataset
X, y = make_classification(n_samples=1000, n_features=4, n_informative=2, n_redundant=0, n_clusters_per_class=1,
                           random_state=42)
y = y.reshape(-1, 1)

# One-hot encode the labels
encoder = OneHotEncoder()
y_onehot = encoder.fit_transform(y)

print(y_onehot)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and compile the UNFIS model
num_inputs = X_train.shape[1]
num_rules = 5
num_outputs = y_train.shape[1]

model = create_unfis_model(num_inputs, num_rules, num_outputs, is_classification=True)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test), verbose=1)

# Predict on the test set
y_pred = model.predict(X_test)
# y_pred_classes = np.argmax(y_pred, axis=1)
# y_test_classes = np.argmax(y_test, axis=1)

# Generate and print the classification report
report = classification_report(y_test, y_pred, target_names=['Class 0', 'Class 1'])
print(report)
