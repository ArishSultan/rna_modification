import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from scipy.stats import wasserstein_distance
from umap import UMAP
from src.dataset import load_benchmark_dataset, Species, Modification
from src.features.encodings import multiple, binary, ncp, pstnpss, pse_knc, dmf

# Encoder definition
encoder = multiple.Encoder([
    # binary.Encoder(),
    # ncp.Encoder(),
    dmf.Encoder(3, 5),
    # pstnpss.Encoder(),
    # pse_knc.Encoder()
])

# UMAP transformer
umap_transformer = UMAP(n_components=2, random_state=42)

# Load datasets
test_dataset = load_benchmark_dataset(Species.human, Modification.psi, True)
train_dataset = load_benchmark_dataset(Species.human, Modification.psi)

# Transform datasets
train_samples = encoder.fit_transform(train_dataset.samples, y=train_dataset.targets)
train_samples = umap_transformer.fit_transform(train_samples, y=train_dataset.targets)

test_samples = encoder.transform(test_dataset.samples)
test_samples = umap_transformer.transform(test_samples)

# Separate positive and negative samples in train and test datasets
train_pos_samples = train_samples[train_dataset.targets == 1]
train_neg_samples = train_samples[train_dataset.targets == 0]

test_pos_samples = test_samples[test_dataset.targets == 1]
test_neg_samples = test_samples[test_dataset.targets == 0]

# Fit GMMs for positive samples on the training dataset
gmm_train_pos = GaussianMixture(n_components=2, random_state=0)
gmm_train_pos.fit(train_pos_samples)

# Fit GMMs for negative samples on the training dataset
gmm_train_neg = GaussianMixture(n_components=2, random_state=0)
gmm_train_neg.fit(train_neg_samples)



# Generate new samples
n_augment = 500
aug_train_pos_samples = gmm_train_pos.sample(n_augment)[0]
aug_test_pos_samples = gmm_train_pos.sample(n_augment)[0]

aug_train_neg_samples = gmm_train_neg.sample(n_augment)[0]
aug_test_neg_samples = gmm_train_neg.sample(n_augment)[0]

# Perform similarity analysis using Wasserstein distance
distance_pos_neg_original = wasserstein_distance(train_pos_samples.ravel(), train_neg_samples.ravel())
distance_pos_neg_test = wasserstein_distance(test_pos_samples.ravel(), test_neg_samples.ravel())
distance_pos_original = wasserstein_distance(train_pos_samples.ravel(), test_pos_samples.ravel())
distance_pos_train = wasserstein_distance(train_pos_samples.ravel(), aug_train_pos_samples.ravel())
distance_pos_test = wasserstein_distance(test_pos_samples.ravel(), aug_test_pos_samples.ravel())
distance_pos_between = wasserstein_distance(aug_train_pos_samples.ravel(), aug_test_pos_samples.ravel())

distance_neg_original = wasserstein_distance(train_neg_samples.ravel(), test_neg_samples.ravel())
distance_neg_train = wasserstein_distance(train_neg_samples.ravel(), aug_train_neg_samples.ravel())
distance_neg_test = wasserstein_distance(test_neg_samples.ravel(), aug_test_neg_samples.ravel())
distance_neg_between = wasserstein_distance(aug_train_neg_samples.ravel(), aug_test_neg_samples.ravel())

print(f"Wasserstein Distance (Train Positives vs. Train Negatives): {distance_pos_neg_original}")
print(f"Wasserstein Distance (Test Positives vs. Test Negatives): {distance_pos_neg_test}")
print(f"Wasserstein Distance (Train Positives vs. Test Positives): {distance_pos_original}")
print(f"Wasserstein Distance (Train Positives vs. Augmented Train Positives): {distance_pos_train}")
print(f"Wasserstein Distance (Test Positives vs. Augmented Test Positives): {distance_pos_test}")
print(f"Wasserstein Distance (Augmented Train Positives vs. Augmented Test Positives): {distance_pos_between}")

print(f"Wasserstein Distance (Train Negatives vs. Test Negatives): {distance_neg_original}")
print(f"Wasserstein Distance (Train Negatives vs. Augmented Train Negatives): {distance_neg_train}")
print(f"Wasserstein Distance (Test Negatives vs. Augmented Test Negatives): {distance_neg_test}")
print(f"Wasserstein Distance (Augmented Train Negatives vs. Augmented Test Negatives): {distance_neg_between}")

# Visualize the results
plt.figure(figsize=(12, 18))

# Original positive samples
plt.subplot(3, 2, 1)
plt.scatter(train_pos_samples[:, 0], train_pos_samples[:, 1], color='blue', alpha=0.5, label='Train +ive')
plt.scatter(test_pos_samples[:, 0], test_pos_samples[:, 1], color='red', alpha=0.5, label='Test +ive')
plt.title('Original Positive Samples')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()

# Augmented positive samples
plt.subplot(3, 2, 2)
plt.scatter(aug_train_pos_samples[:, 0], aug_train_pos_samples[:, 1], color='cyan', alpha=0.5, marker='x', label='Augmented Train +ive')
plt.scatter(aug_test_pos_samples[:, 0], aug_test_pos_samples[:, 1], color='magenta', alpha=0.5, marker='x', label='Augmented Test +ive')
plt.title('Augmented Positive Samples')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()

# Original negative samples
plt.subplot(3, 2, 3)
plt.scatter(train_neg_samples[:, 0], train_neg_samples[:, 1], color='green', alpha=0.5, label='Train -ive')
plt.scatter(test_neg_samples[:, 0], test_neg_samples[:, 1], color='orange', alpha=0.5, label='Test -ive')
plt.title('Original Negative Samples')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()

# Augmented negative samples
plt.subplot(3, 2, 4)
plt.scatter(aug_train_neg_samples[:, 0], aug_train_neg_samples[:, 1], color='yellow', alpha=0.5, marker='x', label='Augmented Train -ive')
plt.scatter(aug_test_neg_samples[:, 0], aug_test_neg_samples[:, 1], color='purple', alpha=0.5, marker='x', label='Augmented Test -ive')
plt.title('Augmented Negative Samples')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()

# Combined positive and negative samples
plt.subplot(3, 2, 5)
plt.scatter(train_pos_samples[:, 0], train_pos_samples[:, 1], color='blue', alpha=0.5, label='Train +ive')
plt.scatter(test_pos_samples[:, 0], test_pos_samples[:, 1], color='red', alpha=0.5, label='Test +ive')
plt.scatter(train_neg_samples[:, 0], train_neg_samples[:, 1], color='green', alpha=0.5, label='Train -ive')
plt.scatter(test_neg_samples[:, 0], test_neg_samples[:, 1], color='orange', alpha=0.5, label='Test -ive')
plt.title('Combined Original Samples')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()

plt.subplot(3, 2, 6)
plt.scatter(aug_train_pos_samples[:, 0], aug_train_pos_samples[:, 1], color='cyan', alpha=0.5, marker='x', label='Augmented Train +ive')
plt.scatter(aug_test_pos_samples[:, 0], aug_test_pos_samples[:, 1], color='magenta', alpha=0.5, marker='x', label='Augmented Test +ive')
plt.scatter(aug_train_neg_samples[:, 0], aug_train_neg_samples[:, 1], color='yellow', alpha=0.5, marker='x', label='Augmented Train -ive')
plt.scatter(aug_test_neg_samples[:, 0], aug_test_neg_samples[:, 1], color='purple', alpha=0.5, marker='x', label='Augmented Test -ive')
plt.title('Combined Augmented Samples')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()

plt.tight_layout()
plt.show()
