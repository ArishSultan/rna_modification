# import pandas as pd
# import random
#
# from sklearn.model_selection import train_test_split
#
#
# def extract_sequences(samples: list[str], seq_len: int, nucleotide: str) -> list[str]:
#     # Calculate the center position based on the desired sequence length
#     center_pos = seq_len // 2
#     extracted_sequences = []
#
#     for seq in samples:
#         # Check if the sequence is long enough to extract a subsequence of length `seq_len`
#         if len(seq) < seq_len:
#             continue
#
#         # Search for `nucleotide` within each sequence
#         for i in range(center_pos, len(seq) - center_pos):
#             # Check if the nucleotide at the center position matches the target nucleotide
#             if seq[i] == nucleotide:
#                 # Extract the substring centered at `nucleotide` with length `seq_len`
#                 start = i - center_pos
#                 end = i + center_pos + 1
#                 extracted_seq = seq[start:end]
#
#                 # Ensure the extracted sequence is of correct length
#                 if len(extracted_seq) == seq_len:
#                     extracted_sequences.append(extracted_seq)
#                 break  # Stop after finding the first valid center nucleotide in the sequence
#
#     return list(set(extracted_sequences))
#
#
# def subsample_center_sequences(samples: list[str], seq_len: int) -> list[str]:
#     # Calculate the half-length to determine the start position for cropping
#     center_pos = seq_len // 2
#     subsampled_sequences = []
#
#     for seq in samples:
#         # Only process sequences that are long enough
#         if len(seq) >= seq_len:
#             # Calculate the start and end positions for cropping
#             start = (len(seq) // 2) - center_pos
#             end = start + seq_len
#             # Crop the sequence from the center
#             cropped_seq = seq[start:end]
#             subsampled_sequences.append(cropped_seq)
#
#     return subsampled_sequences
#
#
# if __name__ == '__main__':
#     psi_dataset = \
#         pd.read_csv('/Users/arish/Research/research/rna_modification/dataset/intermediate/psi/h.sapiens.csv',
#                     header=None)[
#             0].values
#     m6a_dataset = \
#         pd.read_csv('/Users/arish/Research/research/rna_modification/dataset/intermediate/m6a/h.sapiens.csv',
#                     header=None)[
#             0].values
#
#     extracted_sequences = extract_sequences(m6a_dataset, 39, 'U')
#     sub_sampled_sequences = extract_sequences(psi_dataset, 39, 'U')
#
#     sequences_to_remove = []
#     for seq in extracted_sequences:
#         if seq in sub_sampled_sequences:
#             sequences_to_remove.append(seq)
#
#     for seq in sequences_to_remove:
#         extracted_sequences.remove(seq)
#
#     # Label data: m6A (negative) as 0, psi (positive) as 1
#     negative_samples = pd.DataFrame({'sequence': extracted_sequences, 'label': 0})
#     positive_samples = pd.DataFrame({'sequence': sub_sampled_sequences, 'label': 1})
#
#     # Balance the dataset by undersampling the negative samples
#     num_positives = len(positive_samples)
#     balanced_negatives = negative_samples.sample(n=num_positives, random_state=42)
#
#     # Combine and shuffle the dataset
#     balanced_dataset = pd.concat([positive_samples, balanced_negatives], ignore_index=True)
#     balanced_dataset = balanced_dataset.sample(frac=1, random_state=42).reset_index(drop=True)
#
#     # Split the dataset into training and testing sets (80% train, 20% test)
#     train_set, test_set = train_test_split(balanced_dataset, test_size=0.2, random_state=42,
#                                            stratify=balanced_dataset['label'])
#
#     # Save the datasets to CSV files
#     train_set.to_csv('/Users/arish/Research/research/rna_modification/dataset/train_set.csv', index=False,
#                      header=['sequence', 'label'])
#     test_set.to_csv('/Users/arish/Research/research/rna_modification/dataset/test_set.csv', index=False,
#                     header=['sequence', 'label'])