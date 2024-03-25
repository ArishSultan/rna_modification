from pandas import DataFrame, Series


def encode_df(df: Series, encoder, col_name: str, columns: list | None = None):
    sequences = df['sequence'] if 'sequence' in df else df[0]

    encoded_df = sequences.apply(lambda seq: encoder(seq))
    if columns is not None:
        encoded_df = encoded_df[columns]
    encoded_df.columns = [f'{col_name}_{i}' for i, col in enumerate(encoded_df.columns)]

    return encoded_df


def encode_df_indexed(df: DataFrame, encoder, col_name: str, columns: list | None = None):
    sequences = df['sequence'] if 'sequence' in df else df[0]

    encoded_seqs = {}
    for i in range(len(sequences)):
        encoded_seqs[sequences[i]] = encoder(i, sequences[i])

    encoded_df = sequences.apply(lambda seq: Series(encoded_seqs[seq]))
    if columns is not None:
        encoded_df = encoded_df[columns]
    encoded_df.columns = [f'{col_name}_{i}' for i, col in enumerate(encoded_df.columns)]

    return encoded_df
