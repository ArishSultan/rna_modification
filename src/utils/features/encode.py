from pandas import DataFrame, Series


def encode_df(df: DataFrame, encoder, col_name: str, columns: list | None = None):
    sequences = df['sequence'] if 'sequence' in df else df[0]

    encoded_df = sequences.apply(lambda seq: Series(encoder(seq)))
    if columns is not None:
        encoded_df = encoded_df[columns]
    encoded_df.columns = [f'{col_name}_{i}' for i, col in enumerate(encoded_df.columns)]

    return encoded_df
