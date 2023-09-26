import pandas as pd


if __name__ == '__main__':
    artists_df = pd.read_csv('./data/artists.csv')

    artists_df['followers'].fillna(0.0, inplace=True)

    artists_df.to_csv("./data/artists.csv", index=False)
