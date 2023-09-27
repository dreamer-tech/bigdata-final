import pandas as pd


def convert_date(d):
    if d.count('-') == 0:
        return d + '-01-01'
    if d.count('-') == 1:
        return d + '-01'
    return d


if __name__ == '__main__':
    artists_df = pd.read_csv('./data/artists.csv')
    tracks_df = pd.read_csv('./data/tracks.csv')

    artists_df['followers'].fillna(0.0, inplace=True)

    tracks_df['release_date'] = tracks_df['release_date'].apply(convert_date)

    artists_df.to_csv("./data/artists.csv", index=False)
    tracks_df.to_csv("./data/tracks.csv", index=False)
