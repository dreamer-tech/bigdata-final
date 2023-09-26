import pandas as pd


def convert_date(d):
    if d.count('-') == 0:
        return d + '-01-01'
    if d.count('-') == 1:
        return d + '-01'
    return d


def pure_text(d):
    return ''.join([c for c in d if c.isalnum() or c == ' '])


if __name__ == '__main__':
    artists_df = pd.read_csv('./data/artists.csv')
    tracks_df = pd.read_csv('./data/tracks.csv')

    artists_df['followers'].fillna(0.0, inplace=True)
    artists_df['name'] = artists_df['name'].apply(pure_text)

    tracks_df['release_date'] = tracks_df['release_date'].apply(convert_date)
    tracks_df['artists'] = tracks_df['artists'].apply(lambda artists: str([pure_text(artist)
                                                                           for artist in eval(artists)]))

    artists_df.to_csv("./data/artists.csv", index=False)
    tracks_df.to_csv("./data/tracks.csv", index=False)
