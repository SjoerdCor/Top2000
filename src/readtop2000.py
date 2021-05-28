"""
Created on Fri Jan  8 10:50:11 2021

@author: Gebruiker
"""

import os
import datetime
import locale


import pandas as pd
import numpy as np
import tqdm

import wikipediareader

URL_TOP2000_HISTORY = 'https://nl.wikipedia.org/wiki/Lijst_van_Radio_2-Top_2000%27s'


class Top2000Cleaner:
    def __init__(self):
        locale.setlocale(locale.LC_TIME, 'nl_NL.utf8') # We download from dutch wikipedia, so we need dutch month names

    def _check_column_completeness(self, data):
        current_year = datetime.datetime.now().year
        columns_should_be_present = ['Artiest', 'Titel', 'Jaar', 'HP', 'TitelLink', 'ArtiestLinks']
        columns_should_be_present += [str(i%100).zfill(2) for i in range(1999, current_year) ]
        assert all(col in data.columns for col in columns_should_be_present)
        
    def _validate(self, data):
        assert data.notnull().all().all()
        self._check_column_completeness(data)
        return data
        
    def rename_columns(self, data):
        column_rename =  {'Titel': 'Title',
                          'Artiest': 'Artist',
                          'Jaar': 'YearMade',
                          'TitelLink': 'TitleLink',
                          'ArtiestLinks': 'ArtistLinks',
                          }
        data = data.rename(columns=column_rename)
        return data
        
    def split_into_model(self, data):
        '''
        Split data into tables according to data model
        '''

        data['SongID'] = (data[['Title', 'Artist']].applymap(lambda x: x.lower())
                               .apply(lambda x: hash(tuple(x)), axis=1)
                               )
        self.ranking = data[['SongID'] + [col for col in data.columns if col.isnumeric()]]
        self.song = data.set_index('SongID')[['Title', 'YearMade', 'TitleLink']]

        exploded = data.explode('ArtistLinks')
        exploded[['Name', 'ArtistLink']] = pd.DataFrame(exploded['ArtistLinks'].tolist(), index=exploded.index)
        
        # We hash on ArtistLink because names can be doubly used 
        # e.g. Nena as band and singer, Mr. Big is the name of two different bands
        exploded['ArtistID'] = exploded['ArtistLink'].apply(hash)

        self.songartist = exploded[['SongID', 'ArtistID']].set_index(['SongID', 'ArtistID']).copy()
        self.artist = (exploded.set_index('ArtistID')[['Name', 'ArtistLink']]
                      .assign(ArtistLink = lambda df: 'https://nl.wikipedia.org' + df['ArtistLink'])
                      .rename(columns={'ArtistLink': 'Link'})
                      )
        self.artist = self.artist[~self.artist.index.duplicated(keep='first')]
        return self.ranking, self.song, self.songartist, self.artist

    def clean_ranking(self):
        
        # Column names of the big Wikipedia table are two-digit numbers instead of four
        self.ranking = (self.ranking.melt(id_vars=['SongID'], var_name='Year', value_name='Rank')
                                    .assign(Year = lambda df: (df['Year'].astype(int)
                                                                .add(2000)
                                                                .replace(2099, 1999)
                                                                ),
                                            # Rank is NaN if a song is not ranked in Top2000 in a year
                                            Rank = lambda df: pd.to_numeric(df['Rank'],
                                                                            errors='coerce',
                                                                            downcast='integer')
                                            )
                                    .dropna()
                                    )

    def validate_ranking(self):
        assert (self.ranking.groupby('Year')['Rank'].apply(set) == set(range(1, 2001))).all()
        current_year = datetime.datetime.now().year
        assert self.ranking['Year'].isin(range(1999, current_year)).all()

    def clean_song(self):
        self.song = (self.song.rename(columns={'TitleLink': 'Link'})
                          .assign(Link = lambda df: 'https://nl.wikipedia.org' + df['Link'])
                    )
    
    def clean_artist(df):
        df = df.assign(DateOfDeath = lambda df: wikipediareader.wikipedia_datetime_to_datetime(df['Overleden']),
                       DateOfBirth = lambda df: wikipediareader.wikipedia_datetime_to_datetime(df['Geboren']),
                                        )
        return df
        
    
    def clean(self, data):
        (data.pipe(self._validate)
             .pipe(self.rename_columns)
             .pipe(self.split_into_model)
        )
        self.clean_ranking()
        self.validate_ranking()
        self.clean_song()
        return self.ranking, self.song, self.songartist, self.artist


class Top2000Downloader:
    # Wikipedia redirects Anita Garbo to her song, instead of a page about her as an artist
    # Wikipedia does have an infobox for Space Monkey, but it is about their song, not about them as an artist
    ignored_artists = ['Anita Garbo',  
                       'Space Monkey',  
                      ]
    
    def _read_full_list(self):
        fulllist_reader = wikipediareader.TableExtractor(URL_TOP2000_HISTORY, [1], [0])
        df = fulllist_reader.extract_table_as_dataframe()
        return df
    
    def _clean(self, df):
        cleaner = Top2000Cleaner()
        ranking, song, songartist, artist = cleaner.clean(df)
        return ranking, song, songartist, artist
    
    def _download_infobox_details(self, links):
        result = []
        for link in tqdm.tqdm(links):
            info = wikipediareader.InfoboxReader(link, allow_errors=True).read()
            result.append(info)
        return pd.concat(result)
    
    def _add_detailed_information(self, artist):
        links = artist.loc[~artist['Name'].isin(self.ignored_artists), 'Link']
        artist_details = self._download_infobox_details(links)
        
        
        # The band members are much harder to handle because of all the functions they can have:
        # it gives a many-to-many relation for members and bands
        # So we ignore those. The same goes vice versa for the member pages in which it is discussed
        # in what bands they were active
        extra_artist_details = (artist_details[~artist_details['Header'].isin(['Leden', 'Oud-leden', 'Bezetting'])
                                               & ~artist_details['Header'].str.startswith('Actief')]  
                                .set_index(['OriginalLink', 'Variable'])['Value'].unstack()
                               )
        artist_full = artist.merge(extra_artist_details, left_on='Link', right_index=True, how='left')
        return artist_full
    
                  
    def download_and_write(self, output_folder='Output'):
        ranking, song, songartist, artist = self._read_full_list().pipe(self._clean)
        artist = self._add_detailed_information(artist).pipe(Top2000Cleaner.clean_artist)
        
        tables = {'ranking': ranking,
                  'song': song,
                  'songartist': songartist,
                  'artist': artist,
                 }
        for name, table in tables.items():
            table.to_parquet(os.path.join(output_folder, f'{name}.parquet'))