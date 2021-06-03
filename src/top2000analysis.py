# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 08:06:30 2021

@author: Gebruiker
"""
import os
import numpy as np
import pandas as pd
import voteestimator

class AnalysisSetCreator:
    votesmodels = {'Meindertsma': voteestimator.MeindertsmaVotesEstimator,
                   'Exponential': voteestimator.ExponentialVotesEstimator,
                   'Linear': voteestimator.LinearVotesEstimator,
                   }

    def __init__(self, filefolder, votesmodel='Meindertsma'):

        self.votesmodel = self.votesmodels[votesmodel]()
        self._read_base_tables(filefolder)

        self.analysis_song = None
        self.analysis_artist = None
        self.analysis_total = None

    def _read_base_tables(self, filefolder):
        self.ranking = pd.read_parquet(os.path.join(filefolder, 'ranking.parquet'))
        self.song = pd.read_parquet(os.path.join(filefolder, 'song.parquet'))
        self.songartist = pd.read_parquet(os.path.join(filefolder, 'songartist.parquet'))
        self.artist = (pd.read_parquet(os.path.join(filefolder, 'artist.parquet')))
        self.artist_incl_features = self.artist.pipe(self._artist_features, filefolder)

    def combine_data(self, artist_features=False):

        artist = self.artist_incl_features if artist_features else self.artist

        df = (self.ranking.merge(self.song, left_on='SongID', right_index=True)
                          .merge(self.songartist.reset_index())
                          .merge(artist, left_on='ArtistID', right_index=True,
                                 suffixes=('Song', 'Artist'))
             )
        return df

    def _read_stemperiodes(self, filefolder='Data'):
        path = os.path.join(filefolder, 'VotingPeriod.xlsx')
        einde_stemperiode = (pd.read_excel(path, engine='openpyxl')  # openpyxl does support xlsx
                               .dropna(subset=['DateEndOfVoting'])
                               .drop(columns=['Source'])
                               .sort_values('DateEndOfVoting')
                            )
        return einde_stemperiode


    def _check_passed_away_during_top2000(self, df, top2000_stemperiodes):
        first_stemperiode = top2000_stemperiodes['DateEndOfVoting'].min()
        relevant_date_of_death = first_stemperiode + pd.Timedelta('365 days')
        df['HasPassed'] = df['DateOfDeath'].ge(relevant_date_of_death)
        return df

    def _find_next_top2000_after_death(self, df, top2000_stemperiodes):
        not_passed_away_during_top_2000 = df[~df['HasPassed']].copy()
        passed_away_during_top2000 = (df.loc[df['HasPassed']]
                                      .sort_values('DateOfDeath')
                                      .reset_index()
                                     )

        passed_away_during_top2000 = (pd.merge_asof(passed_away_during_top2000,
                                                    top2000_stemperiodes,
                                                    left_on='DateOfDeath',
                                                    right_on='DateEndOfVoting',
                                                    direction='forward')
                                     .rename(columns={'YearTop2000': 'Top2000YearOfDeath'})
                                     .set_index('ArtistID')
                                     )
        df = pd.concat([not_passed_away_during_top_2000, passed_away_during_top2000], sort=False)
        return df


    def _artist_features(self, df, filefolder='Data'):
        einde_stemperiode = self._read_stemperiodes(filefolder)

        columns_possible_country_of_birth = ['Oorsprong',
                                             'Land',
                                             'Land van oorsprong',
                                             'Geboorteland',
                                             'Geboorteplaats',
                                             'Nationaliteit'
                                            ]

        df = (df.pipe(self._check_passed_away_during_top2000, einde_stemperiode)
                .pipe(self._find_next_top2000_after_death, einde_stemperiode)
                .assign(AgePassing = lambda df: (df['DateOfDeath']
                                                 .sub(df['DateOfBirth']).dt.days
                                                 .div(365.25)),
                        PassingTooEarly = lambda df: df['AgePassing'].sub(80).mul(-1).clip(lower=0),
                        IsDutch = lambda df: (df.filter(columns_possible_country_of_birth, axis='columns')
                                              .apply(lambda c: c.str.lower()
                                                                .str.contains('nederland'))
                                              .any('columns').astype(int)
                                              ),
                        DaysToEndOfVoting = lambda df: (df['DateOfDeath'].sub(df['DateEndOfVoting']).dt.days),
                        )
             )
        return df

    def _normalize_by_years_before_death(self, df, years_to_normalize=1):
        mi = pd.MultiIndex.from_product([df.query('HasPassed')['SongID'].unique(),
                                         df.query('HasPassed')['YearsSinceOverlijden'].unique(),],
                                        names=['SongID', 'YearsSinceOverlijden'])

        votes_before_death = (pd.DataFrame(index=mi)
                              .join(self.songartist)
                              .join(df.set_index(['SongID', 'YearsSinceOverlijden', 'ArtistID'])[['Year', 'PctVotes']])
                              .join(self.artist_incl_features[['Top2000YearOfDeath']])
                              .join(self.song[['YearMade']])
                              .assign(YearTop2000 = lambda df: df['Top2000YearOfDeath'].add(
                                                              df.index.get_level_values('YearsSinceOverlijden')),
                                      PctVotes = lambda df: np.where(df['YearTop2000'].gt(df['YearMade'])
                                                                     & df['YearTop2000'].le(df['Year'].max()),
                                                             df['PctVotes'].fillna(self.votesmodel.lower_than_2000),
                                                             np.nan)
                                     )
                              ['PctVotes']
                              .unstack('YearsSinceOverlijden')
                              .loc[:, range(-years_to_normalize, 0)]
                              .mean(axis='columns')
                              .rename('PctVotesBeforeDeath')
                              )

        df = df.merge(votes_before_death, how='left',
                      left_on=['SongID', 'ArtistID'],  right_index=True,
                      validate='many_to_one')
        return df

    def _add_rank_last_year(self, df):
        ranklastyear = (self.ranking.set_index(['SongID', 'Year'])
                            .unstack().stack(dropna=False)
                            .groupby('SongID')['Rank'].shift()
                            .rename('RankLastYear')
                            )
        df = df.merge(ranklastyear, how='left',
                      left_on=['SongID', 'Year'], right_index=True,
                      validate='many_to_one')
        return df

    def _song_features(self, df):
        df = df.assign(PctVotes = lambda df: df['Rank'].apply(self.votesmodel.percentage_of_votes))


        # Doing this beforehand instead of in an `apply` speeds up 20x
        avgpctvotesartist = df.groupby(['ArtistID', 'Year'])['PctVotes'].transform('mean')
        earliesttop2000song = df.groupby(['ArtistID', 'Year'])['YearMade'].transform('min')
        latesttop2000song = df.groupby(['ArtistID', 'Year'])['YearMade'].transform('max')
        top2000span = latesttop2000song - earliesttop2000song

        df = (df.assign(NrArtists = lambda df: df.groupby(['SongID', 'Year'])['Rank'].transform('count'),
                        MultiplePerformers = lambda df: df['NrArtists'].gt(1).astype(int),
                        YearsSinceOverlijden = lambda df: df['Year'].sub(df['Top2000YearOfDeath']),

                        PopularityWithinArtist = lambda df: df['PctVotes'].div(avgpctvotesartist),
                        LogSongPopularityWithinArtist = lambda df: np.log10(df['PopularityWithinArtist']),
                        RecencyWithinArtist = lambda df: (df['YearMade'].sub(earliesttop2000song)
                                                          .div(top2000span)
                                                          .fillna(0.5)
                                                          ),
                       )
                .pipe(self._add_rank_last_year)
                .pipe(self._normalize_by_years_before_death)
             )
        return df

    def _song_features_after_passing(self, df):
        df = (df.assign(Boost = lambda df: df['PctVotesAfterDeath'].div(df['PctVotesBeforeDeath']),
                        LogBoost = lambda df: np.log(df['Boost']),

                        YearsBeforeDeath = lambda df: df['YearMade'].sub(df['Top2000YearOfDeath']),
                        JarenGeleden = lambda df: df['Top2000YearOfDeath'].sub(df['Top2000YearOfDeath'].max()),
                        )
             )
        return df

    def create_analysis_set(self):
        df = (self.combine_data(artist_features=True)
                  .pipe(self._song_features)
                  .query('YearsSinceOverlijden == 0')
                  .rename(columns={'PctVotes': 'PctVotesAfterDeath'})
                  .query(f'PctVotesBeforeDeath > {self.votesmodel.lower_than_2000}')
                  .pipe(self._song_features_after_passing)
             )
        self.analysis_song = df
        return self.analysis_song

    def create_artist_set(self):
        columns = ['Name',
                   'IsDutch',
                   'AgePassing',
                   'Top2000YearOfDeath',
                   'DateOfDeath',
                   'DaysToEndOfVoting',
                   ]
        if self.analysis_song is None:
            self.create_analysis_set()
        df_artist = (self.analysis_song.groupby('ArtistID')
                                       .agg(PctVotesAfterDeath = ('PctVotesAfterDeath', 'sum'),
                                            PctVotesBeforeDeath = ('PctVotesBeforeDeath', 'sum'),
                                            LastSongYearInTop2000 = ('YearMade', 'last'),
                                            NSongsBeforeDeath = ('ArtistID', 'count')
                                            )
                                       .join(self.artist_incl_features[columns])
                                       .assign(LogPopularity = lambda df: np.log10(df['PctVotesBeforeDeath']),
                                               LogPopularityNorm = lambda df: (df['LogPopularity']
                                                                               .sub(df['LogPopularity'].median())),
                                               Boost = lambda df: (df['PctVotesAfterDeath']
                                                                   .div(df['PctVotesBeforeDeath'])),
                                               LogBoost = lambda df: np.log(df['Boost']),
                                               YearsSinceLastHit = lambda df: (df['Top2000YearOfDeath']
                                                                               .sub(df['LastSongYearInTop2000'])),

                                               )
                    )
        self.analysis_artist = df_artist
        return self.analysis_artist

    def create_full_feature_set(self):
        columns = ['Name',
                   'IsDutch',
                   'AgePassing',
                   'Top2000YearOfDeath',
                   'DateOfDeath',
                   'DaysToEndOfVoting',
                   ]
        if self.analysis_song is None:
            self.create_analysis_set()
        if self.analysis_artist is None:
            self.create_artist_set()


        full_set = (self.analysis_song.drop(columns=columns)  # Use duplicate columns from artist
                                      .merge(self.analysis_artist, left_on='ArtistID', right_index=True,
                                             suffixes=('Song', 'Artist'))
                                      .assign(
                                              SongRelativeBoost = lambda df: df['BoostSong'].div(df['BoostArtist']),
                                              LogRelativeBoost = lambda df: np.log(df['SongRelativeBoost']),
                                              LogBoost = lambda df: np.log(df['BoostSong']),
                                             )
           )
        return full_set
