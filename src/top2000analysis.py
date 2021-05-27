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

    def __init__(self, votesmodel='Meindertsma'):

        votesmodels = {'Meindertsma': voteestimator.MeindertsmaVotesEstimator(),
                      'Exponential': voteestimator.ExponentialVotesEstimator(),
                      'Linear': voteestimator.LinearVotesEstimator(),
                      }
        self.votesmodel = votesmodels[votesmodel]

    def _combine_data(self, filefolder):
        self.ranking = pd.read_parquet(os.path.join(filefolder, 'ranking.parquet'))
        self.song = pd.read_parquet(os.path.join(filefolder, 'song.parquet'))
        self.songartist = pd.read_parquet(os.path.join(filefolder, 'songartist.parquet'))
        self.artist = (pd.read_parquet(os.path.join(filefolder, 'artist.parquet')) 
                          .pipe(self._artist_features, filefolder=filefolder) # TODO: This should not happen here
                        )

        df = (self.ranking.merge(self.song, left_on='SongID', right_index=True)
                          .merge(self.songartist.reset_index())
                          .merge(self.artist, left_on='ArtistID', right_index=True,
                                 suffixes=('Song', 'Artist'))
             )
        return df

    def _read_stemperiodes(self, filefolder='Data'):
        path = os.path.join(filefolder, 'EindeStemperiode.xlsx')
        einde_stemperiode = (pd.read_excel(path, engine='openpyxl')  # openpyxl does support xlsx
                               .dropna(subset=['EindeStemperiode'])
                               .drop(columns=['Bron'])
                               .sort_values('EindeStemperiode')
                            )
        return einde_stemperiode


    def _check_passed_away_during_top2000(self, df, top2000_stemperiodes):
        first_stemperiode = top2000_stemperiodes['EindeStemperiode'].min()
        relevant_date_of_death = first_stemperiode + pd.Timedelta('365 days')
        df['IsOverleden'] = df['Overlijdensdatum'].ge(relevant_date_of_death)
        return df

    def _find_next_top2000_after_death(self, df, top2000_stemperiodes):
        not_passed_away_during_top_2000 = df[~df['IsOverleden']].copy()
        passed_away_during_top2000 = (df.loc[df['IsOverleden']]
                                      .sort_values('Overlijdensdatum')
                                      .reset_index()
                                     )

        passed_away_during_top2000 = (pd.merge_asof(passed_away_during_top2000,
                                                    top2000_stemperiodes,
                                                   left_on='Overlijdensdatum',
                                                   right_on='EindeStemperiode',
                                                   direction='forward')
                                     .set_index('ArtistID')
                                     )
        df = pd.concat([not_passed_away_during_top_2000, passed_away_during_top2000], sort=False)
        return df


    def _artist_features(self, df, filefolder='Data'):
        einde_stemperiode = self._read_stemperiodes(filefolder)
        df = (df.pipe(self._check_passed_away_during_top2000, einde_stemperiode)
                .pipe(self._find_next_top2000_after_death, einde_stemperiode)
                .assign(AgePassing = lambda df: (df['Overlijdensdatum']
                                                 .sub(df['Geboortedatum']).dt.days
                                                 .div(365.25)),
                        PassingTooEarly = lambda df: df['AgePassing'].sub(80).mul(-1).clip(lower=0),
                        IsDutch = lambda df: df['IsDutch'].astype(int),
                        )
             )
        return df

    def _normalize_by_years_before_death(self, df, years_to_normalize=1):
        mi = pd.MultiIndex.from_product([df.query('IsOverleden')['SongID'].unique(),
                                         df.query('IsOverleden')['YearsSinceOverlijden'].unique(),],
                                        names=['SongID', 'YearsSinceOverlijden'])
        votes_before_death = (pd.DataFrame(index=mi)
                              .join(self.songartist)
                              .join(df.set_index(['SongID', 'YearsSinceOverlijden', 'ArtistID'])[['Year', 'PctVotes']])
                              .join(self.artist[['JaarTop2000']])
                              .join(self.song[['YearMade']])
                              .assign(YearTop2000 = lambda df: df['JaarTop2000'].add(
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

        df = df.merge(votes_before_death, right_index=True, how='left', validate='many_to_one')
        return df
    def _add_rank_last_year(self, df):
        ranklastyear = self.notering.set_index(['SongID', 'Year'])
            .unstack().stack(dropna=False)
            .groupby('SongID')['Rank'].shift()
            )
        
        return df.merge(ranklastyear, how='left', right_index=True, validate='many_to_one')

    def _song_features(self, df):

        df = (df.assign(NrArtists = lambda df: df.groupby(['SongID', 'Year'])['Rank'].transform('count'),
                        YearsSinceOverlijden = lambda df: df['Year'].sub(df['JaarTop2000']),
                        PctVotes = lambda df: df['Rank'].apply(self.votesmodel.percentage_of_votes),
                       )
                .pipe(self._add_rank_last_year)
                .pipe(self._normalize_by_years_before_death)
             )
        return df

    def _song_features_after_passing(self, df):
        df = (df.assign(Boost = lambda df: df['PctVotesAfterDeath'].div(df['PctVotesBeforeDeath']),
                        LogBoost = lambda df: np.log(df['Boost']),
                        
                        
                        PopularityWithinArtist = lambda df: (df.groupby('ArtistID')['PctVotesBeforeDeath']
                                                             .apply(lambda v: v.div(v.mean()))),
                        LogSongPopularityWithinArtist = lambda df: np.log10(df['PopularityWithinArtist']),
                        
                        RecencyWithinArtist = lambda df: (df.groupby('ArtistID')['YearMade']
                                                          .apply(lambda v: v.sub(v.min()).div(v.max() - v.min()))),
                        YearsBeforeDeath = lambda df: df['YearMade'].sub(df['JaarTop2000']),
                        JarenGeleden = lambda df: df['JaarTop2000'].sub(df['JaarTop2000'].max()),
                        
                        MultiplePerformers = lambda df: df['NrArtists'].gt(1).astype(int),
                        )
             )
        return df

    def create_analysis_set(self, filefolder):
        df = (self._combine_data(filefolder)
                  .pipe(self._song_features)
                  .query('YearsSinceOverlijden == 0')
                  .rename(columns={'PctVotes': 'PctVotesAfterDeath'})
                  .query(f'PctVotesBeforeDeath > {self.votesmodel.lower_than_2000}')
                  .pipe(self._song_features_after_passing)
             )
        return df

    def create_artist_set(self, filefolder):
        columns = ['Name',
                   'IsDutch',
                   'AgePassing',
                   'JaarTop2000',
                   'Overlijdensdatum',
                   'EindeStemperiode',
                   ]

        df = self.create_analysis_set(filefolder)
        df_artist = (df.groupby('ArtistID')
                        .agg(PctVotesAfterDeath = ('PctVotesAfterDeath', 'sum'),
                             PctVotesBeforeDeath = ('PctVotesBeforeDeath', 'sum'),
                             LastYearInTop2000 = ('YearMade', 'last'),
                             NrsBeforeDeath = ('ArtistID', 'count')
                            )
                        .join(self.artist[columns])
                        .assign(DaysToStemperiode = lambda df: (df['Overlijdensdatum']
                                                                .sub(df['EindeStemperiode']).dt.days),
                                YearsSinceLastHit = lambda df: df['JaarTop2000'].sub(df['LastYearInTop2000']),
                                LogPopularity = lambda df: np.log10(df['PctVotesBeforeDeath']),
                                LogPopularityNorm = lambda df: (df['LogPopularity']
                                                                .sub(df['LogPopularity'].median())),
                                Boost = lambda df: df['PctVotesAfterDeath'].div(df['PctVotesBeforeDeath']),
                                LogBoost = lambda df: np.log(df['Boost']),
                                )
                    )
        return df_artist

    def create_full_feature_set(self, filefolder):
        df = self.create_analysis_set(filefolder)
        df_artist = self.create_artist_set(filefolder)
        columns = ['Name',
                   'IsDutch',
                   'AgePassing',
                   'JaarTop2000',
                   'Overlijdensdatum',
                   'EindeStemperiode',
                   ]
        
        full_set = (df.drop(columns=columns)  # Use duplicate columns from artist
                      .merge(df_artist, left_on='ArtistID', right_index=True,
                             suffixes=('Song', 'Artist'))
                      .assign(
                              SongRelativeBoost = lambda df: df['BoostSong'].div(df['BoostArtist']),
                              LogRelativeBoost = lambda df: np.log2(df['SongRelativeBoost']),
                              LogBoost = lambda df: np.log(df['BoostSong']),
                             )
           )
        return full_set
    
    
class BoostExplainer:
    
    def __init__(self, parameters, idata):
        self.parameters = parameters
        self.idata = idata
        self.df_songs = idata.constant_data.to_dataframe().reset_index().query('Artist == artist_idx')
        self.artist_boost = idata.posterior['za_artist'].to_dataframe().median(level='Artist')
        
    def _get_song(self, ix):
        song = self.df_songs.query('obs_id == @ix').squeeze().to_dict()
        return song
    
    def _calculate_artist_effect(self, song):
        base_boost = np.exp(self.parameters['a'])
        history_effect = np.exp(self.parameters['history_effect'] * song['jaren_geleden'] )
        recency_effect = np.exp((np.exp(10**self.parameters['recency_effect_exponent'] * song['days_to_stemperiode'])
                                 - np.exp(10**self.parameters['recency_effect_exponent'] * -365))
                                * self.parameters['max_recency_effect'])
        popularity_effect = np.exp(self.parameters['effect_popularity'] * song['logpopularity'])
        dutch_effect = np.exp(self.parameters['is_dutch_effect'] * song['is_dutch'])
        age_effect = np.exp(self.parameters['age_passing_effect'] * song['passing_too_early'])
        
        artist_idx = song['artist_idx']
        artist_magic = np.exp(self.parameters['sigma_a'] * self.artist_boost.loc[artist_idx].values[0])
        effects = {
                    'Base': base_boost,
                    'History': history_effect,
                    'Popularity': popularity_effect,
                    'Dutch': dutch_effect,
                    'PassingTooEarly': age_effect,
                    'Recency': recency_effect,
                    'ArtistMagic': artist_magic
                    }
        return effects
    
    
    def all_effects(self, song_idx, difference=False, idata=None):
        song = self._get_song(song_idx)
        effects_artist = self._calculate_artist_effect(song)
        effects_song = self._calculate_song_effects(song)
        effects = (pd.concat([pd.concat({'Artist': pd.Series(effects_artist)}),
                            pd.concat({'Song': pd.Series(effects_song)})])
                   .rename('EffectSize')
                  .to_frame()
                  )
        if difference:
            effects.loc[('Prediction', ''), 'EffectSize'] = 1
            effects.loc[('FinalDifference', ''), 'EffectSize'] = self.get_factor_off(song_idx, idata)
        return effects
    
    def _calculate_song_effects(self, song):
        oeuvre_effect = np.exp(self.parameters['within_oeuvre_effect'] * song['popularity_within_oeuvre'])
        sharing_effect = np.exp(self.parameters['sharing_effect'] * song['multiple_performers'])
        effects = {
                    'WithinOeuvrePopularity': oeuvre_effect,
                    'MultiplePerformers': sharing_effect
                    }
        return effects
    
    def _print_effects(self, effects, starting_point=1):
        effect = starting_point
        for name, size in effects.items():
            effect *= size
            print(f'The effect is {effect:.2f} after {name} - distinct effect: {size:.2f}')
            
    def explain(self, song):
        """
        Explain the boost of a song
        
        Ignores the artist specific boost, which we cannot know beforehand
        
        Parameters
        ----------
        song: int or str
            int: the index of the song in df
            str: the title of the song (must be unique)
        """
        song = self._get_song(song)
        print('\033[1m' + f"{song.loc['Title']} by {song.loc['NameArtist']}" + '\033[0m')  # Using bold face
        effects_artist = self._calculate_artist_effect(song)
        total_artist_effect = np.prod(list(effects_artist.values()))
        effects_song = self._calculate_song_effects(song)
        self._print_effects(effects_artist)
        print('-')
        self._print_effects(effects_song, total_artist_effect)
        print('')
        print(f'The actual boost was {song.loc["BoostSong"]:.2f}')
    
    def get_factor_off(self, song, idata):
        boost = np.exp(idata.observed_data['y_like'].values[song])
        predicted_boost = self.all_effects(song)['EffectSize'].prod()
        factor = boost / predicted_boost
        return factor

    