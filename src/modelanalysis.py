# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

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
