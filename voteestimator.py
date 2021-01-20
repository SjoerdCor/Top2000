# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 17:03:01 2021

@author: Gebruiker
"""

class VotesEstimator:
    def __init__(self, nr_songs=3000):
        self.position_votes = {}
        self.total_votes = sum(self.votes_per_position(nr) for nr in range(1, nr_songs))
        self.lower_than_2000 = 0.5 * self.percentage_of_votes(2000)

    def votes_per_position(self, position):
        try:
            return self.position_votes[position]
        except KeyError:
            nr_votes = self._calculate_position_votes(position)
            self.position_votes[position] = nr_votes
            return nr_votes

    def percentage_of_votes(self, position):
        return self.votes_per_position(position) / self.total_votes

class MeindertsmaVotesEstimator(VotesEstimator):
    """
    Calculate the number of votes at a certain position of the Top 2000 according to a model developed by Peter Meindertsma

    Source: https://www.petermeindertsma.nl/blog/benadering-aantal-stemmen-per-liedje-in-de-top-2000-van-2014/
    """

    def __init__(self, votes_first_place=10000, diff_factor=30):
        self.votes_first_place = votes_first_place
        self.diff_factor = diff_factor
        super().__init__()

    def _calculate_position_votes(self, position):

        nr_votes = self.votes_first_place / (1 + ( (position - 1) / self.diff_factor))
        return nr_votes

class ExponentialVotesEstimator(VotesEstimator):
    """
    Calculate the number of votes at a certain position of the Top 2000 according to a
    """
    def __init__(self, votes_first_place=10000, rho=0.55):
        self.votes_first_place = votes_first_place
        self.rho = rho
        super().__init__()

    def _calculate_position_votes(self, position):

        nr_votes = self.votes_first_place / (position ** self.rho)
        return nr_votes