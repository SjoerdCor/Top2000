"""
Created on Fri Jan  8 10:50:11 2021

@author: Gebruiker
"""

import datetime
from typing import Iterable
import locale

import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np


class WikipediaTableExtractor:

    def __init__(self, link: str,
                 single_link_columns: Iterable[int] = None,
                 multiple_link_columns: Iterable[int] = None,
                 attrs: dict = None,
                 table_index: int = 0,
                 ):
        '''
        Extract Wikipedia Table, adding underlying wikipedia redirects as extra columns

        Parameters
        ----------
        link : str
            the wikipedia link where the table should be extracted; it is always
            the first table that is extracted
        single_link_columns : iterable of ints
            indexes for columns that contain a single wikipedia link
        multiple_link_columns : iterable of ints
            index(es) for columns that (may) contain multiple wikipedia links
            these columns are put into tuples of length 2, with the first
            element being the text for each link and the second being the underlying link
        table_index : int
            Which table from wikipedia to use

        Returns
        -------
        None.

        '''
        self.link = link
        self.single_link_columns = single_link_columns or []
        self.multiple_link_columns = multiple_link_columns or []
        self.attrs = attrs
        self.table_index = table_index
        self.table = None
        self.dataframe = None

    def get_html_soup(self):
        response = requests.get(self.link)
        html = response.content
        soup = BeautifulSoup(response.content, 'html.parser')
        for br in soup.find_all("br"):
            br.replace_with(", " + br.text)

        return soup

    def read_df(self):
        '''
        Reads the table as a pandas dataframe

        Returns
        -------
        None.

        '''
        soup = self.get_html_soup()
        self.dataframe = pd.read_html(str(soup), attrs=self.attrs)[self.table_index]

    def find_table(self):
        '''
        Find the soup of the table

        Returns
        -------
        None.

        '''
        soup = self.get_html_soup()
        # soup = BeautifulSoup(html, 'html.parser')
        table = soup.findAll('table', attrs=self.attrs)[self.table_index]
        self.table = table

    def find_wikipedia_links(self):
        '''
        Return dictionary with extra columns for each column containing links

        Returns
        -------
        data : dict
        '''
        if self.table is None:
            self.find_table()

        data = {i: [] for i in self.single_link_columns + self.multiple_link_columns}
        for tablerow in self.table.findAll("tr"):
            cells = tablerow.findAll("td")
            if not cells:  # header
                continue
            
            for i in self.single_link_columns:
                try:
                    cell = cells[i]
                except IndexError:
                    data[i].append(None)
                    continue
                try:
                    data[i].append(cell.find('a')['href'])
                except TypeError:
                    data[i].append(None)
            for i in self.multiple_link_columns:
                try:
                    cell = cells[i]
                except IndexError:
                    data[i].append(None)
                    continue
                try:
                    info = tuple((l.get_text(), l['href']) for l in cell.find_all('a'))
                except TypeError:
                    info = None
                data[i].append(info)

        columnnames_dict = self.find_correct_columnnames()
        for i, column_name in columnnames_dict.items():
            data[column_name] = data.pop(i)

        return data

    def find_correct_columnnames(self):
        '''
        Rename new link columns from column index to a readable human name

        Append "Link" to columns which contain only a single link
        Append "Links" to columns which contain multiple links

        Returns
        -------
        rename_dict : TYPE
            DESCRIPTION.

        '''
        #TODO: check whether we do not create duplicate column names
        rename_dict = {}
        if self.dataframe is None:
            self.read_df()
        for i in self.single_link_columns:
            rename_dict[i] = str(self.dataframe.columns[i]) + 'Link'
        for i in self.multiple_link_columns:
            rename_dict[i] = str(self.dataframe.columns[i]) + 'Links'
        return rename_dict

    def extract_table_as_dataframe(self):
        '''
        Reads table with extra columns

        Returns
        -------
        df : pd.DataFrame

        '''
        if self.dataframe is None:
            self.read_df()

        links_data = self.find_wikipedia_links()
        columnnames_dict = self.find_correct_columnnames()

        df_links = pd.DataFrame(links_data)
        df = pd.concat([self.dataframe, df_links], axis='columns')
        return df


class Top2000Cleaner:
    def __init__(self, data: pd.DataFrame):
        locale.setlocale(locale.LC_TIME, 'nl_NL.utf8') # We download from dutch wikipedia, so we need dutch month names

        self.data = data
        self._validate()

    def _check_column_completeness(self):
        current_year = datetime.datetime.now().year
        columns_should_be_present = ['Artiest', 'Titel', 'Jaar', 'HP', 'TitelLink', 'ArtiestLinks']
        columns_should_be_present += [str(i%100).zfill(2) for i in range(1999, current_year) ]
        assert all(col in self.data.columns for col in columns_should_be_present)
    def _validate(self):
        assert self.data.notnull().all().all()
        self._check_column_completeness()
    def rename_columns(self):
        column_rename =  {'Titel': 'Title',
                          'Artiest': 'Artist',
                          'Jaar': 'YearMade',
                          'TitelLink': 'TitleLink',
                          'ArtiestLinks': 'ArtistLinks',
                          }
        self.data = self.data.rename(columns = column_rename)
    def split_into_model(self):
        '''
        Split data into tables according to data model
        '''

        self.data['SongID'] = (self.data[['Title', 'Artist']].applymap(lambda x: x.lower())
                               .apply(lambda x: hash(tuple(x)), axis=1)
                               )
        self.notering = self.data[['SongID'] + [col for col in self.data.columns if col.isnumeric()]]
        self.song = self.data.set_index('SongID')[['Title', 'YearMade', 'TitleLink']]

        exploded = self.data.explode('ArtistLinks')
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
        return self.notering, self.song, self.songartist, self.artist

    def clean_notering(self):
        self.notering = (self.notering.melt(id_vars=['SongID'], var_name='Year', value_name='Rank')
                                     .assign(Year = lambda df: (df['Year'].astype(int)
                                                                .add(2000)
                                                                .mask(lambda s: s.ge(2050),
                                                                      lambda s: s.sub(100))
                                                                ),
                                             Rank = lambda df: pd.to_numeric(df['Rank'],
                                                                             errors='coerce',
                                                                             downcast='integer')
                                             )
                                     .dropna()
                                     )

    def validate_notering(self):
        assert (self.notering.groupby('Year')['Rank'].apply(set) == set(range(1, 2001))).all()
        assert self.notering['Year'].ge(self.notering['YearMade']).all()
        assert self.notering.groupby(['Title', 'Artist'])['YearMade'].nunique().eq(1).all()
        

    def clean_song(self):
        self.song = (self.song.rename(columns={'TitleLink': 'Link'})
                          .assign(Link = lambda df: 'https://nl.wikipedia.org' + df['Link'])
                    )
    
    def clean_artist(self):
        # We download from dutch wikipedia, so we need dutch month names
        locale.setlocale(locale.LC_TIME, 'nl_NL.utf8') 
        self.arist = self.artist.assign(Overleden = lambda df: pd.to_datetime(df['Overleden'],
                                                                              errors='coerce',
                                                                              format='%d %B %Y',
                                                                              exact=False)
                                        )


    
    def clean(self):
        self.rename_columns()
        self.split_into_model()
        self.clean_notering()
        self.validate_notering()
        self.clean_song()
        self.clean_artist()
        return self.notering, self.song, self.songartist, self.artist

class InfoboxReader:
    def __init__(self, link, allow_errors=False):
        self.link = link
        self.details = None
        self.allow_errors = allow_errors
    
    def download(self):
        reader = WikipediaTableExtractor(self.link, [1], attrs={'class': 'infobox'})
        try:
            artist_details = reader.extract_table_as_dataframe()
        except ValueError as e:  # No table found
            if self.allow_errors:
                self.details = pd.DataFrame()
                return self.details
            raise e
        assert artist_details.columns.tolist() == [0, 1, 2, '1Link']
        self.details = artist_details
    
    def clean(self):
        if self.details is None:
            self.download()
        elif self.details.empty:
            return self.details
        self.details = (self.details.rename(columns={0: 'Variable', 1: 'Value', 2: 'Extra'})
                                    # Fill nans so we can check whether Value and Extra are indeed identical as expected
                                    .assign(Variable = lambda df: df['Variable'].fillna(''),
                                            Value = lambda df: df['Value'].fillna(''),
                                            Extra = lambda df: df['Extra'].fillna('')
                                            )
                                     .iloc[:-1]  # Drop Portaal information
                        )
        assert (self.details['Value'] == self.details['Extra']).all()
        self.details = (self.details.drop(columns=['Extra'])  # Drop headers
                                    .assign(Header = lambda df: pd.Series(np.where(df['Variable'] == df['Value'], df['Variable'], None)).ffill(),
                                            OriginalLink = self.link
                                           )
                                      .loc[lambda df: df['Variable'] != df['Value']]
                                 )
    
    def read(self):
        self.download()
        self.clean()
        return self.details
