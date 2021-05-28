# -*- coding: utf-8 -*-
"""
Created on Fri May 28 08:23:34 2021

@author: Gebruiker
"""

class TableExtractor:

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
        attrs : dict
            passed to BeautifulSoup: dictionary of attributes that you can pass
            to use to identify the table in the HTML
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
        rename_dict : dct
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

        df_links = pd.DataFrame(links_data)
        df = pd.concat([self.dataframe, df_links], axis='columns')
        return df



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
        return self.details
    
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
                                    .assign(Header = lambda df: pd.Series(np.where(df['Variable'] == df['Value'],
                                                                                   df['Variable'], None)).ffill(),
                                            OriginalLink = self.link
                                           )
                                      .loc[lambda df: df['Variable'] != df['Value']]
                                 )
    
    def read(self):
        self.download()
        self.clean()
        return self.details


def wikipedia_datetime_to_datetime(series):
    # We download from dutch wikipedia, so we need dutch month names
    locale.setlocale(locale.LC_TIME, 'nl_NL.utf8')

    return pd.to_datetime(series, errors='coerce', format='%d %B %Y', exact=False)

