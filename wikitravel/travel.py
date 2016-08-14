import numpy as np
import pandas as pd
import json
import urllib2
import wikipedia as wk
import googlemaps as gm
import networkx
import itertools


# TODOs
# Build in recrusion to build graphs off initital graph and link

class Travel(object):
    """
    Class for finding nearby sites of interest via google maps and wikipedia
    Site popularity evaluated by page views in the past (or specified month)

    Create Travel object based on location, factors + weights (indev)

    :param str location: string specifying location, default is geocode
    :param str gkey: Google Maps api key
    :param int results: max number of wiki results to query for site
    :param int radius: search radius (meters)
    :param str popDate: month to grab wiki page views from (YYYYMM) \
            note month pages must exist at http://stats.grok.se/
    :param dict factors: additional criteria INDEV
    :param dict weights: additional criteria INDEV
    """

    def __init__(self, location, gkey, numResults=15, radius=1000,
                 popDate='201506', factors=None, weights=None):

        self._location = location
        self._gkey = gkey
        self._numResults = numResults
        self._radius = radius
        self._popDate = popDate
        self._factors = factors
        self._weights = weights

        # Populated by methods
        self.coordinates = []
        self.df = pd.DataFrame()
        self.wiki_resp = None

    @property
    def location(self):
        """
        Initial location
        """
        return self._location

    @property
    def numResults(self):
        """
        Number of restults to search for in each wiki geocode search
        """
        return self._numResults

    @property
    def radius(self):
        """
        Search radius for wiki geocode search
        """
        return self._radius

    @property
    def popDate(self):
        """
        YYYYMM for popularity search (if applicable)
        """
        return self._popDate

    @property
    def factors(self):
        """
        (IN DEV)
        """
        return self._factors

    @property
    def weights(self):
        """
        (IN DEV)
        """
        return self._weights

    def locate(self):
        """
        Find major site location: self.coordinates = (latitude, longitude)
        """

        # Check if wiki page exists to grab meta data from
        try:
            self.wiki = wk.WikipediaPage(self.location)
            [latitude, longitude] = list(self.wiki.coordinates)

        # If not check with google maps
        except wk.PageError:

            # Initialize google maps client
            maps = gm.Client(self.gkey)
            loc_resp = maps.geocode(self.location)

            (latitude, longitude) = loc_resp[0]['geometry']\
                                               ['location'].values()

        self._coordinates = (latitude, longitude)

    def local_search(self):
        """
        Find nearby sites to self.location
        """

        # Search Wikipedia for nearby pages
        self.wiki_resp = wk.geosearch(self._coordinates[0],
                                      self._coordinates[1],
                                      results=self.results,
                                      radius=self.radius)

    def build_df(self):
        """
        Build :py:class:`pandas.DataFrame` sorted by popularity containing

        * name
        * page views
        * latitude
        * longitude

        """
        # Set source for popularity information
        pop_url = 'http://stats.grok.se/json/en/' + self.popDate + '/'

        index = self.wiki_resp
        columns = ['latitude', 'longitude', 'pageViews']
        self.df = pd.DataFrame(index=index, columns=columns)

        for resp in self.wiki_resp:
            # Get Wikipedia page
            wiki_page = wk.WikipediaPage(resp)
            wiki_url_tag = wiki_page.url.split('/')[-1]

            # Get popularity
            url_resp = urllib2.urlopen(pop_url + wiki_url_tag)
            json_resp = json.load(url_resp)
            numViews = sum(json_resp['daily_views'].itervalues())

            # Try Except Patch, with google maps (needs new method)
            self.df['latitude'][resp] = wiki_page.coordinates[0]
            self.df['longitude'][resp] = wiki_page.coordinates[1]
            self.df['pageViews'][resp] = numViews

        self.df = self.df.sort(['pageViews'], ascending=False)

    def user_select(self):
        """
        User selects additional site to include for trip
        """

        print(self.df)
        print('Select indices of places to visit i.e. 1 2 4 :')

        indices = []
        index_input = raw_input()
        for i in index_input:
            try:
                indices.append(int(i))
            except:
                pass

        logical = np.in1d(self.df.index, indices)

        self.df = self.df[logical]

        print('Your trip selections:')
        print(self.df)


class Trip(object):
    """
    Class for optimizing paths from Travel

    Constructor for Trip class, requries Travel & Google Maps api key

    :param Travel travel: Travel object (with df attribute)
    :param str start: Starting point of trip
    :param str end: Ending point of trip. Defaults to start
    """

    def __init__(self, travel, start, end=None):

        self.travel = travel
        self.start = start
        self.sites = travel.df['place'].values
        if end is None:
            self.end = start
            self.waypoints = len(self.sites) + 1
        else:
            self.end = end
            self.waypoints = len(self.sites)

        self.graph = networkx.Graph()

        self.maps = gm.Client(self.travel.gkey)

        # Assigned in methods
        self.distances = []
        self.routes = []
        self.cost = None
        self.route = None
        self.directions = []

    def build_graph(self):
        """
        Link all locations together in graph (completely connected)
        """
        if len(self.sites) > 2:
            self.graph.add_nodes_from(self.sites)
            self.edges = itertools.combinations(self.sites, 2)
            self.calculate_distances()
            self.graph.add_weighted_edges_from(self.distances)

        else:
            raise Exception('Not a trip, fewer then 3 sites: ' + self.sites)

    def calculate_distances(self):
        """
        Calculate distances between sites to weight graph 
        """
        for i in self.edges:
            l = np.in1d(self.travel.df['place'].values, i)
            coords = self.travel.df[['latitude', 'longitude']].values
            coords = coords[l, :]
            coords1 = {'lat': coords[0, 0], 'lng': coords[0, 1]}
            coords2 = {'lat': coords[1, 0], 'lng': coords[1, 1]}

            self.distances.append((i[0], i[1],
                self.maps.distance_matrix(coords1, coords2, mode='walking')\
                ['rows'][0]['elements'][0]['distance']['value']))

    def find_routes(self):
        """
        Find shortest route for given constraints
        """

        routes = networkx.all_simple_paths(self.graph, source=self.start,
                                           target=self.end,
                                           cutoff=self.waypoints
                                           )

        for route in routes:
            if len(route) == self.waypoints:
                self.routes.append(route)
                cost = self.calculate_cost(route)
                if not self.cost:
                    self.cost = cost
                    self.route = route
                elif cost < self.cost:
                    self.cost = cost
                    self.route = route

    def calculate_cost(self, route):
        """
        Calculate the cost of a given route

        :param array-like route: order of waypoints in route to evaluate

        :return cost: cost of route based on weights in graph
        :rtype: float
        """
        starts = route[:-1]
        ends = route[1:]
        cost = 0.

        for i, j in zip(starts, ends):
            cost += self.graph[i][j]['weight']

        return cost

    def get_directions(self):
        """
        Get directions from Google Maps for route
        """

        for i, k in zip(self.route[:-1], self.route[1:]):
            l = np.in1d(self.travel.df['place'].values, np.array([i, k]))
            coords = self.travel.df[['latitude', 'longitude']].values
            coords = coords[l, :]
            coords1 = {'lat': coords[0, 0], 'lng': coords[0, 1]}
            coords2 = {'lat': coords[1, 0], 'lng': coords[1, 1]}

            directions = self.maps.directions(coords1, coords2, mode='walking')

            self.directions.append(directions)

    def print_directions(self):
        """
        Print out directions leg by leg for each site
        """
        for i in self.directions:
            count = 0
            for j in i[0]['legs'][0]['steps']:
                count += 1
                print('#' * 50)
                print(str(count) + ' Leg of Trip : ')
                print('#' * 50)
                print(j)
                print(j['html_instructions'])
                print(j['distance']['text'])
                print('#' * 50)
                raw_input()

            print('#' * 30)
            print('You Have Arrived at your destination')
            raw_input()


def wiki_travel(major_site, gkey, radius=800, results=10):
    """
    Front end for running Wiki Travel
    """
    start = major_site

    t = Travel(major_site, radius=radius, results=results)
    t.locate(gkey)
    t.local_search()
    t.user_select()

    r = Trip(t, start, start)
    r.build_graph()
    r.find_routes()
    r.get_directions()
    r.print_directions()
