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
    :param int results: max number of wiki results to query for sight
    :param int radius: search radius (meters)
    :param str pop_date: month to grab wiki page views from (YYYYMM) \
            note month pages must exist at http://stats.grok.se/
    :param dict factors: additional criteria INDEV
    :param dict weights: additional criteria INDEV
    """

    def __init__(self, location, gkey, results=15, radius=1000,
                 pop_date='201506', factors=None, weights=None):

        self.location = location
        self.gkey = gkey
        self.results = results
        self.radius = radius
        self.pop_date = pop_date
        self.factors = factors
        self.weights = weights

        # Populated by methods
        self.coordinates = []
        self.df = pd.DataFrame()
        self.wiki_resp = None

    def locate(self):
        """
        Find major sight location: self.coordinates = [latitude, longitude]
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

            [latitude, longitude] = loc_resp[0]['geometry']\
                                               ['location'].values()

        self.coordinates = [latitude, longitude]

    def local_search(self):
        """
        Find nearby sites to self.location
        """

        latitude = self.coordinates[0]
        longitude = self.coordinates[1]

        # Search Wikipedia for nearby pages
        self.wiki_resp = wk.geosearch(latitude, longitude,
                                      results=self.results,
                                      radius=self.radius)

        names = []
        ranks = []
        latitudes = []
        longitudes = []

        # Set source for popularity information
        pop_url = 'http://stats.grok.se/json/en/' + self.pop_date + '/'

        for resp in self.wiki_resp:

            # Get Wikipedia page
            wiki_page = wk.WikipediaPage(resp)
            wiki_url_tag = wiki_page.url.split('/')[-1]

            latitudes.append(wiki_page.coordinates[0])
            longitudes.append(wiki_page.coordinates[1])

            # Get popularity
            url_resp = urllib2.urlopen(pop_url + wiki_url_tag)
            json_resp = json.load(url_resp)
            num_views = sum(json_resp['daily_views'].itervalues())

            ranks.append(num_views)
            names.append(resp)

        # Populate Pandas Data Frame
        latitudes = np.asarray(latitudes).astype(float)
        longitudes = np.asarray(longitudes).astype(float)

        self.df = pd.DataFrame(data={'place': names,
                                     'latitude': latitudes,
                                     'longitude': longitudes,
                                     'views': ranks
                                     })

        self.df = self.df.sort(['views'], ascending=False)
        self.df.index = np.arange(len(self.df)) + 1

    def user_select(self):
        """
        User selects additional sight to include for trip

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
        self.sights = travel.df['place'].values
        if end is None:
            self.end = start
            self.waypoints = len(self.sights) + 1
        else:
            self.end = end
            self.waypoints = len(self.sights)

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
        if len(self.sights) > 2:
            self.graph.add_nodes_from(self.sights)
            self.edges = itertools.combinations(self.sights, 2)
            self.calculate_distances()
            self.graph.add_weighted_edges_from(self.distances)

        else:
            raise Exception('Not a trip, fewer then 3 sights: ' + self.sights)

    def calculate_distances(self):
        """
        Calculate distances between sights to weight graph 
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
        :rtype float:
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
        Print out directions leg by leg for each sight
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


def wiki_travel(major_sight, gkey, radius=800, results=10):
    """
    Front end for running Wiki Travel
    """
    start = major_sight

    t = Travel(major_sight, radius=radius, results=results)
    t.locate(gkey)
    t.local_search()
    t.user_select()

    r = Trip(t, start, start)
    r.build_graph()
    r.find_routes()
    r.get_directions()
    r.print_directions()