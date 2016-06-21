import numpy as np
import pandas as pd
import json
import urllib2
import wikipedia as wk
import googlemaps as gm
import networkx
import itertools
import copy


# TODOs
# Build in recrusion to build graphs off initital graph and link

class Travel(object):
    """
    Class for finding nearby sites of interest via google maps and wikipedia
    Site popularity evaluated by page views in the past (or specified month)

    """

    def __init__(self, location, gkey, results=4, radius=500,
                 pop_date='201605', factors=None, weights=None):
        """
        Create Travel object based on location, factors + weights (indev)

        :param str location: string specifying location, default is geocode
        :param str gkey: Google Maps api key
        :param int results: max number of wiki results to query for sight
        :param int radius: search radius (meters)
        :param str pop_date: month to grab wiki page views from (YYYYMM)
        :param dict factors: additional criteria INDEV
        :param dict weights: additional criteria INDEV
        """

        self.location = location
        self.gkey = gkey
        self.results = results
        self.radius = radius
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

            [latitude, longitude] = loc_resp[0]['geometry'] \
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

        # !!!TODO cleanup into object
        # Initialize rank and position as lists
        names = []
        ranks = []
        latitudes = []
        longitudes = []


        #Set source for popularity information
        pop_url = 'http://stats.grok.se/json/en/' + self.pop_date +'/'

        for resp in self.wiki_resp:

            #Get Wikipedia page
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

    """

    def __init__(self, travel, start, end=None):
        """
        Constructor for Trip class, requries Travel & Google Maps api key

        :param Travel travel: Travel object (with df attribute)
        :param str start: Starting point of trip
        :param str end: Ending point of trip. Defaults to start

        """

        self.travel = travel
        self.start = start
        if end is None:
            self.end = start
        else:
            self.end = end
        self.sights = travel.df['place'].values

        # Assigned in methods
        self.graph = networkx.empty_graph(len(self.sights))
        self.route = {}
        self.directions = {}

    def build_graph(self):
        """
        Link all locations together in graph (completely connected)
        """

        if len(self.sights) > 1:
            edges = itertools.combinations(self.sights, 2)
            self.graph.add_edges_from(edges)

        else:
            raise Exception('Not a trip, only one sight: ' + self.sights)

    def populate_graph(self):
        """
        Populate graph with distances between nodes
        """
        maps = gm.Client(self.travel.gkey)

        for i in self.sights:
            for k in self.graph[i]:

                if self.graph[k][i]:
                    self.graph[i][k] = self.graph[k][i]
                else:
                    l = np.in1d(self.travel.df['place'].values,
                                np.array([i, k]))
                    coords = self.travel.df[['latitude', 'longitude']].values
                    coords = coords[l, :]
                    coords1 = {'lat': coords[0, 0], 'lng': coords[0, 1]}
                    coords2 = {'lat': coords[1, 0], 'lng': coords[1, 1]}

                    dist = maps.distance_matrix(coords1, coords2,
                                                mode='walking')\
                            ['rows'][0]['elements'][0]['distance']['value']
                    self.graph[i][k] = dist

    def find_routes(self, route_method='AStar'):
        """
        Find route based on optimization method

        :param str route_method: Currently supports 'brute'(computes all permutations) \
                and AStar (A* from networkx)
        """

        #Need parameter for fixed/free Initial Condition
        nodes = copy.copy(self.sights.tolist())
        nodes.remove(self.start)

        if self.route_method == 'brute':
            self.brute_route()

        elif self.route_method == 'AStar':
            self.astar_route()

        print('Your optimal route is:')
        print(self.route)

    def brute_route(self):
        """
        Use brute force to optimize route (compute all permutations)
        """

        perm = list(itertools.permutations(nodes))

        for path_perm in perm:

            path = list(path_perm)

            path.insert(0, self.start)
            path.append(self.end)

            cost = self.route_cost(path)

            if 'dist' in self.route.keys():
                if self.route['distance (m)'] > cost:
                    self.route['distance (m)'] = cost
                    self.route['route'] = path 

            else:
                self.route['distance (m)'] = cost
                self.route['route'] = path

    def route_cost(self, route):
        """
        """

        cost = 0

        for i, j in zip(route[:-1], route[1:]):
            cost += self.graph[i][j]

        return cost

    def astar_route(self):
        """
        Use A* to optimize route (newtorkx)
        """

    def get_directions(self):
        """
        """

        path = self.route['route']
        count = 0
        maps = gm.Client(self.travel.gkey)

        for i, k in zip(path[:-1], path[1:]):
            l = np.in1d(self.travel.df['place'].values, np.array([i, k]))
            coords = self.travel.df[['latitude', 'longitude']].values
            coords = coords[l, :]
            coords1 = {'lat': coords[0, 0], 'lng': coords[0, 1]}
            coords2 = {'lat': coords[1, 0], 'lng': coords[1, 1]}

            directions = maps.directions(coords1, coords2, mode='walking')

            self.directions[count] = directions
            count += 1

    def print_directions(self):
        """
        """
        for i in self.directions.itervalues():
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


def hidden_gems(major_sight, gkey, radius=800, results=10):
    """
    """
    start = major_sight

    t = Travel(major_sight, radius=radius, results=results)
    t.locate(gkey)
    t.local_search()
    t.user_select()

    r = Trip(t, gkey, start, start)
    r.build_graph()
    r.populate_graph(gkey)
    r.find_routes()
    r.get_directions(gkey)
    r.print_directions()
