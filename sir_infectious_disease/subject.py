class Subject(object):
    '''
    '''

    def __init__(self, location=(0., 0.), susceptible = False, infectious=False, duration=2):
        '''
        '''

        self.location = location
        self.susceptible = susceptible
        self.infectious = infectious
        self.infected_time = 0.
        self.duration = duration

    def infect(self):
        self.infectious = True

    def innoculate(self):
        self.infectious = False
        self.suscetible = False
