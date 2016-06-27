from scipy import spatial as spl
import scipy as sp
import numpy as np
import subject
#import h5infect
from matplotlib import pyplot as plt

#Metrics infection rate: density vs time, clusters(size + shape), critical saturation
#quantifiable impact of vaccination


class kd_infect(spl.kdtree.KDTree):
    '''
    '''
    def __init__(self, members=1000, size=(1, 1), infect_points=1, dist_type='uniform',
             infect_radius=.001, infect_pct=.01, susceptible_pct=.85, duration=2, time=0,
             record_name=None, local_path=None, group_name='default'):
        '''
        '''
        if dist_type == 'uniform':
            xs = size[0] * sp.rand(members, 1)
            ys = size[1] * sp.rand(members, 1)

        coordinates = np.c_[xs, ys]

        #BCs
        spl.kdtree.KDTree.__init__(self, coordinates)
        self.infect_points = infect_points
        self.infect_radius = infect_radius
        self.infect_pct = infect_pct
        self.susceptible_pct = susceptible_pct
        self.duration = duration
        self.subjects = {}
        self.time = time

        #Records
        self.infected = None
        self.innoculated = None
        self.recovered = None
        self.infected_pct = infect_points / float(members)
        self.innoculated_pct = None
        self.recovered_pct = None
        
        #RecordKeeping
        self.local_path = local_path
        if group_name:
            self.group_name = group_name
        #Geometry
        self.area = size[0] * size[1]
        self.rho  = members / self.area
        self.radius = infect_radius / self.area
        self.infect_damage = self.radius * infect_pct

        #PInumbers/metrics
        self.infect_damage_rho = self.infect_damage * self.rho

        if record_name:
            self.record = h5infect.InfectFile(record_name)

    def init_infect(self):
        '''
        '''
        self.infected = np.zeros((self.data.shape[0]), dtype=bool)
        self.innoculated = np.zeros((self.data.shape[0]), dtype=bool)
        self.infected[0: self.infect_points] = True

        for i, j in zip(self.data, self.infected):
            susceptible_seed = bool(rnum() < self.susceptible_pct)
            self.subjects[tuple(i)] = subject.Subject(i, susceptible_seed,
                                               j, self.duration)

    def step_time(self):
        '''
        '''

        infected_index = []
        innoculated_index = []

        for i, j in zip(self.data, np.arange(self.data.shape[0])):
            sid = tuple(i)

            if self.subjects[sid].susceptible and not self.subjects[sid].infectious:
                neighbors_index = self.query_ball_point(i, self.infect_radius)

                if neighbors_index:
                    for n in self.infected[neighbors_index]:
                        if self.infect_pct > rnum():
                            infected_index.append(j)
                            self.subjects[sid].infect()
                            break

            elif self.subjects[sid].infectious:
                self.subjects[sid].infected_time += 1

                if self.subjects[sid].infected_time > self.duration:
                    self.subjects[sid].innoculate()
                    innoculated_index.append(j)

        if infected_index:
            self.infected[np.array(infected_index)] = True
        if innoculated_index:
            self.infected[np.array(innoculated_index)] = False
            self.innoculated[np.array(innoculated_index)] = True

        if self.record_name:
            self.record_step(self.record_name, self.local_path, self.group_name)

    def record_step(self, file_path, local_path, group_name):
        '''
        '''

        self.format_record()
        self.record.create_group(local_path)
        self.record[local_path].create_dataset('data', data=self.data)
        self.record[local_path].create_dataset('infected', data=self.infected)
        self.record[local_path].create_dataset('innoculated', data=self.innoculated)


    def format_record(self):
        '''
        '''
        self.create_empty_record()


    def create_empty_write_record(self):
        """
        """
        self.write_record = np.array(
                                     tuple([' '] * len(self.fields)),
                                         dtype=self.formats
                                    )

    def plot_step(self, file_path):
        '''
        '''
        x = self.data[:, 0]
        y = self.data[:, 1]
        plt.plot(x, y, 'k.', label='population')
        plt.plot(x[self.infected], y[self.infected], 'r*', markersize=15, label='infected')
        plt.plot(x[self.innoculated], y[self.innoculated], 'g*', markersize=15, label='innoculated')
        plt.savefig(file_path)
        plt.close()
        print(sum(self.infected))
        print(sum(self.innoculated))




def rnum():
    '''
    '''
    return np.random.rand(1)

if __name__ == 'main':
    tree = tree = owner.kd_infect()
    tree.init_infect()

