from scipy import spatial as spl
import scipy as sp
import numpy as np
from matplotlib import pyplot as plt
import subject


class kd_sir(spl.kdtree.KDTree):
    """
    Class for modeling SIR infectious disease

    :param int members: initial population size
    :param tuple size: (1x2) cell dimensions
    :param int infect_points: number of initially infected members
    :param str dist_type: distribution of members in cell
    :param float infect_radius: distance constraint a member can infect\
            another member
    :param float infect_pct: percent chance an interaction leads to infection
    :param float susceptible_pct: percent chance an individual is not immune
    :param int duration: duration of infection
    :param int time: initial day of model
    :param sirlog record: object to handle recording simulation data

    """
    def __init__(self, members=1000, size=(1, 1), infect_points=1,
                 dist_type='uniform', infect_radius=.001, infect_pct=.01,
                 susceptible_pct=.85, duration=2, time=0, record=None):

        if dist_type == 'uniform':
            xs = size[0] * sp.rand(members, 1)
            ys = size[1] * sp.rand(members, 1)

        coordinates = np.c_[xs, ys]

        # BCs
        spl.kdtree.KDTree.__init__(self, coordinates)
        self.infect_points = infect_points
        self.infect_radius = infect_radius
        self.infect_pct = infect_pct
        self.susceptible_pct = susceptible_pct
        self.duration = duration
        self.subjects = {}
        self.time = time

        # Records
        self.infected = None
        self.innoculated = None
        self.recovered = None
        self.infected_pct = infect_points / float(members)
        self.innoculated_pct = None
        self.recovered_pct = None

        # Geometry
        self.area = size[0] * size[1]
        self.rho = members / self.area
        self.radius = infect_radius / self.area
        self.infect_damage = self.radius * infect_pct

        # PInumbers/metrics
        self.infect_damage_rho = self.infect_damage * self.rho

        self.record = record

    def init_infect(self):
        """
        Initialize members with attributes
        """
        self.infected = np.zeros((self.data.shape[0]), dtype=bool)
        self.innoculated = np.zeros((self.data.shape[0]), dtype=bool)
        self.infected[0: self.infect_points] = True

        for i, j in zip(self.data, self.infected):
            susceptible_seed = bool(np.random.rand(1) < self.susceptible_pct)
            self.subjects[tuple(i)] = subject.Subject(i, susceptible_seed,
                                                      j, self.duration)

    def step_time(self):
        """
        Step forward 1 cycle
        """

        infected_index = []
        innoculated_index = []

        for i, j in zip(self.data, np.arange(self.data.shape[0])):
            sid = tuple(i)

            if self.subjects[sid].susceptible and not self.subjects[sid].infectious:
                neighbors_index = self.query_ball_point(i, self.infect_radius)

                if neighbors_index:
                    for n in self.infected[neighbors_index]:
                        if self.infect_pct > np.random.rand(1):
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

        if self.record:
            if self.time % self.record.frequency == 0:
                self.record.record_step()

        self.time += 1

    def plot_step(self):
        """
        Plot current state
        """
        x = self.data[:, 0]
        y = self.data[:, 1]
        plt.plot(x, y, 'k.', label='population')
        plt.plot(x[self.infected], y[self.infected], 'r*', markersize=15,
                 label='infected')
        plt.plot(x[self.innoculated], y[self.innoculated], 'g*', markersize=15,
                 label='innoculated')
        if self.record:
            plt.savefig(self.record.name + str(self.time))
        else:
            plt.savefig(str(self.time))
        plt.close()
        print(sum(self.infected))
        print(sum(self.innoculated))
