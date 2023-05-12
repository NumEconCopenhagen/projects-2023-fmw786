from types import SimpleNamespace
import numpy as np
from scipy import optimize

class SolowModelClass():

    def __init__(self,do_print=True):
        """ create the model """

        if do_print: print('initializing the model:')

        self.par = SimpleNamespace()
        self.ss = SimpleNamespace()
        self.path = SimpleNamespace()

        if do_print: print('calling .setup()')
        self.setup()

        if do_print: print('calling .allocate()')
        self.allocate()
    
    def setup(self):
        """ baseline parameters """

        par = self.par

        # a. household
        par.sigma = 2.0 # CRRA coefficient
        par.beta = np.nan # discount factor

        # b. firms
        par.A = np.nan
        par.production_function = 'cobb-douglas'
        par.alpha = 0.30 # capital weight
        par.theta = 0.05 # substitution parameter        
        par.delta = 0.05 # depreciation rate

        # c. initial
        par.K_lag_ini = 1.0

        # d. misc
        par.solver = 'broyden' # solver for the equation syste, 'broyden' or 'scipy'
        par.Tpath = 500 # length of transition path, "truncation horizon"

    def allocate(self):
        """ allocate arrays for transition path """
        
        par = self.par
        path = self.path

        allvarnames = ['A','K','C','rk','w','r','Y','K_lag']
        for varname in allvarnames:
            path.__dict__[varname] =  np.nan*np.ones(par.Tpath)