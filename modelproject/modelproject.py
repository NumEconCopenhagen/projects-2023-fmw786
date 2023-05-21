from types import SimpleNamespace
import numpy as np
from scipy import optimize
import pandas as pd 
import matplotlib.pyplot as plt

class HumanCapitalSolowModelClass:  
    
    def __init__(self,do_print=True):
        """ create the model """

        if do_print: print('initializing the model:')

        self.par = SimpleNamespace()
        self.path = SimpleNamespace()

        if do_print: print('calling .setup()')
        self.setup()

        if do_print: print('calling .allocate()')
        self.allocate()
     
    def setup(self):
        """ baseline parameters """

        par = self.par

        # Parameters
        par.alpha = 1/3
        par.phi = 1/3
        par.s_K = 0.2
        par.s_H = 0.15
        par.n = 0.0
        par.g = 0.015
        par.delta = 0.06

        # c. initial
        par.K_lag_ini = 1.0
        par.H_lag_ini = 1.0

        # d. misc
        par.solver = 'broyden' # solver for the equation system, 'broyden'.
        par.Tpath = 500 # length of transition path.

    def allocate(self):
        """ allocate arrays for transition path """
        
        par = self.par
        path = self.path

        allvarnames = ['Y', 'K', 'H', 'K_lag', 'H_lag']
        for varname in allvarnames:
            path.__dict__[varname] =  np.nan*np.ones(par.Tpath)

    def trans_eqs(self):
        """Input is the values of the parameters as described in the analytical section. 
        Output is the two transition equations for k and h respectivly """

        par = self.par
        path = self.path

        # Transition equation for k:
        trans_eq_k = (1 / ((1 + par.n) * (1 + par.g))) * (par.s_K * par.k**par.alpha * par.h**par.phi + (1 - par.delta) * par.k)-par.k

        # Transition equation for h:
        trans_eq_h = (1 / ((1 + par.n) * (1 + par.g))) * (par.s_H * par.k**par.alpha * par.h**par.phi + (1 - par.delta) * par.h)-par.h
    
        return trans_eq_k, trans_eq_h

    def find_steady_state(self, do_print=True):
        """Find steady state"""

        par = self.par
        path = self.path

        # Solving the model using optimize.root:
        objective = lambda x: [HumanCapitalSolowModelClass.trans_eqs(x[0],x[1], self)]
        solution = optimize.root(objective,[1,1], method='broyden1')
        num_solution = solution.x

        # Calculating steady state value of y
        ss_y = num_solution[0]**par.alpha * num_solution[1]**par.phi

        if do_print:
            print(f"The steady state value of k is: {num_solution[0]}")
            print(f"The steady state value of h is: {num_solution[1]}")
            print(f"The steady state value of y is: {ss_y}")
