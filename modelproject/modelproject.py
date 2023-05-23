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
        par.n = 0.00
        par.g = 0.015
        par.delta = 0.06

        # c. initial
        par.K_ini = 10
        par.H_ini = 10

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

    def trans_eqs(self, K, H):
        """Input is the values of K and H, and the values of the parameters as described in the analytical section. 
        Output is the two transition equations for K and H, respectively."""

        par = self.par
        path = self.path

        # Transition equation for K:
        trans_eq_K = (1 / ((1 + par.n) * (1 + par.g))) * (par.s_K * K**par.alpha * H**par.phi + (1 - par.delta) * K) - K

        # Transition equation for H:
        trans_eq_H = (1 / ((1 + par.n) * (1 + par.g))) * (par.s_H * K**par.alpha * H**par.phi + (1 - par.delta) * H) - H

        return trans_eq_K, trans_eq_H


    def find_steady_state(self, do_print=True):
        """Find steady state"""

        par = self.par
        path = self.path

        # Solving the model using optimize.root:
        objective = lambda x: self.trans_eqs(x[0], x[1])
        initial_guess = [par.K_ini, par.H_ini]  # Use initial values from the setup method
        solution = optimize.root(objective, initial_guess, method='broyden1')
        num_solution = solution.x

        # Extracting the values of trans_eq_K and trans_eq_H
        trans_eq_K, trans_eq_H = self.trans_eqs(num_solution[0], num_solution[1])

        # Calculating steady state value of y
        ss_y = num_solution[0]**par.alpha * num_solution[1]**par.phi

        if do_print:
            print(f"The steady state value of \u0303k* is: {num_solution[0]:.5f}")
            print(f"The steady state value of \u0303h* is: {num_solution[1]:.5f}")
            print(f"The steady state value of \u0303y* is: {ss_y:.5f}")

        return num_solution[0], num_solution[1], ss_y
    
    def simulate_transition_path(self):
        """Simulate transition path for k, h and y"""

        par = self.par
        path = self.path

        # Time
        T = np.arange(0, par.Tpath)

        # Variables for transition path
        Kt = np.zeros(par.Tpath)
        Ht = np.zeros(par.Tpath)
        Yt = np.zeros(par.Tpath)
        Lt = np.zeros(par.Tpath)
        At = np.zeros(par.Tpath)

        # Production function
        def production(K, H, L, A):
            return K**par.alpha * H**par.phi * (A*L)**(1 - par.alpha - par.phi)

        # Initial values
        Kt[0] = 1
        Ht[0] = 1
        Lt[0] = 1
        At[0] = 1
        Yt[0] = production(Kt[0], Ht[0], Lt[0], At[0])

        # Model simulation
        for t in range(1, par.Tpath):
            Kt[t] = par.s_K * Yt[t-1] + (1 - par.delta) * Kt[t-1]
            Ht[t] = par.s_H * Yt[t-1] + (1 - par.delta) * Ht[t-1]
            Lt[t] = Lt[t-1] * (1 + par.n)
            At[t] = At[t-1] * (1 + par.g)
            Yt[t] = production(Kt[t], Ht[t], Lt[t], At[t])

        # Define tilde variables:
        kt_tilde = Kt / (At * Lt)
        ht_tilde = Ht / (At * Lt)
        yt_tilde = Yt / (At * Lt)

        return kt_tilde, ht_tilde, yt_tilde
    






    









        
