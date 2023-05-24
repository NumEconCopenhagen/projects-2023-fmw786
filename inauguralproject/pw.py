from types import SimpleNamespace

import numpy as np
from scipy import optimize

import pandas as pd 
import matplotlib.pyplot as plt

class HouseholdSpecializationModelClass:

    def __init__(self):
        """ setup model """

        # a. create namespaces
        par = self.par = SimpleNamespace()
        sol = self.sol = SimpleNamespace()

        # b. preferences
        par.rho = 2.0
        par.nu = 0.001
        par.epsilon = 1.0
        par.omega = 0.5 

        # ekstra parameter
        par.phi = 0

        # c. household production
        par.alpha = 0.5
        par.sigma = 1.0

        # d. wages
        par.wM = 1.0
        par.wF = 1.0
        par.wF_vec = np.linspace(0.8,1.2,5)

        # e. targets
        par.beta0_target = 0.4
        par.beta1_target = -0.1

        # f. solution
        sol.LM_vec = np.zeros(par.wF_vec.size)
        sol.HM_vec = np.zeros(par.wF_vec.size)
        sol.LF_vec = np.zeros(par.wF_vec.size)
        sol.HF_vec = np.zeros(par.wF_vec.size)

        sol.beta0 = np.nan
        sol.beta1 = np.nan

    def calc_utility(self,LM,HM,LF,HF):
        """ calculate utility """

        par = self.par
        sol = self.sol

        # a. consumption of market goods
        C = par.wM*LM + par.wF*LF

        # b. home production
        H = np.nan

        power = (par.sigma - 1)/par.sigma

        if par.sigma == 1:
            H = HM**(1-par.alpha)*HF**par.alpha
        elif par.sigma == 0:
            H = np.fmin(HM, HF)
        else: 
            HM = np.fmax(HM, 1e-07)
            HF = np.fmax(HF, 1e-07)
            inner = (1-par.alpha)*HM**((par.sigma-1)/par.sigma) +par.alpha*HF**((par.sigma-1)/par.sigma)
            H = np.fmax(inner, 1e-07) **(par.sigma/(par.sigma-1))

        # c. total consumption utility
        Q = C**par.omega*H**(1-par.omega)
        utility = np.fmax(Q,1e-8)**(1-par.rho)/(1-par.rho)

        # d. disutlity of work
        eps = 1+1/par.epsilon
        TM = LM+HM
        TF = LF+HF
        disutility = par.nu*(TM**eps/eps+TF**eps/eps + par.phi*HM) # Where par.phi*HM is the extra parameter for the extra disutility af men working at home.
        
        return utility - disutility

    def solve_discrete(self,do_print=False):
        """ solve model discretely """
        
        par = self.par
        sol = self.sol
        opt = SimpleNamespace()
        
        #a. define what can be choosen from:
        x = np.linspace(0,24,49)
        LM,HM,LF,HF = np.meshgrid(x,x,x,x) 
    
        LM = LM.ravel()
        HM = HM.ravel()
        LF = LF.ravel()
        HF = HF.ravel()

        #b. calculate the utility depending on the choises of LM, HM, LF and HF:
        u = self.calc_utility(LM,HM,LF,HF)
    
        #c. add "punishment if one (or two) of the constraints does not hold"
        I = (LM+HM > 24) | (LF+HF > 24)
        u[I] = -np.inf
    
        # d. find maximizing argument
        j = np.argmax(u)
        
        opt.LM = LM[j]
        opt.HM = HM[j]
        opt.LF = LF[j]
        opt.HF = HF[j]

        # e. print
        if do_print:
            for k,v in opt.__dict__.items():
                print(f'{k} = {v:6.4f}')

        return opt



    def solve(self,do_print=False):
        """ solve model continously """

        par = self.par
        sol = self.sol
        opt = SimpleNamespace()
    
    # a. objective function 
        def obj(x):
            LM, HM, LF, HF = x
            return - self.calc_utility(LM, HM, LF, HF)
    
    #b. Constraints and Bounds (to minimize) 
        def constraints(x):
            LM, HM, LF, HF = x
            return [24 - LM-HM, 24 -LF-HF]
    

        constraints = ({'type': 'ineq', 'fun':constraints}) 
        bounds = ((0,24), (0,24), (0,24), (0,24))

        initial_guess = [6,6,6,6]

    #c. Solver 
        solution = optimize.minimize(obj, initial_guess, method="SLSQP", bounds=bounds, constraints=constraints, tol = 0.000000001)

        opt.LM = solution.x[0]
        opt.HM = solution.x[1]
        opt.LF = solution.x[2]
        opt.HF = solution.x[3]
        
        return opt

    def solve_wF_vec(self,discrete=False):
        """ solve model for vector of female wages """
        
        par = self.par
        sol = self.sol
        
        results = np.zeros(par.wF_vec.size)

        for i, wF in enumerate(par.wF_vec):
            par.wF = wF
            
            opt = self.solve()
            
            sol.HM = opt.HM
            sol.HF = opt.HF
            results[i] = sol.HF/sol.HM

        sol.results = results

        return results

    def solve_wF_vec_2(self,discrete=False):
        """ solve model for vector of female wages """
        
        par = self.par
        sol = self.sol

        for i, wF in enumerate(par.wF_vec):
            par.wF = wF
            
            opt = self.solve()

            sol.LM_vec[i] = opt.LM
            sol.HM_vec[i] = opt.HM
            sol.LF_vec[i] = opt.LF
            sol.HF_vec[i] = opt.HF

        return sol 


    def run_regression(self):
        """ run regression """
        
        #Setting up parameters
        par = self.par
        sol = self.sol
        
        #Running regression
        x = np.log(par.wF_vec)
        y = np.log(sol.HF_vec/sol.HM_vec)
        A = np.vstack([np.ones(x.size),x]).T
        sol.beta0,sol.beta1 = np.linalg.lstsq(A,y,rcond=None)[0]
        return sol 
    
    def estimate(self, alpha=None, sigma=None):

        par = self.par
        sol = self.sol

        def obj(q):
            par.alpha, par.sigma = q

            self.solve_wF_vec_2()
            self.run_regression()

            err =  (par.beta0_target - sol.beta0)**2 + (par.beta1_target-sol.beta1)**2
            return err


        bounds = [(0, 0.99),(0.01, 2)]
        initial_guess = (0.5, 1)

        reg_opt = optimize.minimize(obj, initial_guess, method='Nelder-Mead', bounds=bounds, tol = 0.000000001)


        alpha_hat = reg_opt.x[0]
        sigma_hat = reg_opt.x[1]

        err = obj(reg_opt.x)



        print (f'Minimizing the squared errrors gives the regression:')
        print(f"    Beta0_hat =  {sol.beta0:.2f}")
        print(f"    Beta1_hat =  {sol.beta1:.2f}")

        print(f'This gives the parameters: \n    alpha = {alpha_hat:.2f} \n    sigma = {sigma_hat:.2f}')
        print(f' With the squared error {err:.2f}')


    def estimate_5(self, alpha=None, sigma=None):

        par = self.par
        sol = self.sol

        def obj(q):
            par.sigma, par.phi = q

            self.solve_wF_vec_2()
            self.run_regression()

            err =  (par.beta0_target - sol.beta0)**2 + (par.beta1_target-sol.beta1)**2
            return err


        bounds = [(0.01, 2), (0, 5)]
        initial_guess = (0.5, 2.5)

        reg_opt = optimize.minimize(obj, initial_guess, method='Nelder-Mead', bounds=bounds, tol = 0.000000001)


        sigma_hat = reg_opt.x[0]
        phi_hat = reg_opt.x[1]

        err = obj(reg_opt.x)



        print (f'Minimizing the squared errrors gives the regression:')
        print(f"    Beta0_hat =  {sol.beta0:.2f}")
        print(f"    Beta1_hat =  {sol.beta1:.2f}")

        print(f'This gives the parameters: \n    Fixed: alpha = {par.alpha:.2f} \n    sigma = {sigma_hat:.2f} \n    phi = {phi_hat:.2f}')
        print(f' With the squared error {err:.2f}')