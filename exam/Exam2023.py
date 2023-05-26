from types import SimpleNamespace
import numpy as np
from scipy import optimize
import pandas as pd 
import matplotlib.pyplot as plt
import sympy as sm
from scipy.optimize import fsolve
from scipy.optimize import minimize
from IPython.display import Image

class Problem1:  
    
    def lagrangian_optimization(do_print=True):
        """Solve the Lagrangian optimization problem"""

        # Define variables and parameters
        L = sm.symbols('L')
        G = sm.symbols('G')
        C = sm.symbols('C')
        kappa = sm.symbols('kappa')
        alpha = sm.symbols('alpha')
        nu = sm.symbols('nu')
        w = sm.symbols('w')
        w_tilde = sm.symbols(r'\tilde{w}', latex=True)
        tau = sm.symbols('tau')
        lambda_ = sm.symbols('lambda')

        # Define consumption
        C = kappa + w_tilde*L

        # Set up the Lagrangian parts:
        utility = sm.log(C**alpha * G**(1-alpha)) - nu * L**2 / 2
        constraint_eq = kappa + w_tilde*L - C

        # Set up the Lagrangian:
        Lagrangian = utility - lambda_ * constraint_eq

        # Take the derivative wrt. L
        Lagrangian_L = sm.diff(Lagrangian, L)

        # Take the derivative wrt. lambda
        Lagrangian_lambda = sm.diff(Lagrangian, lambda_)

        # Set the derivatives equal to zero
        Lagrangian_L = sm.Eq(Lagrangian_L, 0)
        Lagrangian_lambda = sm.Eq(Lagrangian_lambda, 0)

        # Solve the system of equations
        solution = sm.solve([Lagrangian_L, Lagrangian_lambda], [L, lambda_])

        if do_print:
            # Print the solution
            print(f'The optimal labor supply is: L = {sm.latex(solution[1][0])}')

        # Lambdify the solution
        L_star = sm.lambdify((kappa, alpha, nu, w_tilde), solution[1][0])
        
        return L_star
    
    def numerical_lagrangian_optimization(tau_=0.3):
        """Solve the Lagrangian and numerical optimization problem"""

        # Define variables and parameters
        L = sm.symbols('L')
        kappa = sm.symbols('kappa')
        alpha = sm.symbols('alpha')
        nu = sm.symbols('nu')
        w = sm.symbols('w')
        tau = sm.symbols('tau')
        lambda_ = sm.symbols('lambda')

        # Define equations
        w_tilde = (1 - tau) * w
        C = kappa + w_tilde * L
        G = tau * w * L

        # Set up the Lagrangian parts
        utility = sm.log(C ** alpha * G ** (1 - alpha)) - nu * L ** 2 / 2
        constraint_eq = kappa + w_tilde * L - C

        # Set up the Lagrangian
        Lagrangian = utility - lambda_ * constraint_eq

        # Calculate the derivatives
        Lagrangian_L = sm.diff(Lagrangian, L)
        Lagrangian_lambda = sm.diff(Lagrangian, lambda_)

        # Convert the equations to lambdify functions
        Lagrangian_L_func = sm.lambdify((L, lambda_, kappa, alpha, nu, w, tau), Lagrangian_L)
        Lagrangian_lambda_func = sm.lambdify((L, lambda_, kappa, alpha, nu, w, tau), Lagrangian_lambda)

        # Define a function that represents the system of equations
        def equations(variables, *args):
            L, lambda_ = variables
            kappa, alpha, nu, w, tau = args
            return [
                Lagrangian_L_func(L, lambda_, kappa, alpha, nu, w, tau),
                Lagrangian_lambda_func(L, lambda_, kappa, alpha, nu, w, tau)
            ]

        # Set the values for the parameters
        kappa_val = 1
        alpha_val = 0.5
        nu_val = 1 / (2 * 16 ** 2)
        w_val = 1
    
        # Set the initial guess for L and lambda
        initial_guess = [1, 1]

        # Solve the system of equations numerically
        L_star_G = fsolve(equations, initial_guess, args=(kappa_val, alpha_val, nu_val, w_val, tau_))

        # Calculate government consumption
        G_star = tau_ * w_val * L_star_G[0]

        # Worker utility
        utility = np.log((kappa_val + (1 - tau_) * w_val * L_star_G[0]) ** alpha_val * (tau_ * w_val * L_star_G[0]) ** (1 - alpha_val)) - nu_val * L_star_G[0] ** 2 / 2

        # Return the optimal labor supply
        return L_star_G[0], G_star, utility
        """Solve the Lagrangian and numerical optimization problem with the more general preference formulation"""

        # Define variables and parameters
        L = sm.symbols('L')
        G = sm.symbols('G')
        kappa = sm.symbols('kappa')
        alpha = sm.symbols('alpha')
        nu = sm.symbols('nu')
        w = sm.symbols('w')
        tau = sm.symbols('tau')
        lambda_ = sm.symbols('lambda')
        sigma_ = sm.symbols('sigma')
        rho_ = sm.symbols('rho')
        epsilon_ = sm.symbols('epsilon')

        # Define equations
        w_tilde = (1 - tau) * w
        C = kappa + w_tilde * L

        # Set up the Lagrangian parts
        utility = ((alpha * C ** ((sigma_ - 1) / sigma_)) + ((1 - alpha) * G ** ((sigma_ - 1) / sigma_))) ** (sigma_ / (sigma_ - 1))
        utility1 = ((utility ** (1 - rho_)) -1) / (1 - rho_)
        disutility = nu * L ** (1 + epsilon_) / (1 + epsilon_)
        Lagrangian = utility1 - disutility - lambda_ * (kappa + w_tilde * L - C)

        # Calculate the derivatives
        Lagrangian_L = sm.diff(Lagrangian, L)

        # Convert the equation to a lambdify function
        Lagrangian_L_func = sm.lambdify((L, G, lambda_, kappa, alpha, nu, w, tau, sigma_, rho_, epsilon_), Lagrangian_L)

        # Define a function that represents the equation
        def equation(L, *args):
            G, kappa, alpha, nu, w, tau, sigma, rho, epsilon = args
            return G - tau * w * L

        # Set the values for the parameters
        G_val = 1  # Initial guess for G

        # Set the values for the parameters
        kappa_val = 1
        alpha_val = 0.5
        nu_val = 1 / (2 * 16 ** 2)
        w_val = 1

        # Solve for L that satisfies the Lagrangian equation
        L_star = optimize.fsolve(Lagrangian_L_func, 1, args=(G_val, 0, kappa_val, alpha_val, nu_val, w_val, tau_, sigma, rho, epsilon), xtol=1e-8)[0]

        # Solve for G that satisfies the equation
        G_star = optimize.fsolve(equation, 1, args=(L_star, kappa_val, alpha_val, nu_val, w_val, tau_, sigma, rho, epsilon), xtol=1e-8)[0]

        # Calculate utility
        utility_star = ((alpha_val * (kappa_val + w_val * L_star) ** ((sigma - 1) / sigma)) + ((1 - alpha_val) * G_star ** ((sigma - 1) / sigma))) ** (sigma / (sigma - 1))
        utility_star1 = ((utility_star ** (1 - rho)) -1) / (1 - rho)
        disutility_star = nu_val * L_star ** (1 + epsilon) / (1 + epsilon)
        total_utility = utility_star1 - disutility_star
        

        # Return the optimal labor supply, G, and utility
        return L_star, G_star, total_utility

    def new_optimization(tau_=0.52257, sigma=1.001, rho=1.001, epsilon=1):
        """Solve the Lagrangian and numerical optimization problem with the more general preference formulation"""

        # Define variables and parameters
        L = sm.symbols('L')
        G = sm.symbols('G')
        kappa = sm.symbols('kappa')
        alpha = sm.symbols('alpha')
        nu = sm.symbols('nu')
        w = sm.symbols('w')
        tau = sm.symbols('tau')
        lambda_ = sm.symbols('lambda')
        sigma_ = sm.symbols('sigma')
        rho_ = sm.symbols('rho')
        epsilon_ = sm.symbols('epsilon')

        # Define equations
        w_tilde = (1 - tau) * w
        C = kappa + w_tilde * L

        # Set up the Lagrangian parts
        utility = ((alpha * C ** ((sigma_ - 1) / sigma_)) + ((1 - alpha) * G ** ((sigma_ - 1) / sigma_))) ** (sigma_ / (sigma_ - 1))
        utility1 = ((utility ** (1 - rho_)) - 1) / (1 - rho_)
        disutility = nu * L ** (1 + epsilon_) / (1 + epsilon_)
        Lagrangian = utility1 - disutility - lambda_ * (kappa + w_tilde * L - C)

        # Calculate the derivatives
        Lagrangian_L = sm.diff(Lagrangian, L)

        # Convert the equation to a lambdify function
        Lagrangian_L_func = sm.lambdify((L, G, lambda_, kappa, alpha, nu, w, tau, sigma_, rho_, epsilon_), Lagrangian_L)

        # Define the utility function
        def utility_function(L, G, kappa, alpha, nu, w, tau, sigma, rho, epsilon, nu_):
            C = kappa + (1 - tau) * w * L
            utility = ((alpha * C ** ((sigma - 1) / sigma)) + ((1 - alpha) * G ** ((sigma - 1) / sigma))) ** (sigma / (sigma - 1))
            utility1 = ((utility ** (1 - rho)) - 1) / (1 - rho)
            disutility = nu * L ** (1 + epsilon) / (1 + epsilon)
            return utility1 - disutility - nu_ * L ** (1 + epsilon) / (1 + epsilon)

        # Set the values for the parameters
        G_val = 1  # Initial guess for G
        kappa_val = 1
        alpha_val = 0.5
        nu_val = 1 / (2 * 16 ** 2)
        w_val = 1

        # Solve for L that maximizes the utility function
        result = optimize.minimize(lambda L: -utility_function(L, G_val, kappa_val, alpha_val, nu_val, w_val, tau_, sigma, rho, epsilon, nu_val), 1)
        L_star = result.x[0]

        # Solve for G that satisfies the equation
        G_star = optimize.fsolve(lambda G: G - tau_ * w_val * L_star, 1)[0]

        # Calculate utility using the defined utility function
        total_utility = utility_function(L_star, G_star, kappa_val, alpha_val, nu_val, w_val, tau_, sigma, rho, epsilon, nu_val)

        # Return the optimal labor supply, G, and utility
        return L_star, G_star, total_utility

class Problem2:  


    def Q1part1():
        eta = 0.5
        w = 1.0
        kappas = [1.0, 2.0]

        def calculate_profit(kappa, l):
            return kappa * l**(1 - eta) - w * l

        def find_optimal_l(kappa):
            l_values = np.linspace(0.1, 10.0, num=1000)
            profits = [calculate_profit(kappa, l) for l in l_values]
            max_profit = max(profits)
            max_profit_index = profits.index(max_profit)
            optimal_l = l_values[max_profit_index]
            return optimal_l

        for kappa in kappas:
            optimal_l = find_optimal_l(kappa)
            print(f"Optimal l for kappa calculated with optimizer for kappa={kappa}: {optimal_l:.4f}")

        for kappa in kappas:
            l = ((1 - eta) * kappa / w) ** (1 / eta)
            profit = calculate_profit(kappa, l)
            print(f"Optimal l calculated with analytic solution kappa={kappa}: {profit:.4f}")

    def Q1part2():
        
        eta = 0.5
        w = 1.0
        kappas = [1.0, 2.0]

        def calculate_profit(kappa, l):
            return kappa * l**(1 - eta) - w * l

        def find_optimal_l(kappa):
            l_values = np.linspace(0.1, 10.0, num=1000)
            profits = [calculate_profit(kappa, l) for l in l_values]
            max_profit = max(profits)
            max_profit_index = profits.index(max_profit)
            optimal_l = l_values[max_profit_index]
            return optimal_l

        for kappa in kappas:
            optimal_l = find_optimal_l(kappa)


        for kappa in kappas:
            l = ((1 - eta) * kappa / w) ** (1 / eta)
            profit = calculate_profit(kappa, l)

        l_values = np.arange(0.0, 2.0, 0.1)

        for kappa in kappas:
            profit_values = [calculate_profit(kappa, l) for l in l_values]
            plt.plot(l_values, profit_values, label=f'kappa={kappa}')


        plt.xlabel('l_t')
        plt.ylabel('Pi_t')
        plt.legend()
        plt.grid(True)
        plt.title('Profit as a Function of l_t for Different Kappa Values')
        plt.show()


    def Q2():

        eta = 0.5
        w = 1.0
        rho = 0.9
        iota = 0.01
        sigma_epsilon = 0.1
        R = (1 + 0.01) ** (1 / 12)

        def calculate_profit(kappa, ell, ell_prev):
            return kappa * ell**(1 - eta) - w * ell - (ell != ell_prev) * iota

        def generate_shock_series():
            np.random.seed(0)  # For reproducibility
            epsilon = np.random.normal(-0.5 * sigma_epsilon**2, sigma_epsilon, size=120)
            log_kappa = np.zeros(120)
            for t in range(1, 120):
                log_kappa[t] = rho * log_kappa[t-1] + epsilon[t]
            kappa = np.exp(log_kappa)
            return kappa

        K = 1000  # Number of shock series to generate
        h_values = np.zeros(K)
        for k in range(K):
            kappa_series = generate_shock_series()
            ell_prev = 0
            h = 0
            for t in range(120):
                kappa = kappa_series[t]
                ell = ((1 - eta) * kappa / w) ** (1 / eta)
                h += (R ** t) * calculate_profit(kappa, ell, ell_prev)
                ell_prev = ell
            h_values[k] = h

        H = np.mean(h_values)
        print(f"Ex ante expected value (H): {H:.4f}")

    def Q3():

        eta = 0.5
        w = 1.0
        rho = 0.9
        iota = 0.01
        sigma_epsilon = 0.1
        R = (1 + 0.01) ** (1 / 12)
        delta = 0.05

        def calculate_profit(kappa, ell, ell_prev):
            return kappa * ell**(1 - eta) - w * ell - (ell != ell_prev) * iota

        def generate_shock_series():
            np.random.seed(0)  # For reproducibility
            epsilon = np.random.normal(-0.5 * sigma_epsilon**2, sigma_epsilon, size=120)
            log_kappa = np.zeros(120)
            for t in range(1, 120):
                log_kappa[t] = rho * log_kappa[t-1] + epsilon[t]
            kappa = np.exp(log_kappa)
            return kappa

        K = 1000  # Number of shock series to generate
        h_values = np.zeros(K)
        for k in range(K):
            kappa_series = generate_shock_series()
            ell_prev = 0
            h = 0
            for t in range(120):
                kappa = kappa_series[t]
                ell_star = ((1 - eta) * kappa / w) ** (1 / eta)
                if abs(ell_prev - ell_star) > delta:
                    ell = ell_star
                else:
                    ell = ell_prev
                h += (R ** t) * calculate_profit(kappa, ell, ell_prev)
                ell_prev = ell
            h_values[k] = h

        H = np.mean(h_values)
        print("Ex ante expected value (H_old): 39.3185")
        print(f"Ex ante expected value (H_new): {H:.4f}")


    def Q4():

        eta = 0.5
        w = 1.0
        rho = 0.9
        iota = 0.01
        sigma_epsilon = 0.1
        R = (1 + 0.01) ** (1 / 12)

        def calculate_profit(kappa, ell, ell_prev):
            return kappa * ell**(1 - eta) - w * ell - (ell != ell_prev) * iota

        def generate_shock_series():
            np.random.seed(0)  # For reproducibility
            epsilon = np.random.normal(-0.5 * sigma_epsilon**2, sigma_epsilon, size=120)
            log_kappa = np.zeros(120)
            for t in range(1, 120):
                log_kappa[t] = rho * log_kappa[t-1] + epsilon[t]
            kappa = np.exp(log_kappa)
            return kappa

        K = 1000  # Number of shock series to generate

        delta_values = np.linspace(0.01, 0.4, num=50)  # Range of Delta values to try
        H_values = np.zeros_like(delta_values)

        for i, delta in enumerate(delta_values):
            h_values = np.zeros(K)
            for k in range(K):
                kappa_series = generate_shock_series()
                ell_prev = 0
                h = 0
                for t in range(120):
                    kappa = kappa_series[t]
                    ell_star = ((1 - eta) * kappa / w) ** (1 / eta)
                    if abs(ell_prev - ell_star) > delta:
                        ell = ell_star
                    else:
                        ell = ell_prev
                    h += (R ** t) * calculate_profit(kappa, ell, ell_prev)
                    ell_prev = ell
                h_values[k] = h
            H_values[i] = np.mean(h_values)

        optimal_delta = delta_values[np.argmax(H_values)]
        max_H = np.max(H_values)

        print(f"Optimal Delta: {optimal_delta:.4f}")
        print(f"Maximum H: {max_H:.4f}")

        plt.plot(delta_values, H_values)
        plt.xlabel('Delta')
        plt.ylabel('H')
        plt.title('Ex Ante Expected Value (H) as a Function of Delta')
        plt.grid(True)
        plt.show()

    def Q5():

        eta = 0.5
        w = 1.0
        rho = 0.9
        iota = 0.01
        sigma_epsilon = 0.1
        R = (1 + 0.01) ** (1 / 12)

        def calculate_profit(kappa, ell, ell_prev):
            return kappa * ell**(1 - eta) - w * ell - (ell != ell_prev) * iota

        def generate_shock_series():
            np.random.seed(0)  # For reproducibility
            epsilon = np.random.normal(-0.5 * sigma_epsilon**2, sigma_epsilon, size=120)
            log_kappa = np.zeros(120)
            for t in range(1, 120):
                log_kappa[t] = rho * log_kappa[t-1] + epsilon[t]
            kappa = np.exp(log_kappa)
            return kappa

        K = 1000  # Number of shock series to generate
        h_values = np.zeros(K)
        for k in range(K):
            kappa_series = generate_shock_series()
            ell_prev = 0
            ell_prev2 = 0  # Previous hairdresser level two periods ago
            h = 0
            for t in range(120):
                kappa = kappa_series[t]
                ell = ((1 - eta) * kappa / w) ** (1 / eta)
                if t >= 2:  # Apply lagged response from the third period onwards
                    ell = ell + 0.5 * (ell_prev - ell_prev2)
                h += (R ** t) * calculate_profit(kappa, ell, ell_prev)
                ell_prev2 = ell_prev
                ell_prev = ell
            h_values[k] = h

        H = np.mean(h_values)
        print(f"Ex ante expected value (H) with modified policy: {H:.4f}")
        print("H_max_old: 40.1462")

class Problem3:

    def Q1part1():

        def griewank_function(x, y):
            n = 2  # The Griewank function is defined for two variables
            sum_term = (x**2 + y**2) / 4000
            prod_term = np.cos(x / np.sqrt(1)) * np.cos(y / np.sqrt(2))
            return sum_term - prod_term + 1

        # Generate data for plotting
        x = np.linspace(-100, 100, 100)
        y = np.linspace(-100, 100, 100)
        X, Y = np.meshgrid(x, y)
        Z = griewank_function(X, Y)

        # Create the plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, Z, cmap='viridis')

        # Set labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Griewank Function')

        # Show the plot
        plt.show()


    def Q1part2():

        def griewank_function(x):
            n = len(x)
            sum_term = np.sum(x**2 / 4000)
            prod_term = np.prod(np.cos(x / np.sqrt(np.arange(1, n + 1))))
            return sum_term - prod_term + 1

        def refined_global_optimizer(bounds, tau, K_warmup, K_max):
            x_ast = None
            for k in range(K_max):
                x_k = np.random.uniform(bounds[0], bounds[1], size=2)  # Generate 2 values for x_k
                
                if k >= K_warmup:
                    chi_k = 0.5 * 2 / (1 + np.exp((k - K_warmup) / 100))
                    x_k0 = chi_k * x_k + (1 - chi_k) * x_ast
                else:
                    x_k0 = x_k

                result = minimize(griewank_function, x_k0, method='BFGS', tol=tau)
                x_k_ast = result.x

                if k == 0 or griewank_function(x_k_ast) < griewank_function(x_ast):
                    x_ast = x_k_ast

                if griewank_function(x_ast) < tau:
                    break

            return x_ast

        # Settings
        bounds = [-600, 600]
        tau = 1e-8
        K_warmup = 10
        K_max = 1000

        # Run optimizer for x1
        x_ast = refined_global_optimizer(bounds, tau, K_warmup, K_max)

        # Plotting the effective initial guesses for x1
        initial_guesses_x1 = []
        for k in range(K_max):
            if k >= K_warmup:
                chi_k = 0.5 * 2 / (1 + np.exp((k - K_warmup) / 100))
                x_k = np.random.uniform(bounds[0], bounds[1], size=2)
                x_k0 = chi_k * x_k + (1 - chi_k) * x_ast
            else:
                x_k0 = np.random.uniform(bounds[0], bounds[1], size=2)

            initial_guesses_x1.append(x_k0[0])  # Store only x1 values

        plt.figure(figsize=(10, 6))
        plt.plot(range(K_max), initial_guesses_x1, label='x1')
        plt.xlabel('Iteration (k)')
        plt.ylabel('Initial Guess for x1')
        plt.title('Variation of Effective Initial Guesses for x1')
        plt.legend()
        plt.show()

        # Run optimizer for x2
        x_ast = refined_global_optimizer(bounds, tau, K_warmup, K_max)

        # Plotting the effective initial guesses for x2
        initial_guesses_x2 = []
        for k in range(K_max):
            if k >= K_warmup:
                chi_k = 0.5 * 2 / (1 + np.exp((k - K_warmup) / 100))
                x_k = np.random.uniform(bounds[0], bounds[1], size=2)
                x_k0 = chi_k * x_k + (1 - chi_k) * x_ast
            else:
                x_k0 = np.random.uniform(bounds[0], bounds[1], size=2)

            initial_guesses_x2.append(x_k0[1])  # Store only x2 values

        plt.figure(figsize=(10, 6))
        plt.plot(range(K_max), initial_guesses_x2, label='x2')
        plt.xlabel('Iteration (k)')
        plt.ylabel('Initial Guess for x2')
        plt.title('Variation of Effective Initial Guesses for x2')
        plt.legend()
        plt.show()

    def Q2():

        def griewank_function(x):
            n = len(x)
            sum_term = np.sum(x**2 / 4000)
            prod_term = np.prod(np.cos(x / np.sqrt(np.arange(1, n + 1))))
            return sum_term - prod_term + 1

        def refined_global_optimizer(bounds, tau, K_warmup, K_max):
            x_ast = None
            num_iterations = 0
            for k in range(K_max):
                x_k = np.random.uniform(bounds[0], bounds[1], size=2)  # Generate 2 values for x_k

                if k >= K_warmup:
                    chi_k = 0.5 * 2 / (1 + np.exp((k - K_warmup) / 100))
                    x_k0 = chi_k * x_k + (1 - chi_k) * x_ast
                else:
                    x_k0 = x_k

                result = minimize(griewank_function, x_k0, method='BFGS', tol=tau)
                x_k_ast = result.x

                if k == 0 or griewank_function(x_k_ast) < griewank_function(x_ast):
                    x_ast = x_k_ast

                if griewank_function(x_ast) < tau:
                    num_iterations = k + 1  # Count the number of iterations
                    break

            return x_ast, num_iterations

        def compare_convergence():
            # Set seed for reproducibility
            np.random.seed(42)

            # Settings
            bounds = [-600, 600]
            tau = 1e-8
            K_warmup_1 = 10
            K_warmup_2 = 100
            K_max = 1000

            # Run optimizer multiple times for each K_warmup
            num_runs = 10
            iterations_1 = []
            iterations_2 = []

            for _ in range(num_runs):
                # Run optimizer for K_warmup = 10
                _, num_iterations_1 = refined_global_optimizer(bounds, tau, K_warmup_1, K_max)
                iterations_1.append(num_iterations_1)

                # Run optimizer for K_warmup = 100
                _, num_iterations_2 = refined_global_optimizer(bounds, tau, K_warmup_2, K_max)
                iterations_2.append(num_iterations_2)

            # Plotting the number of iterations
            plt.figure(figsize=(10, 6))
            plt.plot(range(num_runs), iterations_1, label='K_warmup = 10')
            plt.plot(range(num_runs), iterations_2, label='K_warmup = 100')
            plt.xlabel('Run')
            plt.ylabel('Number of Iterations to reach threshold (tau)')
            plt.title('Number of Iterations Comparison')
            plt.legend()
            plt.show()

        compare_convergence()



        


            
        


        