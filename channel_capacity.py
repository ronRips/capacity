import numpy as np
import os
import control as ct
import cvxpy as cp
from scipy.linalg import lu

os.environ["PATH"] += ";C:\\Users\\aharon.rips\\Downloads\\OpenBLAS-0.3.28-x86\\bin"



def dare_first_model(F,H,W,V,L):
    sigma,I, k_p = ct.dare(F.T,H.T,W,V,L)
    psi = H@sigma@H.T +V
    return sigma, psi, k_p.T

def parameters(k,p,m):
    J = np.random.rand(p,m)
    F = np.random.rand(k,k)
    F[0,0] = 0
    G = np.random.rand(k,m)
    H = np.random.rand(p,k)
    H[0,0] = 0.5
    a = np.random.rand(k,k)
    W = (a @ a.T)
    W[0,0] = 1
    ab = np.random.rand(p,p)
    V = (ab @ ab.T) 
    V[0,0] = 1
    L = np.random.rand(k,p)
    L[0,0] = 1
    a = np.random.rand(k,k)
    Q = (a @ a.T) 
    a = np.random.rand(m,m)
    R = (a @ a.T) 
    return J, F, G, H, W, V, L, Q, R

def optimization_parameters(F, G, Q, R):
    E, E1, E2 = ct.dare(F, G, Q, R)
    B = R +G.T@E @G
    C= F - G@ np.linalg.inv(B) @G.T @E@F
    D = F.T @ E @ G @ np.linalg.inv(B) @ G.T @ E @ F
    return B, C, D, E

def log():
    c = np.array([[1, 2, 3], [4, 5, 6], [3, 2, 1]])
    print(cp.log_det(c))

def capacity(k,p,m, power):
    J, F, G, H, W, V, L, Q, R = parameters(k,p,m)
    sigma,psi, k_p = dare_first_model(F,H,W,V,L)
    B, C, D, E = optimization_parameters(F, G, Q, R)
    
    Pi = cp.Variable((m, m), symmetric=True)  
    Gamma = cp.Variable((m, k))  
    Sigma_hat = cp.Variable((k, k), symmetric=True) 
    N = cp.Variable((k, k), symmetric=True)
    
    Psi_Y = J @ Pi @ J.T + H @ Sigma_hat @ H.T + H @ Gamma.T @ J.T + J @ Gamma @ H.T + psi
    K_Y = F @ Gamma.T @ J.T + F  @ Sigma_hat @ H.T + G @ Pi @ J.T + G @ Gamma @ H.T + k_p @ psi
    
    objective = cp.Maximize(0.5 * cp.log_det(Psi_Y) - cp.log_det(psi))
    
    constraint0 = cp.trace(B @ Pi) + cp.trace( F @ Sigma_hat @ F.T @ Q + 2 * F.T @ Q @ G @ Gamma + k_p @ psi @ k_p.T @ Q + D @ N ) + cp.trace(sigma @ Q) 
    
    constraint1 = cp.bmat([[Pi, Gamma], [Gamma.T, Sigma_hat]])
    
    cons2 = F @ Sigma_hat @ F.T + F @Gamma.T @ G.T + G @ Gamma @ F.T + G @ Pi @ G.T + k_p @ psi @k_p.T - Sigma_hat

    constraint2 = cp.bmat([[cons2, K_Y],
                            [K_Y.T, Psi_Y]])
    
    constraint3 = cp.bmat([[C @ N @ C.T - N, K_Y], [K_Y.T, Psi_Y]])


    # Constraints: Pi and Sigma_hat must be positive semidefinite
    constraints = [Pi >> 0, Sigma_hat >> 0, N >> 0, constraint0 <= power, constraint1 >> 0, constraint2 >> 0, constraint3 >> 0]

    # Define the optimization problem
    problem = cp.Problem(objective, constraints)
    problem.solve()  # Use MOSEK if available
    
def little_problem():
    power = 12
    J, F, G, H, W, V, L, Q, R = 0, 0, 0, 1.2, 1, 1, 1, 0.5, 1
    sigma,I, k_p = ct.dare(F,H,W,V,L)
    sigma = sigma[0,0]
    k_p = k_p[0,0]
    print("sigma" , sigma , "k_p" , k_p)

    psi = H*sigma*H +V
    print("psi", psi)
    E, E1, E2 = ct.dare(F, G, Q, R)
    B = R 
    C= F 
    D = F 
    Pi = cp.Variable()  
    Gamma = cp.Variable()  
    Sigma_hat = cp.Variable() 
    N = cp.Variable()
    Psi_Y =  H * Sigma_hat * H + psi
    K_Y = k_p * psi
    objective = cp.Maximize(0.5 * cp.log(Psi_Y) - cp.log(psi))
    
    constraint0 = Pi +    k_p * psi *k_p *Q   + sigma *Q
    
    constraint1 = cp.bmat([[Pi, Gamma], [Gamma, Sigma_hat]])
    
    cons2 =  k_p * psi * k_p - Sigma_hat

    constraint2 = cp.bmat([[cons2, K_Y],
                            [K_Y, Psi_Y]])
    print('constraint', N)
    print('constraint', K_Y)
    print('constraint', Psi_Y)


    constraint3 = cp.bmat([[- N, K_Y], [K_Y, Psi_Y]])
    constraints = [Pi >= 0, Sigma_hat >= 0, N >= 0, constraint0 <= power, constraint1 >> 0, constraint2 >> 0, constraint3 >> 0]
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.SCS)
    print("Optimal Value (SCS):", problem.value)
    # problem.solve(solver=cp.ECOS)
    # print("Optimal Value (ECOS):", problem.value)

# Output results
    print("Optimal value:", problem.value)
    # print("SCS:", problem.solve(solver=cp.SCS))
    # print("ECOS:", problem.solve(solver=cp.ECOS))
    # print("Is the problem DCP-compliant?", problem.is_dcp())  # Check if problem satisfies CVXPY rules
    # print("Is the problem solvable?", problem.status) 
    # print(problem.solver_stats)

   
    
    

if __name__ == "__main__":
    capacity(1,1,3, 90)
    # x = cp.Variable()
    # y = cp.Variable()

    # cons = cp.vstack([cp.hstack([x , 2]), cp.hstack([2, y])])
    
    # constraints = [x + y == 1, x >= 0, y >= 0, cons >> 0]
    # objective = cp.Minimize(x**2 + y**2)
    # problem = cp.Problem(objective, constraints)

    # Solve with different solvers
    # print("SCS:", problem.solve(solver=cp.SCS))
    # print("ECOS:", problem.solve(solver=cp.ECOS))
    # little_problem()
    # log()
   # print(os.environ["PATH"])
    # capacity(1,1,3, 90)
    # print("NumPy BLAS Info:", np.__config__.show())
# A = np.array([[3, 1], [1, 2]])
# P, L, U = lu(A)
# print("LU Factorization:", L, U)