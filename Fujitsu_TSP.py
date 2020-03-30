#%%
# import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
from tqdm import tqdm as tqdm
from tqdm import trange
import math
pi = math.pi

#%%
# Prepare for distance matrix and points
# call distance_mat by distance_mat[start][end] e.g. distance_mat[1][2]
city_size = 32
mu, sigma = 0, 100
X_coordinate = np.random.normal(mu, sigma, city_size)
Y_coordinate = np.random.normal(mu, sigma, city_size)
point_idx = np.arange(1,city_size+1,1)
coordinate = np.array(list(zip(X_coordinate, Y_coordinate)))
point_df  = pd.DataFrame(coordinate, columns=['xcord', 'ycord'], index=point_idx)
distance_mat = pd.DataFrame(distance_matrix(point_df.values, point_df.values), index=point_df.index, columns=point_df.index).to_numpy()
qubits_mat = np.random.binomial(1,0.5, size = city_size**2).reshape((city_size, city_size))

#%%
# circle version
city_size = 32
radius = 10
def PointsInCircum(r,n=100):
    return [(math.cos(2*pi/n*x)*r,math.sin(2*pi/n*x)*r) for x in range(0,n)]
coordinate = np.array(PointsInCircum(radius,city_size))
X_coordinate = coordinate[:,0]
Y_coordinate = coordinate[:,1]
point_idx = np.arange(1,city_size+1,1)
point_df  = pd.DataFrame(coordinate, columns=['xcord', 'ycord'], index=point_idx)
distance_mat = pd.DataFrame(distance_matrix(point_df.values, point_df.values), index=point_df.index, columns=point_df.index).to_numpy()
qubits_mat = np.random.binomial(1,0.5, size = city_size**2).reshape((city_size, city_size))

plt.scatter(X_coordinate,Y_coordinate,)
#%%
# defining functions
def calculate_E(city_size, qubit_mat, dist_mat, A, B):

    a_term = np.sum((1-np.sum(qubit_mat, axis = 0))**2) + np.sum((1-np.sum(qubit_mat, axis = 1))**2)
    b_term = 0
    for i in range(city_size-1):
        first = np.where(qubit_mat[:,i]==1)[0]
        second = np.where(qubit_mat[:,i+1]==1)[0]
        first, second = np.repeat(first, len(second)), np.tile(second, len(first))
        b_term += np.sum(distance_mat[(first,second)])
    return A*a_term , B*b_term
        
def calculate_delta_Ei(xi, xj, qubit_mat, dist_mat, A, B):
    specify_qubit = qubit_mat[xi,xj]
    delta_x = 1-2*specify_qubit

    a_term1 = -4*A*delta_x
    a_term2 = A*(2*delta_x*(np.sum(qubit_mat[xi,:])+np.sum(qubit_mat[:,xj]))+2)
    b_term = B * delta_x * (np.sum(dist_mat[xi,:]*qubit_mat[:,(xj+1)%32]) + np.sum(dist_mat[xi,:]*qubit_mat[:,(xj+31)%32]))
    return a_term1 + a_term2 + b_term

def ADB(delta_Ei, beta, E_off, accept_comp):
    delta_Ei = delta_Ei - E_off
    alpha = np.min([1, np.exp(-delta_Ei*beta)])
    if alpha >= accept_comp:
        return 1
    else:
        return 0

def E_off_gen(flag, delta_Ei, E_off):
    if flag == 1:
        return 0
    else:
        return E_off + delta_Ei

#%%
# main
def TSPsolver(city_size, qubits_mat, distance_mat, beta, beta_increment, E_off, flip_count, E_off_increment):
    delta_Ei_list = []
    accept_list = []
    qubit_nums = city_size ** 2
    accept_comp = np.random.uniform(0,1,qubit_nums)
    for now_qubit in range(qubit_nums):
        delta_Ei = calculate_delta_Ei(now_qubit//city_size, now_qubit%city_size, qubits_mat, distance_mat, A,B)
        delta_Ei_list.append(delta_Ei)
        accept_list.append(ADB(delta_Ei, beta, E_off, accept_comp[now_qubit]))
    
    beta += beta_increment
    accept_list = np.array(accept_list)
    flip_array = np.where(accept_list==1)[0]

    if len(flip_array) == 0:
        # E_off = E_off_gen(0, np.sum(delta_Ei_list), E_off)
        # E_off += np.max(np.abs(delta_Ei_list))
        E_off += E_off_increment
    else:
        flip_pos = np.random.choice(flip_array, 1)
        row, col = flip_pos//city_size, flip_pos%city_size
        E_off = 0
        flip_count+=1

        qubits_mat[row,col] = 1 - qubits_mat[row,col]
    # print(flip_array)
    # print(flip_pos)
    
    return qubits_mat,beta, E_off, flip_count

#%%

max_beta, min_beta, iteration = 5,0.1,10000
beta = min_beta
B = 1
A = np.max(distance_mat)*B+1
E_off_increment = 1
beta_increment = (max_beta - min_beta) / iteration
E_off = 0
flip_count = 0
trange = tqdm(range(iteration), total = iteration)
print((A,B))
for i in trange:
    qubits_mat, beta, E_off, flip_count = TSPsolver(city_size,
                                                    qubits_mat,
                                                    distance_mat,
                                                    beta,
                                                    beta_increment, 
                                                    E_off, 
                                                    flip_count,
                                                    E_off_increment)
    a_term,b_term = calculate_E(city_size, qubits_mat, distance_mat, A,B)
    trange.set_postfix(Energy = a_term+b_term,
                       one_bits = np.sum(qubits_mat), 
                       beta = beta, 
                       E_off = E_off,
                       Aterm = a_term,
                       Bterm = b_term,
                       flip = flip_count)

#%%
# make plot
# find city order
start_point, end_point = [],[]
for i in range(city_size-1):
    first = np.where(qubits_mat[:,i]==1)[0]
    second = np.where(qubits_mat[:,i+1]==1)[0]
    first, second = np.repeat(first, len(second)), np.tile(second, len(first))
    start_point.extend(first)
    end_point.extend(second)

start_point = np.array(start_point).reshape(-1)
end_point = np.array(end_point).reshape(-1)
all_point = np.concatenate((start_point, end_point))

plot_x_set = []
plot_y_set = []
for i in all_point:
    plot_x_set.append(coordinate[i][0])
    plot_y_set.append(coordinate[i][1])

plt.plot(plot_x_set,plot_y_set,'r')
plt.scatter(plot_x_set,plot_y_set,)
x = [plot_x_set[0],plot_x_set[-1]]
y = [plot_y_set[0],plot_y_set[-1]]
plt.plot(x, y, 'k') 
plt.show()
# %%
