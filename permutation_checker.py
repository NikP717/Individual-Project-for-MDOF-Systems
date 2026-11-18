import numpy as np

fix_c = False
fix_k = False
k_range = [1,100000]
kstep = 10000
dof = 4
c_range = [1,10]
cstep = 1

k_value_array = np.arange(k_range[0],k_range[1]+kstep, kstep).tolist()
c_value_array = np.arange(c_range[0],c_range[1]+cstep, cstep).tolist()
if fix_c:
    permutations = len(k_value_array)**(dof-1)
if fix_k:
    permutations = len(c_value_array)**(dof-1)
else:
    permutations = len(k_value_array)**(dof-1)*len(c_value_array)**(dof-1)
print(f"Number of permutations: {permutations:,}")
print(f"name='', dof = {dof}, k_range = {k_range}, k_step={kstep}, c_range = {c_range}, c_step={cstep}, fix_k={fix_k},fix_c={fix_c}")

# name='', dof = 2, k_range = [1, 100000], k_step=10, c_range = [0.1, 10], c_step=0.1, fix_k=False,fix_c=False
# name='', dof = 3, k_range = [1, 100000], k_step=1000, c_range = [1, 10], c_step=1, fix_k=False,fix_c=False
# name='', dof = 4, k_range = [1, 100000], k_step=10000, c_range = [1, 10], c_step=1, fix_k=False,fix_c=False