import numpy as np

def gf2elim(M):
    m,n = M.shape
    i=0
    j=0
    while i < m and j < n:
        # find value and index of largest element in remainder of column j
        k = np.argmax(M[i:, j]) +i
        # swap rows
        #M[[k, i]] = M[[i, k]] this doesn't work with numba
        temp = np.copy(M[k])
        M[k] = M[i]
        M[i] = temp
        aijn = M[i, j:]
        col = np.copy(M[:, j]) #make a copy otherwise M will be directly affected
        col[i] = 0 #avoid xoring pivot row with itself
        flip = np.outer(col, aijn)
        M[:, j:] = (M[:, j:] ^ flip)%2
        i += 1
        j +=1
    return M

def a_list_to_parity_check(a_list):
    n = a_list[0][0]
    m = a_list[0][1]
    
    var_idx = a_list[4:4+n]
    check_idx = a_list[4+n:4+n+1+m]

    parity_check_matrix = np.zeros([n,m])
    parity_check_matrix_2 = np.zeros([n,m])
    
    # V1
    for i in range(n):
        for idx in var_idx[i]:
            if idx != 0:
                parity_check_matrix[i][(idx-1)] = 1

    # V2
    for i in range(m):
        for idx in check_idx[i]:
            if idx != 0:
                parity_check_matrix_2[(idx-1)][i] = 1

    return parity_check_matrix

def gm_to_generator(gm):
    for line_index in np.arange(len(gm)):
        if gm[line_index][0] == 'nb_cols':
            m = int(gm[line_index][3])
        elif gm[line_index][0] == 'nb_rows':
            n = int(gm[line_index][3])
        elif gm[line_index][0] == 'src_indx':
            src_index = np.array(gm[line_index+1], dtype=np.int32)
        elif gm[line_index][0] == 'par_indx':
            par_index = np.array(gm[line_index+1], dtype=np.int32)
        elif gm[line_index][0] == 'nzentries_per_row':
            start = line_index
            continue

    parity_matrix = np.zeros(shape=(n,m), dtype=np.float32)
    generator_line = 0

    for line_index in np.arange(start=start+1, stop=len(gm)):
        for a in gm[line_index]:
            parity_matrix[generator_line][int(a)] = 1
        generator_line += 1
    
    k = len(src_index)
    generator_matrix = np.zeros((m,k), dtype=np.float32)

    basis = np.zeros(k, dtype=np.float32)
    basis[0] = 1.

    for basis_index in np.arange(k):
        src_word = np.roll(basis, basis_index)
        generator_matrix[src_index, basis_index] = src_word
        
    for line_index in np.arange(n):
        res = np.matmul(parity_matrix[line_index,:], generator_matrix) % 2
        generator_matrix[par_index[line_index], :] = res

    generator_matrix = np.transpose(generator_matrix)

    return (generator_matrix, src_index, par_index)

def generate_code(a_list_path):
    with open(a_list_path, "r") as myfile:
        data = myfile.readlines()
        a_list = [[int(x) for x in elt.replace(' \n','').replace(' ',',').split(',')] for elt in data]
    
    n = a_list[0][0]
    k = n - a_list[0][1] 

    H_non_systematic = np.array(a_list_to_parity_check(a_list).T, dtype=np.uint8)
    
    H_systematic = np.array(gf2elim(np.copy(H_non_systematic)), dtype=np.float32)

    redundancy_H = np.array([H_systematic[i][n-k:] for i in range(n-k)])

    H_systematic = np.concatenate((redundancy_H,np.eye(n-k)),axis=-1)
    
    G = np.concatenate((np.eye(k),redundancy_H.T),axis=-1)
    
    redundancy_G = np.array([G[i][k:] for i in range(k)])

    return(H_non_systematic.astype(np.float32), H_systematic, G, redundancy_H, redundancy_G)

def ldpc_generate_code(a_list_path, gm_path):
    with open(a_list_path, "r") as myfile:
        data = myfile.readlines()
        a_list = [[int(x) for x in elt.replace(' \n','').replace(' ',',').split(',')] for elt in data]
    
    n = a_list[0][0]
    k = n - a_list[0][1] 

    H_non_systematic = np.array(a_list_to_parity_check(a_list).T, dtype=np.float32)

    with open(gm_path, "r") as myfile:
        data = myfile.readlines()
        gm = [[x for x in elt.replace(' \n','').replace('\n','').replace(' ',',').split(',')] for elt in data]

    G, src_index, par_index = gm_to_generator(gm)

    H_systematic = H_non_systematic

    return (H_non_systematic, H_systematic, G, src_index, par_index)

def save_as_npz(code_name="BCH_7_4", main_save_path="./"):
    a_list_path = main_save_path + code_name + "_alist.txt"
    H_non_systematic, H_systematic, G, redundancy_H, redundancy_G = generate_code(a_list_path)

    npz_path = main_save_path + code_name + ".npz"
    np.savez(npz_path, G=G, H_systematic=H_systematic, H_non_systematic=H_non_systematic, redundancy_H=redundancy_H, redundancy_G=redundancy_G)

def ldpc_save_as_npz(code_name, main_save_path="./"):
    a_list_path = main_save_path + code_name + ".alist"
    gm_path = main_save_path + code_name + ".gm"

    H_non_systematic, H_systematic, G, src_index, par_index = ldpc_generate_code(a_list_path, gm_path)

    npz_path = main_save_path + code_name + ".npz"
    np.savez(npz_path, G=G, H_systematic=H_systematic, H_non_systematic=H_non_systematic, src_index=src_index, par_index=par_index)