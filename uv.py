# CREDIT: Abhishek Dhameja @abhishekdhameja on GitHub
# LINK: https://github.com/abhishekdhameja/UV-Decomposition-for-Recommender-Systems-in-Python
import sys
from collections import defaultdict
import numpy as np
np.set_printoptions(threshold=np.inf) # Turn off truncation
# rating-50k.csv: python uv.py rating-50k.csv 370 5382 40 1
# rating-10k.csv: python uv.py rating-100K.csv 370 5382 40 1
# cut_rating: python uv.py ratings_cut.csv 100 4382 40 10

def decomposeMatrixU(): # procedure decomposeMatrixU(n, d, M, U, V)
    # Require: N > 0, d > 0
    for r in range(0,n): # n - # of rows (users of matrix) | for 0, n do
        for s in range(0,f): # f - d dimensions | # for 0, d do
            m_array=np.array(M[r,:]) # m_array <- row n from the M matrix
            v_array=np.array(V[s,:]) # v_array <- row d from the V matrix
            # ∀row, col index location ∈ m_array if m_array_(row, col) is null then
                # v_array_(row, col) <- null
            # end if
            v_array[np.isnan(m_array)]=np.nan # 
            # print (m_array)
            # print v_array
            # denominator <- Σ value^2 ∈ v_array
            denominator=np.nansum(np.square(v_array))
            # print denominator
            # sum_array <- (U_n ⋅ V) - (U_(n, d) × V_d)
            sum_array=np.matmul(U[r,:],V[:])-(U[r,s]*V[s,:])
            # # print sum_array
            # numerator <- Σ V_d × (m_array - sum_array) 
            numerator=np.nansum(V[s,:]*(m_array-sum_array))
            # numerator <- np.nansum(V[s, :] * sum)
            # print numerator
            # U_(n, d) <- numerator ÷ denominator
            
            alpha = 0.0002
            beta = 0.02
            U[r,s]=float(numerator)/denominator
            U[r,s] += alpha * (2 * V[s, r] - beta * U[r,s])

    return

def decomposeMatrixV():
    for s in range(0,m): # m - # of columns (movies)
        for r in range(0,f): # f - d dimensions
            m_array=np.array(M[:,s])
            u_array=np.array(U[:,r])
            u_array[np.isnan(m_array)]=np.nan
            # print m_array
            # print v_array
            denominator=np.nansum(np.square(u_array))
            # print denominator
            sum_array = np.matmul(U[:], V[:,s]) - (V[r, s] * U[:, r])
            # print sum_array
            numerator = np.nansum(U[:, r] * (m_array - sum_array))
            # print numerator
            
            alpha = 0.0002
            beta = 0.02
            V[r,s]=float(numerator)/denominator
            V[r,s] += alpha * (2 * U[r,r] - beta * V[r,s])

    return

def rmse():
    UV=np.dot(U,V)
    sum=0
    for r in range(0,n):
        for s in range(0,m):
            if np.isnan(M[r,s]):
                continue
            # print M[r,s]
            # print np.sum(U[r,:]*V[:,s])
            #squared_differences.append((M[r,s]-np.sum(U[r,:]*V[:,s]))**2)
            sum+=(M[r,s]-UV[r,s])**2
    # print 'sq_diff=',squared_differences
    # print non_zero_entries
    mean=float(sum)/non_zero_entries
    print ('%.4f' % mean**(0.5))
    return


if __name__ == '__main__':
    if len(sys.argv) != 6:
        print('Usage: python Abhishek_Dhameja_uv.py <ratings-file> <n> <m> <f> <k>') 
        exit(-1)
    inputfile=open(sys.argv[1]) 
    n=int(sys.argv[2]) # Must provide 4 arguments
    m=int(sys.argv[3])
    f=int(sys.argv[4]) # also d
    k=int(sys.argv[5])

    data=defaultdict(dict)
    all_movies=set()
    all_users=set()
    lines = inputfile.readlines()
    for line in lines: # Go line by line
        line=line.strip().split(',')
        if line[0]=='userId':
            continue
        line=line[:-1]
        userid=int(line[0])
        if userid>n:
            break
        else: # Store them in a structured set.
            movieid=int(line[1])
            rating=float(line[2])
            data[userid][movieid]=rating
            all_movies.add(movieid)
            all_users.add(userid)



    all_movies=sorted(all_movies)
    all_movies=all_movies[:m]
    all_users=sorted(all_users)

    # Initialize M matrix
    # For each row corresponding to a user, store the rating of a movie in its respective column
    M=np.empty((n,m))
    M[:]=np.nan
    non_zero_entries=0
    for x in range(0,n):
        user=all_users[x]
        for y in range(0,m):
            movie=all_movies[y]
            if movie in data[user]:
                M[x,y]=data[user][movie]
                non_zero_entries+=1
    #print data
    
    # Initialize UV matrices
    U=np.ones((n,f)) 
    V=np.ones((f,m))
    # U[0,0]=2.6
    # print(np.dot(U,V))

    # For each iteration k, decompose UV matrices and get rmse
    for i in range(0,k):
        decomposeMatrixU()
        decomposeMatrixV()
        rmse()
    # print(U)
    # print(V)