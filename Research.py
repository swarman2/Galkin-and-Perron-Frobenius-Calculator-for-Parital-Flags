from helper_functions import *
import numpy as np
import matplotlib.pyplot as plt
from time import time
import multiprocessing #for threading
import threading #also for threading
"""alph is an ordered pair [l,s] where s is the highest number in alpha, and l is the length of alpha"""
def Thm1(alph,len_u,len_w,sum):
  """
  if not len(d) == len(a):
    print("Invalid Input")
    return -1
  if not max(u)==max(w):
    print("Invalid Input")
    return -1
  for i in range(len(a)-1):
    if a[i]>=a[i+1]:
      print("Invalid Input")
      return -1
  for i in range(len(a)):
    if a[i]>=max(u):
       print("Invalid Input")
       return -1
  for i in range(1,max(u)+1):
    if not i in u or not i in w:
       print("Invalid Input")
       return -1
  if not alph[1] in a or alph[1]<alph[0]:
    print("Invalid Input")
    return -1
  n = max(u)
  j = a.index(alph[1])+1
  if not Piere(d,j):
    return 0
  sum_1 = 0
  sum_2 = 0
  k = len(a)
  a.append(n)
  for i in range(0,k):
    sum_1 = sum_1 + a[i]*(a[i+1]-a[i])
    if i == 0:
      sum_2 = sum_2 + (a[i+1])*d[i]
    else:
      sum_2 = sum_2 + (a[i+1] - a[i-1])*d[i]
  a.pop()
  """
  if len_u + alph[0]==len_w + sum:
    return 1
  else:
    return 0


def make_matrix(alph,d,a,array):
  #find the sum used in the thm
  #print("array = ",array)
  n=max(array[0])
  sum = 0
  k = len(a)
  a.append(n)
  for i in range(0,k):
    if i == 0:
      sum = sum + (a[i+1])*d[i]
    else:
      sum = sum + (a[i+1] - a[i-1])*d[i]
  a.pop()

  t_thm1=0

  #print("Order of permutations:")
  #print(array)
  mat = [ [ None for i in range(len(array)) ] for j in range(len(array)) ]
  n = max(array[0])
  extendeda=copy.deepcopy(a)
  extendeda.append(n)
  sum_2=0
  k=len(a)
  for i in range(0,k):
    if i == 0:
      sum_2 = sum_2 + (extendeda[i+1])*d[i]
    else:
      sum_2 = sum_2 + (extendeda[i+1] - extendeda[i-1])*d[i]
  l_tuple = array[-1]
  largest=[]
  for i in l_tuple:
    largest.append(i-1)

  swap_largest = swap_perm(a,d,largest)

  gammad=np.asarray(inv(largest))[swap_largest]
  #t_thm1=0
  #print("gamma_arr = ",gamma_arr)
  #print("******************************* d = ",d, " *******************************")
  for i in range(len(array)):
    len_u=l(array[i])
    for j in range(len(array)):
      len_w = l(array[j])
      start=time()
      mat[i][j] = Thm1(alph, len_u, len_w,sum)
      t_thm1+=time()-start
      #print()
      #print("[",i,"][",j,"]")
      if mat[i][j]==1:
        #print("test")
        if d == [0] * len(d):
          if not len_w == len_u+1:
            mat[i][j]=0
          else:
            a_j = alph[1] # A_j is the last subscript of alpha which is alph[1]

            ident = np.arange(n)
            equal_exists = 0
            for b in range(alph[1]):
              for c in range(alph[1],n):
                #ident = np.arange(n)
                #temp = ident[b]
                #ident[b]=ident[c]
                #ident[c] = temp
                #u = np.asarray(array[i])
                #for z in range(len(u)):
                #  u[z] = u[z]-1
                #print("u = ",u)
                #print("w = ",)
                #rhs = ident[u]
                #rhs = fix_perm(a,rhs)
                rhs = np.asarray(array[i])
                temp = rhs[b]
                rhs[b] = rhs[c]
                rhs[c]=temp
                rhs = fix_perm(a,rhs)
                equal = 1
                for z in range(len(rhs)):
                  if not array[j][z] == rhs[z]:
                    equal = 0
                if equal:
                  equal_exists=1

            if not equal_exists:
              mat[i][j]=0
        else: # if d != [0 0 0 .. ]
          #print("array = ",array)
          #print("u = ",array[i],"  w = ",array[j])
          #print("largest = ",largest)
          #print("swap_largest = ",swap_largest)
          #print("gammad = ",gammad)
          #print("u * gammad = ",np.asarray(array[i])[gammad])
          #print("")
          w=[]
          for x in array[j]:
            w.append(x)
          #print("not (",l(np.asarray(array[i])[gammad])," == ",l(array[i])+1-sum_2," and ",np.asarray(array[i])[gammad]," == ",w,")")
          if not(l(np.asarray(array[i])[gammad]) == len_u+1-sum_2 and array_equals(np.asarray(array[i])[gammad],w)):
            mat[i][j]=0

  return mat,t_thm1

def largest_real_eig(alpha,a,n,perm):
  j = a.index(alpha[1])+1

  perm= MergeSort_perm(perm)
  #print(perm)
  mat = [ [0 for i in range(len(perm)) ] for j in range(len(perm)) ]
  #print(get_d(j,a,alpha[0]))
  t1=time()
  t_thm1=0
  for d in get_d(j,a,alpha[0]):
    temp,t_thm_temp = make_matrix(alpha,d,a,perm)
    t_thm1+=t_thm_temp
    #Print(temp)
    #print()
    #print()
    """"
    new code
    """
    mat = add_mat(mat,temp)
  #Print(mat)
  #print("    { INFO } matrix generation: ",round(time()-t1,3)," secs")
  #print("        ( INFO ) Thm 1 calculations: ",round(t_thm1,3)," secs")
  t2=time()
  real_eigvals = []
  for val in np.linalg.eigvals(mat):
    real_eigvals.append(np.real(val))
  #print("    { INFO } eigenvalue calculation: ",round(time()-t2,3)," secs")
  maxReal = max(real_eigvals)
  return maxReal,mat


def main():

  max_n = 10
  max_a = 4
  min_a = 4
  min_n=max_a+1
  #if max_a >= multiprocessing.cpu_count():
    #print(" WOULD USE TOO MANY THREADS")

  Threads=[]
  x = []
  y=[]
  for a2 in range(min_a,max_a+1):
    x.append([])
    y.append([])
    print("x[-1]: ",x[-1],"  y:  ",y)
    main_helper(max_n,min_n,a2,x[-1],y[-1])
  print(x)
  print(y)
  for i in range(len(x)):
    _x = x[i]
    _y = y[i]
    lab = 'a = [1, '+str(i+3)+']'
    plt.plot( _x, _y,label=lab)
    plt.plot( _x, _y,'ro')
    temp_x=[]
    temp_y=[]
    scale_factor = max(int(max_n/5),1)
    for i in range(int(len(_x)/scale_factor)):
      temp_x.append(_x[scale_factor*i])
      temp_y.append(_y[scale_factor*i])
    if max_n%2==0:
      temp_x.append(_x[-1])
      temp_y.append(_y[-1])
    for i_x, i_y in zip(temp_x, temp_y):
      plt.text(i_x, i_y-(1.1)*.03, '({}, {})'.format(i_x, i_y))
  plt.ylabel('max(real_eigvals)')
  plt.xlabel('n')
  plt.legend(loc='upper left',frameon=False)
  plt.axis([3,max_n+1,1,2.1])
  plt.show()

def main_helper(max_n,min_n,a2,xarr,yarr):

  a = [1,a2]
  alpha=[1,1]
  perm = S(n,a)
  #points=[[],[]]
  for i in range(min_n,max_n+1,5):
    t=time()
    #print("[ INFO ] solving n = ",i," a = [1, ",a2,"]")
    xarr.append(i)
    yarr.append(round(largest_real_eig(alpha,a,i,perm),5))
    #print("n = ",i,":  ",round(time()-t,3)," secs")
  #xarr[0]=points[0]
  #yarr[0]=points[1]



  #print(mat[7][4])
def run():
  a = [2]
  n= 4
  mat=[]
  perm = S(n,a)
  print("permutations: ",perm)
  for i in a:
    mat.append(largest_real_eig([1,i],a,n,perm)[1])
  #Print(mat[0])
  #Print(mat[1])
  sum_1 = np.zeros((len(mat[0][0]),len(mat[0][1])))
  sum_2 = 0
  a_w_endpoints = copy.deepcopy(a)
  a_w_endpoints = [0] + a_w_endpoints + [n]
  for i in range(len(a)):
    sum_1+=(a_w_endpoints[i+1]-a_w_endpoints[i-1])*np.asarray(mat[i])
    sum_2+=a_w_endpoints[i]*(a_w_endpoints[i+1]-a_w_endpoints[i])
  real_eigvals = []
  for val in np.linalg.eigvals(sum_1):
    real_eigvals.append(np.real(val))
  E = max(real_eigvals)
  #eignen value for s2, n=4 should be 4 sqrt(2)
  print("E: ",E,"  sum_2: ",sum_2)
  print("E - sum_2 -1 = ",E - sum_2 -1)
  print(4*math.sqrt(2)-5)
run()
