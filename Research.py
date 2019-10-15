from helper_functions import *
import numpy as np
import matplotlib.pyplot as plt
from time import time

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

def largest_real_eig(alpha,a,n):
  j = a.index(alpha[1])+1
  perm = S(n,a)
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
  Print(mat)
  print("    { INFO } matrix generation: ",round(time()-t1,3)," secs")
  print("        ( INFO ) Thm 1 calculations: ",round(t_thm1,3)," secs")
  t2=time()
  real_eigvals = []
  for val in np.linalg.eigvals(mat):
    real_eigvals.append(np.real(val))
  print("    { INFO } eigenvalue calculation: ",round(time()-t2,3)," secs")
  maxReal = max(real_eigvals)
  return maxReal
def main():
  alpha=[1,1]
  a = [1,2]
  points=[[],[]]
  max_n = 6
  for i in range(4,max_n+1):
    t=time()
    print("[ INFO ] solving n = ",i)
    points[0].append(i)
    points[1].append(round(largest_real_eig(alpha,a,i),5))
    print("n = ",i,":  ",round(time()-t,3)," secs")
  plt.plot(points[0],points[1])
  plt.plot(points[0],points[1],'ro')
  temp_x=[]
  temp_y=[]
  scale_factor = int(max_n/5)
  for i in range(int(len(points[0])/scale_factor)):
    temp_x.append(points[0][scale_factor*i])
    temp_y.append(points[1][scale_factor*i])
  if max_n%2==0:
    temp_x.append(points[0][-1])
    temp_y.append(points[1][-1])
  for i_x, i_y in zip(temp_x, temp_y):
    plt.text(i_x, i_y-(2.1-min(points[1]))*.025, '({}, {})'.format(i_x, i_y))
  plt.ylabel('max(real_eigvals)')
  plt.xlabel('n')
  plt.axis([3,max_n+1,min(points[1])-(2.1-min(points[1]))*.1,2.1])
  for i in range(len(points[0])):
    print(points[0][i],", ",points[1][i])
  plt.show()


  #print(mat[7][4])
main()
