from helper_functions import *
import numpy as np
import matplotlib.pyplot as plt
from time import time
import multiprocessing #for threading
import threading #also for threading
"""alph is an ordered pair [l,s] where s is the highest number in alpha, and l is the length of alpha"""
def Thm1(alph,len_u,len_w,sum):
  if len_u + alph[0]==len_w + sum:
    return 1
  else:
    return 0

def get_sub_matrix(alph,d,a,array):
  '''finds the matrix for one d value '''
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
  mat = [ [ None for i in range(len(array)) ] for j in range(len(array)) ]
  n = max(array[0])
  '''
  extendeda=copy.deepcopy(a)
  extendeda.append(n)
  sum_2=0
  k=len(a)
  for i in range(0,k):
    if i == 0:
      sum_2 = sum_2 + (extendeda[i+1])*d[i]
    else:
      sum_2 = sum_2 + (extendeda[i+1] - extendeda[i-1])*d[i]
  '''
  l_tuple = array[-1]
  largest=[]
  for i in l_tuple:
    largest.append(i-1)
  swap_largest = swap_perm(a,d,largest)
  gammad=np.asarray(inv(largest))[swap_largest]
  for i in range(len(array)):
    len_u=l(array[i])
    for j in range(len(array)):
      len_w = l(array[j])
      start=time()
      mat[i][j] = Thm1(alph, len_u, len_w,sum)
      t_thm1+=time()-start
      if mat[i][j]==1:
        if d == [0] * len(d):
          if not len_w == len_u+1:
            mat[i][j]=0
          else:
            a_j = alph[1] # A_j is the last subscript of alpha which is alph[1]
            ident = np.arange(n)
            equal_exists = 0
            for b in range(alph[1]):
              for c in range(alph[1],n):
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
        else:
          w=[]
          for x in array[j]:
            w.append(x)
          if not(l(np.asarray(array[i])[gammad]) == len_u+1-sum and array_equals(np.asarray(array[i])[gammad],w)):
            mat[i][j]=0
  return mat,t_thm1

def get_matrix(alpha,a,n,perm):
  j = a.index(alpha[1])+1
  perm= MergeSort_perm(perm)
  mat = [ [0 for i in range(len(perm)) ] for j in range(len(perm)) ]
  t_thm1=0
  for d in get_d(j,a,alpha[0]):
    temp,t_thm_temp = get_sub_matrix(alpha,d,a,perm)
    t_thm1+=t_thm_temp
    mat = add_mat(mat,temp)
  return mat
def largest_real_eig(mat):
    real_eigvals = []
    for val in np.linalg.eigvals(mat):
      real_eigvals.append(np.real(val))
    return max(real_eigvals)
def main():
  max_n = 10
  max_a = 4
  min_a = 4
  min_n=max_a+1
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
    yarr.append(round(largest_real_eig(get_matrix(alpha,a,i,perm),5)))
    #print("n = ",i,":  ",round(time()-t,3)," secs")
  #xarr[0]=points[0]
  #yarr[0]=points[1]



  #print(mat[7][4])
def run():
  a = [4]
  n= 4
  x_arr=[]
  y_arr=[]
  for n in range(5,7):
    print("n = ",n)
    mat=[]
    perm = S(n,a)
    print("permutations: ",perm)
    for i in a:
      mat.append(get_matrix([1,i],a,n,perm))
    #Print(mat[0])
    #Print(mat[1])
    sum_1 = np.zeros((len(mat[0][0]),len(mat[0][1])))
    sum_2 = 0
    a_w_endpoints = copy.deepcopy(a)
    a_w_endpoints = [0] + a_w_endpoints + [n]
    for i in range(1,len(a)+1):
      sum_1+=(a_w_endpoints[i+1]-a_w_endpoints[i-1])*np.asarray(mat[i-1])
      sum_2+=a_w_endpoints[i]*(a_w_endpoints[i+1]-a_w_endpoints[i])
    real_eigvals = []
    print(sum_1)
    for val in np.linalg.eigvals(sum_1):
      real_eigvals.append(np.real(val))
    E = max(real_eigvals)
    x_arr.append(n)
    y_arr.append(E-sum_2-1)
    #eignen value for s2, n=4 should be 4 sqrt(2)
    print("E: ",E,"  sum_2: ",sum_2)
    print("E - sum_2 -1 = ",E - sum_2 -1)
  plt.plot(x_arr,y_arr)
  plt.show()
def run2():
  a = [1,3]
  n = 8
  x_arr=[]
  y_arr=[]
  mat=[]
  perm = S(n,a)
  print("permutations: ",perm)

  mat1 = get_matrix([1,a[0]],a,n,perm)
  mat2 = get_matrix([1,a[1]],a,n,perm)
  Print(mat1)
  Print(mat2)
  mat3 = (np.matmul(mat1,mat2))
  mat4 = add_mat(mat1,mat2)
  real_eigvals = []
  for val in np.linalg.eigvals(mat1):
    real_eigvals.append(np.real(val))

  eigen1 = max(real_eigvals)
  real_eigvals = []
  for val in np.linalg.eigvals(mat2):
    real_eigvals.append(np.real(val))

  eigen2 = max(real_eigvals)
  real_eigvals = []
  for val in np.linalg.eigvals(mat3):
    real_eigvals.append(np.real(val))

  eigen3 = max(real_eigvals)
  real_eigvals = []
  for val in np.linalg.eigvals(mat4):
    real_eigvals.append(np.real(val))
  eigen4 = max(real_eigvals)

  print(eigen1, "  ",eigen2,"  ",eigen3,"  ",eigen4)
  print(eigen1 + eigen2)
  print(eigen1 * eigen2)

def get_valid_n(input_str):
    valid_n = False
    while not valid_n:
      valid_n = True
      n = input(input_str + ": ")
      try:
        n = int(n)
        if n >16:
          print("Enter n less than 16 ")
          valid_n = False
        if n < 3:
          print("Enter n greater than 2 ")
          valid_n = False
      except:
        print("Enter an intger for n")
        valid_n = False
    return n
def get_valid_a(n, input_str):
    valid_a = False
    while not valid_a:
      valid_a = True
      str_a = input(input_str + ": ")
      str_a.replace(" ","") #get rid of spaces
      a = str_a.split(",")
      for i,split in enumerate(a):
        try:
          split = int(split)
          a[i] = split
        except:
          print("a must be all intergers try again")
          valid_a = False
          break
        if split >15:
          print("a values must be less than 15")
          valid_a = False
        if split >n:
          print("a values must be less than n")
          valid_a = False
        if split <=0:
          print("a values must be greater than 0")
          valid_a = False

      #test if a stritly increasing
      if valid_a:
        for i in range(1,len(a)):
          if a[i]<= a[i-1]:
            print("a must be stritly increasing try again")
            valid_a = False
            break
    return a
def printMenu():
  valid_result = False
  while not valid_result:
    valid_result = True
    print("----------- MENU -----------")
    print("1. Calculate bound as n ranges")
    print("2. P, P inverse thing")
    print("3. Find largest real eignen values as n ranges")
    print("4. Quit")
    result = input()
    if not result in ["1","2","3","4","Q","q"]:
      print("Invalid result")
      valid_result = False
  if result in ["q","Q"]:
    result  = "4"
  return int(result)

def get_valid_alpha(A):
  valid_length = False
  alpha_choices = []
  while not valid_length:
    valid_length = True
    length = input("enter length: ")
    try:
      length = int(length)
    except:
      print("Enter an int for length")
      valid_length = False
    if length <= 0 or length > max(A):
      print("Enter a length in range (",0,", ",max(A),"]")
      valid_length = False
  print()
  valid_alpha = False
  while not valid_alpha:
    valid_alpha = True
    print("Enter a choice for alpha: ")
    counter = 0
    for i,a in enumerate(A):
      if a - length >=0:
        counter = counter + 1
        print(counter,".  [",end='')
        for j in range(a - length+1,a ):
          print("S_",j,", ",end='')
        print("S_",a,"]")
        alpha_choices.append([length,a])
    alpha_choice = input()
    try:
      alpha_choice = int(alpha_choice)
    except:
      print("Enter an integer for alpha")
      valid_alpha = False
    if not alpha_choice in range(1,len(alpha_choices)+1):
      print("Enter a number from the list")
      valid_alpha = False
  return alpha_choices[alpha_choice-1]
def UI():
  import matplotlib.animation as animation
  while True:
    result = printMenu()
    if result == 4:
      return
    enter_more_values = True
    while enter_more_values:
      if result == 1:
        min_n = get_valid_n("Enter min n")
        max_n = get_valid_n("Enter max n")
        a = get_valid_a(min_n, "Enter a (a_1, a_2, ...)")
        alpha = get_valid_alpha(a)
        print("you chose: ")
        print(min_n," <= n <= ",max_n)
        print("a = ",a)
        print("alpha (length, last subscript) = ",alpha)
        xarr = []
        yarr = []
        for n in range(min_n,max_n+1):
          perm = S(n,a)
          xarr.append(n)
          yarr.append(round(largest_real_eig(get_matrix(alpha,a,n,perm)),5))
          print("(",xarr[-1],", ",yarr[-1],")")
        plt.plot(xarr,yarr)
        plt.show()
        enter_more_values = False
      elif result == 2:
        n = get_valid_n("Enter n")
        a = get_valid_a(n,"Enter a")
        perm = S(n,a)
        for i in range(len(a)):
          for j in range(i+1, len(a)):
             print("alphas: S_",a[i]," and S_",a[j])
             print(" A : matrix for S_",a[i])
             print(" B : matrix for S_",a[j])
             A = get_matrix([1,a[i]],a,n,perm)
             B = get_matrix([1,a[j]],a,n,perm)
             print("Largest real eigen values:")
             print("A:    ",largest_real_eig(A),"    B: ",largest_real_eig(B))
             print("A*B:  ",largest_real_eig(np.matmul(A,B)),"   A+B: ",largest_real_eig(add_mat(A,B)))
             print()
UI()
