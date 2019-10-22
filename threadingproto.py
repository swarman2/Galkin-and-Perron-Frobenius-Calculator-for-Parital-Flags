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
  #Print(mat)
  #print("    { INFO } matrix generation: ",round(time()-t1,3)," secs")
  #print("        ( INFO ) Thm 1 calculations: ",round(t_thm1,3)," secs")
  t2=time()
  real_eigvals = []
  for val in np.linalg.eigvals(mat):
    real_eigvals.append(np.real(val))
  #print("    { INFO } eigenvalue calculation: ",round(time()-t2,3)," secs")
  maxReal = max(real_eigvals)
  return maxReal


def main():
  info = {
  "min_a":4,
  "max_a":4,
  "max_n":40,
  "numThread": multiprocessing.cpu_count()/2
  }
  info["min_n"]=info["max_a"]+1
  if info["max_a"] - info["min_a"] >= info["numThread"]:
    print(" WOULD USE TOO MANY THREADS")
    return
  info["numThread"]=info["numThread"]-(info["max_a"]-info["min_a"])
  Threads=[]
  x = []
  y=[]
  t=time()
  for a2 in range(info["min_a"],info["max_a"]+1):
    x.append([])
    y.append([])
    #print("x[-1]: ",x[-1],"  y:  ",y)
    #print(Threads)
    Threads.append(threading.Thread(target = main_helper,args=(info,a2,x[-1],y[-1])))
    #print(Threads)
    Threads[-1].start()
  for i in range(len(Threads)):
    Threads[i].join()
  #for i in range(len(points[0])):
    #print(points[0][i],", ",points[1][i])
  for i in range(len(x)):
    _x = x[i]
    #print("_x = ",x[i])
    _y = y[i]
    lab = 'a = [1, '+str(i+info["min_a"])+']'
    plt.plot( _x, _y,label=lab)
    plt.plot( _x, _y,'ro')
    temp_x=[]
    temp_y=[]
    scale_factor = max(int(info["max_n"]/5),1)
    for i in range(int(len(_x)/scale_factor)):
      temp_x.append(_x[scale_factor*i])
      temp_y.append(_y[scale_factor*i])
    if info["max_n"]%2==0:
      temp_x.append(_x[-1])
      temp_y.append(_y[-1])
    for i_x, i_y in zip(temp_x, temp_y):
      plt.text(i_x, i_y-(1.1)*.03, '({}, {})'.format(i_x, i_y))
  plt.ylabel('max(real_eigvals)')
  plt.xlabel('n')
  plt.legend(loc='upper left',frameon=False)
  plt.axis([3,info["max_n"]+1,1,2.1])
  print("[ INFO ] Total time: ",round(time()-t,3)," secs")
  plt.show()

def main_helper(info,a2,xarr,yarr):

  a = [1,a2]
  info["a2"]=a2
  #points=[[],[]]
  Threads=[]
  arr = []
  numN=info["max_n"]+1-info["min_n"]
  inc = max(math.floor(numN/info['numThread']),1)
  print("num Threads = ",info['numThread'])
  print("inc = ",inc)
  e=info["min_n"]
  for i in range(0,math.floor(numN/inc)):
    print(i)
    s = e
    e = info["min_n"]+(i+1)*inc
    Threads.append(threading.Thread(target = thread_eign, args =(info,s,e,arr,a)))
    Threads[-1].start()
  #s = e
  #e = info["max_n"]+1
  #arr.append([])
  #Threads.append(threading.Thread(target = thread_eign, args =(info,s,e,arr[-1],a)))
  #Threads[-1].start()
  print("numN = ",numN/inc)
  print("Number of threads = ",len(Threads))
  new_arr=[]
  for i in range(len(Threads)):
    Threads[i].join()
  for i in range(len(arr)):
    new_arr.append(arr[i])
  new_arr.sort(key=lambda new_arr:new_arr[0])
  #xarr=[]
  #yarr=[]
  for i in range(len(new_arr)):
    xarr.append(new_arr[i][0])
    yarr.append(new_arr[i][1])
  #print(mat[7][4])
def thread_eign(info,start,end,arr,a):
  for i in range(start,end):
    alpha=[1,1]
    t=time()
    #print("[ INFO ] solving n = ",i," a = [1, ",info["a2"],"]")
    arr.append([i,round(largest_real_eig(alpha,a,i),5)])
    print("[ INFO ] Finished a = ",a,"  n = ",i)
    #print("n = ",i,":  ",round(time()-t,3)," secs")
main()
#print(multiprocessing.cpu_count()/2)
