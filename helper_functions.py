"""Find all comibinations of S_i(split[1]...split[length(split)])"""
from itertools import permutations
from time import time
import math
import numpy as np
import copy
import multiprocessing #for threading
import threading #also for threading
alpha = [1,2]
d = [1]
a = [2]
#n=5
"""

"""

def valid_perm(perm,a_temp):
  a = copy.deepcopy(a_temp)
  a = [0]+a+[max(perm)]
  array_of_perm = []
  for i in range(1,len(a_temp)+2):
    array_of_perm.append(perm[a[i-1]:a[i]])
  for perm in array_of_perm:
    for i in range(1,len(perm)):
      if perm[i]<perm[i-1]:
        return False
  return True

len_dict={}
#thread_count = multiprocessing.cpu_count()
def S(maxNum,split_temp = []):
  permutations = []
  f = open("D:\\ResearchFall19\\n13.bin","rb");
  #x = f.readline()
  #x = x.split(" ")
  if maxNum < 13:
    x = list(f.read(13))
    while x[maxNum] == maxNum+1:
      perm=x[:maxNum]
      if valid_perm(perm, split_temp):
        permutations.append(perm)
      x = list(f.read(13))
    #print(permutations)
    return permutations
  if maxNum == 13:
    counter = 0
    perm = list(f.read(13))
    while perm:
      counter += 1
      if counter%10000 == 0:
        print(counter)
      if valid_perm(perm, split_temp):
        permutations.append(perm)
      perm = list(f.read(13))
    print(permutations)
    return
  if maxNum >13:

    perm=list(f.read(13))
    while perm:
      new_perm = []
      for i in range(14,maxNum+1):
        new_perm.append([perm[j:] + [i] + mylist[:j] for j in xrange(len(perm),-1,-1)])
        permutations = new_perm
        new_perm=[]
      for perm in permutations:
          if valid_perm(perm, split_temp):
            permutations.append(perm)
      perm = list(f.read(13))
    return

  for i in range(len(perm)):
    perm[i]=int(perm[i])
  print(perm)
  t=time()
  split = copy.deepcopy(split_temp)
  split.append(maxNum)
  #t = time.time()
  if len(split) == 0:
    print("error invalid vector a")
  #print(split_temp[-1]+2)
  l = list(permutations(range(1,split_temp[-1]+2)))
  l=[list(i)for i in l]
  #print("l = ",l)
  #print(len(l))
 # print(l)
  k=0
  for k in range(len(l)-1,-1,-1):
 # while k < len(l):
    array = []
    prev = 0
    for i in range(len(split)-1):
      array.append(l[k][prev:split[i]])
      prev = split[i]
    array.append(l[k][prev:])
    """array [i] is now the i'th split"""
    i=0
    while i < len(array):
      j=0
      while j < len(array[i])-1:
        if array[i][j]>array[i][j+1]:
          l.pop(k)
          j=len(array[i])
        j=j+1
      if j == len(array[i])+1:
        i=len(array)
      i=i+1
  #print("l = ",l)
  new_l= copy.deepcopy(l)
  for k in range(split[-2]+1,maxNum):
    #print("test")
    #print("l = ",l)
    #reverse_ident = []
    #for i in range(k,0,-1):
    #  reverse_ident.append(i)
    #l.append(reverse_ident)
    #new_l=[]
    #for i in range(len(split)):
    for i in range(len(l)):
      for j in range(len(l)):
        #l[j].insert(split[i]-1,k+1)
        l[j].insert(i,k+1)
        copyl=copy.deepcopy(l[j])
        new_perm = fix_perm(split[0:-1],copyl)
        if not new_perm in new_l:
          new_l.append(new_perm)#-1 bc split is indexed from 1
        l[j].remove(k+1)
    #print()
    #print(new_l)
    #print()
    l=copy.deepcopy(new_l)
  #print(time.time()-t)
  #print()

  #print(len(l))
  #print("new_l = ",  new_l)
  #print("    { INFO } permutations: ",round(time()-t,3)," secs")
  #print("new_l = ",new_l)
  return new_l
   # print(l)
def l(w):
  #print(len_dict.keys())
  if tuple(w) in len_dict.keys():
    return len_dict[tuple(w)]
  copyw = copy.deepcopy(w)
  if min(w) == 0:
    for i in range(len(copyw)):
      copyw[i]+=1
  #find the length of a permution
  length=0
  n = max(copyw)
  for i in range(1,n):
    for j in range(i+1,n+1):
      if copyw[i-1]>copyw[j-1]:
        length+=1
  len_dict[tuple(w)]=length
  return length



"""returns true or false"""
def Piere(d,j):
  """
  max_index = 0
  for i in range(1,len(d)):
    if d[i]>d[max_index]:
      max_index=i
          """
  max_index=j
  for i in range(max_index-1):
    if not i == len(d)-1:
      if d[i]>d[i+1]:
        return 0
  for i in range(max_index+1, len(d)-1):
    if d[i]<d[i+1]:
      return 0
  return 1

def MergeSort_perm(array):
  for i in range(len(array)):
    len_dict[tuple(array[i])]=l(array[i])
  return MergeSort_perm_helper(array)
'''Takes in an array of permutations'''
def MergeSort_perm_helper(array):
  if len(array)<=1:
    return array
  m = math.floor(len(array)/2)
  A_1 = MergeSort_perm_helper(array[0:m])
  A_2 = MergeSort_perm_helper(array[m:len(array)])
  return Merge(A_1, A_2)
def Merge(A_1, A_2):
  A = []
  i=0
  j=0
  for k in range(len(A_1)+len(A_2)):
    if i >= len(A_1) or j >= len(A_2):
      break
    if l(A_1[i])==l(A_2[j]):
      for r in range(len(A_1[i])):
        if A_1[i][r] < A_2[j][r]:
          A.append(A_1[i])
          i = i+1
          break
        elif A_1[i][r] > A_2[j][r]:
          A.append(A_2[j])
          j=j+1
          break
    elif l(A_1[i])<l(A_2[j]):
      A.append(A_1[i])
      i = i+1
    else:
      A.append(A_2[j])
      j=j+1
  if i >= len(A_1):
    while j < len(A_2):
      A.append(A_2[j])
      j=j+1
  else:
    while i < len(A_1):
      A.append(A_1[i])
      i = i+1
  return A

def add_mat(A,B):
  row = len(A)
  col = len(A[0])
  mat = [ [ None for i in range(col) ] for j in range(row) ]
  if not row == len(B) or not col ==len(B[0]):
    print("mismatch sizes")
  else:
    for i in range (len(A)):
      for j in range(len(A[0])):
        mat[i][j]=A[i][j]+B[i][j]
    return mat

'''To print a matrix'''
def Print(matrix):
  print("    ",end = ' ')
  for j in range(len(matrix[0])):
    print("%2d"%j,end = '  ')
  print()
  for i in range(len(matrix)):
    print("%2d"%i,": ",end=' ')
    for j in range(len(matrix[0])):
      print(matrix[i][j],end = '   ')
    print()
def get_d(j,a,l_alpha):
  decreasing = []
  increasing = []
  t_increase=[]
  max_d = l_alpha
  #print("j=",j)
  d = [0 for i in range(len(a)-j)]
  decreasing.append(copy.copy(d))
  furth_index = 0
  while d != [max_d for i in range(len(d))]:
  #for s in range(10):
    d[0]=d[0]+1
    for r in range(len(d)):
      if d[r]==max_d +1:
        d[r] = 0
        if d[r+1]==0:
          furth_index = furth_index+1
        else:
          furth_index = r+1
        d[r+1]=d[r+1]+1
        for i in range(furth_index):
          d[i]=d[furth_index]
    decreasing.append(copy.copy(d))
  d = [0 for i in range (j-1)]
  furth_index=0
  t_increase.append(copy.copy([0 for i in range(len(d))]))
  while d != [max_d  for i in range(len(d))]:
  #for s in range(10):
    d[0]=d[0]+1
    for r in range(len(d)):
      if d[r]==max_d +1:
        d[r] = 0
        if d[r+1]==0:
          furth_index = furth_index+1
        else:
          furth_index = r+1
        d[r+1]=d[r+1]+1
        for i in range(furth_index):
          d[i]=d[furth_index]
    t_increase.append(copy.copy(d))
  for x in t_increase:
    increasing.append(copy.copy(x[::-1]))
  #print (decreasing)
  #print(increasing)
  all_piere = []
  for x in increasing:
    for y in decreasing:
      if not len(y)==0 and not len(x)==0:
        for w in range((max(x[len(x)-1],y[len(y)-1])), max_d +1):
          z = x[:] + [w] + y[:]
          all_piere.append(copy.copy(z))
      elif len(y)==0 and not len(x)==0:
        for w in range(x[len(x)-1],max_d +1):
           z = x[:] + [w]
           all_piere.append(copy.copy(z))
      elif not len(y)==0 and len(x)==0:
        for w in range(y[len(y)-1],max_d +1):
           z =  [w] + y[:]
           all_piere.append(copy.copy(z))
      else:
        for w in range(0,max_d +1):
          all_piere.append([w])
        #print(z)
  return all_piere

def T(d,a,i,p,n):
  temp_a = copy.deepcopy(a)
  temp_a.append(n)
  temp_a.insert(0,0)
  if temp_a[i] - d[i-1] < p and p <= temp_a[i]:
    p = p+ temp_a[i+1]-temp_a[i]
  elif temp_a[i] < p and p <= temp_a[i+1]:
    p= p-d[i-1]
  else:
    p=p
  return p
def gamma(d,a,n,p):
  for i in range(len(a),0,-1):
    p = T(d,a,i,p,n)
  #print()
  return p

def w_a(a,n,j):
  i=0
  temp_a = copy.deepcopy(a)
  temp_a.append(n)
  temp_a.insert(0,0)
  while temp_a[i]<j:
    i=i+1
  i = i-1

  return temp_a[i]+temp_a[i+1]+1-j


def fix_perm(a,perm_og):
  #print()
  #print("a = ",a," perm = ",perm_og)

  perm=copy.deepcopy(perm_og)
  #print()
  #print("  perm[:a[0]] = ",  perm[:a[0]])
  perm[:a[0]]=insertionSort(perm[:a[0]])
  #print("  perm[:a[0]] = ",  perm[:a[0]])
  for r in range(0,len(a)-1):
    #print()
    #print(  "perm[",a,"[",r,"]:a[",r+1,"]] = ",perm[a[r]:a[r+1]])
    perm[a[r]:a[r+1]]=insertionSort(perm[a[r]:a[r+1]])
    #print(  "perm[",a,"[",r,"]:a[",r+1,"]] = ",perm[a[r]:a[r+1]])
  #print()
  #print("perm[a[-1]:] = ",perm[a[-1]:])
  perm[a[-1]:]=insertionSort(perm[a[-1]:])
  #print("perm[a[-1]:] = ",perm[a[-1]:])
  #print("sorted perm = ",perm)
  return perm
def insertionSort(arr):
  #print(arr)
  for i in range(1, len(arr)):
      k = arr[i]
      j = i-1
      while j >=0 and k < arr[j] :
              arr[j+1] = arr[j]
              j -= 1
      arr[j+1] = k
  return arr
def swap_perm(a,d,large):
  largest = copy.deepcopy(large)
  index_first_one = 0
  found1=0 #bool for if first one is found
  index_last_one = len(d)-1
  found2=0 # bool for if last one is found
  for i in range(len(d)):
    if d[i]==1 and found1 == 0:
      index_first_one =i
      found1=1
    if d[len(d)-i-1]==1 and found2 == 0:
      index_last_one = len(d)-1-i
      found2=1
  #print("index_first_one: ",index_first_one,"  index_last_one: ",index_last_one)
  small_swap = a[index_first_one]-1 #the index of the smaller value to swap
  #the index of the larger value to swap is the end - the amount of elms in the last section
  large_swap = a[index_last_one]
  """
  for i in range(len(a)-1,index_last_one,-1):
    print(large_swap," -= ",a[i])
    large_swap -= a[i-1] #add 1 (- -1)to get the number to the right of the partition not the left
  """
  #print(largest)
  #print(largest[small_swap],"   ",largest[large_swap])
  temp = largest[small_swap]
  largest[small_swap]=largest[large_swap]
  largest[large_swap]=temp
  #print("t largest = ",largest)
  largest = fix_perm(a,largest)
  #print("t largest = ",largest)
  return largest
"""
print("Possible permutations:")
Print(S(4,a))
x = (make_matrix(alpha,[0],a,S(4,a)))
y = (make_matrix(alpha,[1],a,S(4,a)))
Print(add_mat(x,y))
print(Thm1(alpha,[3,4,1,2],[1,3,2,4],d,a))
#Print(make_matrix(alpha,d,a,S(4,a)))
"""
def inv(perm):
  #print("perm = ",perm)
  inverse = [0] * len(perm)
  for i, p in enumerate(perm):
      inverse[p] = i
  return inverse
def array_equals(arr1,arr2):
  if len(arr1)!= len(arr2):
    return 0
  else:
    for i in range(len(arr1)):
      if arr1[i]!=arr2[i]:
        return 0
  return 1
#get_d(3,[2,4,6,7])
#print(fix_perm([1,2],[1,6,3,2,4,5]))
#print("d = [1,0]")
#print(swap_perm([1,3],[1,0],[3,1,2,0]))
