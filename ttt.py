#TIC-TAC-TOE Game



import os


def whowon(z):
  if z=="x" :
    print(); print()
    print("Winner Player 1")
    print(); print()
  elif z=="o" :
    print(); print()
    print("Winner Player 2")
    print(); print()



def checkwin():
  if arr[1]==arr[2]==arr[3] and arr[1]!=" " : z=arr[1]; whowon(z); return 0

  elif arr[4]==arr[5]==arr[6] and arr[4]!=" " : z=arr[4]; whowon(z); return 0

  elif arr[7]==arr[8]==arr[9] and arr[7]!=" " : z=arr[7]; whowon(z); return 0

  elif arr[1]==arr[4]==arr[7] and arr[1]!=" " : z=arr[1]; whowon(z); return 0

  elif arr[2]==arr[5]==arr[8] and arr[2]!=" " : z=arr[2]; whowon(z); return 0

  elif arr[3]==arr[6]==arr[9] and arr[3]!=" " : z=arr[3]; whowon(z); return 0

  elif arr[1]==arr[5]==arr[9] and arr[1]!=" " : z=arr[1]; whowon(z); return 0

  elif arr[3]==arr[5]==arr[7] and arr[3]!=" " : z=arr[3]; whowon(z); return 0




def indexsyntax():
  print()
  print("====== TIC-TAC-TOE =====")
  print()
  print("       |       |")
  print("   1   |   2   |   3")
  print("       |       |")
  print("_______ _______ _______")
  print("       |       |")
  print("   4   |   5   |   6")
  print("       |       |")
  print("_______ _______ _______")
  print("       |       |")
  print("   7   |   8   |   9")
  print("       |       |")
  print(); print()
  print(); print()



def syntax():
  print()
  print("====== TIC-TAC-TOE =====")
  print()
  print("       |       |")
  print("   "+arr[1]+"   |   "+arr[2]+"   |   "+arr[3])
  print("       |       |")
  print("_______ _______ _______")
  print("       |       |")
  print("   "+arr[4]+"   |   "+arr[5]+"   |   "+arr[6])
  print("       |       |")
  print("_______ _______ _______")
  print("       |       |")
  print("   "+arr[7]+"   |   "+arr[8]+"   |   "+arr[9])
  print("       |       |")




#main

arr=[" "," "," "," "," "," "," "," "," "," "]

p=1; k=1

while 1:
  p = "2" if k%2==0 else "1"
  
  print(); print()
  print("Player "+p+" : ",end="")
  a = int(input())

  if a<1 or a>9 : print("Enter valid index!"); continue

  if arr[a]=="x" or arr[a]=="o" : print("Enter unused index!"); continue

  os.system('clear')

  indexsyntax()

  arr[a]= "o" if k%2==0 else "x"

  syntax()

  if checkwin()==0 : exit()

  if k==9: print("Draw"); break

  k=k+1
  