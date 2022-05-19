# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
print('Hello Word')
print("Howdy!")
print("Shsall we play a game")
print(1+2)
print('1+2')
print(2*3)
#print(100)
print(1/2+1/2)
x=3
x=x+1
x='make'
print(x+' fgh')
print(3*x)
#a=input("enter a value")
#print(a)
#b=int(input())
#print(a)
##d=int(input())
#print("you entered", b,d)
a=int(input('enter a value'))
print('you entered',a)
#f a<=5:
  #  print('a<=5')
  #  if a==5: print('=')
#else:
 #   print('else')
payment=float(input('enter payment'))
i=0
while (payment==0) and i<4:
    payment=float(input('enter not zero, please'))
    i=i+1
    if i==3: print('nadoelo')
print(payment)
for i in range(4,1,-1):
    print(i)
i=2
while i>1:
    print(i)
    i/=2
#i=int(input('enter a number'))
#for j in range(i,0,-1):
#    print(j)
myfile=open("ff",'w')
myfile.write('test line11')
myfile.close()
myfile=open('ff','r')
linefromdata=myfile.readline()
myfile.close()
print(linefromdata)
import graphtools as gt

#Create graph from data. knn - Number of nearest neighbors (including self)
G = gt.Graph(df, use_pygsp=True, knn=5)