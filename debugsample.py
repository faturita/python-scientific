'''
Using pdb for debugging on python.

Code and tools provided by https://github.com/tpapastylianou

Debugging:

n: next step
s: step in
h: help

'''

a = 3

print('h: help')
print('ll: check sourrounding code.')
print('n: next line')
print('s: step into')

#import pdb_m; pdb_m.pdb.set_trace()



a = a + 3

def callme(b):
    b = b + 9
    b = b + 2
    return b

c = callme(a)

print (id(a))
print (id(c))

print(f'Value of a:{a}')
print(f'Value of c:{c}')

l = [3,2,1]

def addme(thisisalist):
    thisisalist.append(9)
    return thisisalist

anotherlist = addme(l)

print(id(l), end='')    ;       print(l)
print(id(anotherlist),end='');  print(l)











