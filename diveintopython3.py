"""
==================
Dive into python 3
==================

Python 2.x is not supported by anyone, effective Jan 2020 (this means now)
Python 3 is in general a fixed version of python.  So it is MORE RESTRICTIVE.
Remember the mantra: Syntax error >> pair programmed >> unit test >> runtime exception >> silent bug

"""
print(__doc__)

from platform import python_version

print('Python', python_version())
print('Hello, World!')

# Eliminar el CRLF 
print("some text,", end="")
print(' print more text on the same line')


# Integer division
print('Python', python_version())
print('3 / 2 =', 3 / 2)
print('3 // 2 =', 3 // 2)   # Integer division
print('3 / 2.0 =', 3 / 2.0)
print('3 // 2.0 =', 3 // 2.0)   # Integer division but the resut is float


# Unicode
print('Python', python_version())
print('strings are now utf-8 \u03BCnico\u0394é!.  日本語でも書くのはできますよ。')

print('Python', python_version(), end="")
print(' has', type(b' bytes for storing data'))

print('and Python', python_version(), end="")
print(' also has', type(bytearray(b'bytearrays')))

x = 100
def val_in_range(x, val):
    return val in range(x)

assert val_in_range(x, 97) == True

# Python 3 has a contains keyword for ranges
assert range(x).__contains__(97) == True
#assert val_in_range(x, 102) == True, "Not in range"

# Excepciones> Python 3 is stricter in syntax, and "AS" is mandatory
print('Python', python_version())
try:
    raise NameError("Naming Error")
except NameError as err:
    print(err, '--> our error message')

# Python 3 only has the next global method
my_generator = (letter for letter in 'abcdefg')

print (next(my_generator))
print (next(my_generator))


# In python 2 the variables form blocks LEAK into the global namespace.  
# This bad practice should not pass any longer in python 3
print('Python', python_version())

# This also includes lambdas
i = 1
print('before: i =', i)
print('comprehension:', [i for i in range(5)])
print('after: i =', i)

# Python 2 allows to do very UGLY things that do not raise exceptions until runtime when it is too late.
print('Python', python_version())

try:
    print("[1, 2] > 'foo' = ", [1, 2] > 'foo')
except TypeError as err:
    print(err)
try:
    print("(1, 2) > 'foo' = ", (1, 2) > 'foo')
except TypeError as err:
    print(err)

try:
    print("[1, 2] > (1, 2) = ", [1, 2] > (1, 2))
except TypeError as err:
    print(err.__class__, end=":")
    print(err)


# Python 3 has generators and iteratable objects (instead of everything being a list)
print('Python', python_version())

print(range(3))
print(type(range(3)))
print(list(range(3)))

# Change in rounding policy.  Round goes to the nearest EVEN number
print(round(15.5))
print(round(16.5))

# Map and lambdas
import numpy as np
import math
input1 = np.asarray([0.34,0.22,0.12,0.323,0.12,-0.83,0.23, 0.3])
input2 = np.asarray([-0.9, 0.3, 0.2, -0.2,0.75,  0.1, 0.2,-0.3])
input3 = np.asarray([0.23,-0.9,   1,  0.3,-0.1, -0.9, 0.4, 0.2])
input = np.concatenate( ([input1],[input2],[input3]))
input = input.T

real = list(map(lambda x,y,z: ((math.sin(x * math.pi + math.pi) + math.cos(y * math.pi + math.pi) + z) / 3), input[:,0],input[:,1],input[:,2]))
print (real)
biggerthanzero = list(filter(lambda z: z>0, input[:,0]))
print(biggerthanzero)
biggerthanzero2 = [i for i in input[:,0] if i>0]
print(biggerthanzero2) # Equivalent

