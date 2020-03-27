"""
==================
One   Pass  Python
==================

Esta guía está basada en este excelente tutorial: http://cs231n.github.io/python-numpy-tutorial/

Python 101: la indentación del código determina los bloques.

"""
print(__doc__)

# Primero funciones !  Sirven para modularizar el código.

# In[1]: Estos comentarios habilitan el VSC para que opere como un notebook de jupyter.
def f(x=0):
    print('El valor es %2d' % x)
    return 4

variable = f()

# In[1]:
print ('Existen muchas situaciones donde es necesario formatear texto.  El 3 indica la cantidad de digitos a utlizar y d que son enteros.')
a=3
print ('AT'+'{:3d}'.format(a))
print ('AT %3d' % a)
print ('Tuple %3d %3d' % (2,3))

# In[1]:
x = 3
print(type(x)) # Prints "<class 'int'>"
print(x)       # Prints "3"
print(x + 1)   # Addition; prints "4"
print(x - 1)   # Subtraction; prints "2"
print(x * 2)   # Multiplication; prints "6"
print(x ** 2)  # Exponentiation; prints "9"
x += 1
print(x)  # Prints "4"
x *= 2
print(x)  # Prints "8"
y = 2.5
print(type(y)) # Prints "<class 'float'>"
print(y, y + 1, y * 2, y ** 2) # Prints "2.5 3.5 5.0 6.25"


# In[1]:
t = True
f = False
print(type(t)) # Prints "<class 'bool'>"
print(t and f) # Logical AND; prints "False"
print(t or f)  # Logical OR; prints "True"
print(not t)   # Logical NOT; prints "False"
print(t != f)  # Logical XOR; prints "True"

# In[1]:
print('Manejo de cadenas en python....')
hello = 'hello'    # String literals can use single quotes
world = "world"    # or double quotes; it does not matter.
print(hello)       # Prints "hello"
print(len(hello))  # String length; prints "5"
hw = hello + ' ' + world  # String concatenation
print(hw)          # prints "hello world"
hw12 = '%s %s %d' % (hello, world, 12)  # sprintf style string formatting
print(hw12)        # prints "hello world 12"
print (hw[0:1])    # Strings allow some slicing
print (hw[0:-1])

# In[1]:
s = "hello"
print(s.capitalize())  # Capitalize a string; prints "Hello"
print(s.upper())       # Convert a string to uppercase; prints "HELLO"
print(s.rjust(7))      # Right-justify a string, padding with spaces; prints "  hello"
print(s.center(7))     # Center a string, padding with spaces; prints " hello "
print(s.replace('l', '(ell)'))  # Replace all instances of one substring with another;
                                # prints "he(ell)(ell)o"
print('  world '.strip())  # Strip leading and trailing whitespace; prints "world"

#https://docs.scipy.org/doc/numpy-1.12.0/reference/routines.sort.html

# In[1]:
print('Python contain four built-in data structures:lists, containers, sets, and tuples')
print('Lists can be handled as dynamic vectors with heterogeneous elements.')
xs = [3, 1, 2]    # Create a list
print(xs, xs[2])  # Prints "[3, 1, 2] 2"
print(xs[-1])     # Negative indices count from the end of the list; prints "2"
xs[2] = 'foo'     # Lists can contain elements of different types
print(xs)         # Prints "[3, 1, 'foo']"
xs.append('bar')  # Add a new element to the end of the list
print(xs)         # Prints "[3, 1, 'foo', 'bar']"
x = xs.pop()      # Remove and return the last element of the list
print(x, xs)      # Prints "bar [3, 1, 'foo']"

# In[1]:
print('Slicing is crucial in multivariate analysis with python')

nums = list(range(5))     # range is a built-in function that creates an object that represent a list of integer.  List convert that to a python list.
print(nums)               # Prints "[0, 1, 2, 3, 4]"
print(nums[2:4])          # Get a slice from index 2 to 4 (exclusive); prints "[2, 3]"
print(nums[2:])           # Get a slice from index 2 to the end; prints "[2, 3, 4]"
print(nums[:2])           # Get a slice from the start to index 2 (exclusive); prints "[0, 1]"
print(nums[:])            # Get a slice of the whole list; prints "[0, 1, 2, 3, 4]"
print(nums[:-1])          # Slice indices can be negative; prints "[0, 1, 2, 3]"
nums[2:4] = [8, 9]        # Assign a new sublist to a slice
print(nums)               # Prints "[0, 1, 8, 9, 4]"

# In[1]:
print('Looping over elements in a list...')
animals = ['cat', 'dog', 'monkey']
for animal in animals:
    print(animal)
# Prints "cat", "dog", "monkey", each on its own line.

# In[1]:
animals = ['cat', 'dog', 'monkey']
for idx, animal in enumerate(animals):
    print('#%d: %s' % (idx + 1, animal))
# Prints "#1: cat", "#2: dog", "#3: monkey", each on its own line

# In[1]:
print('List comprehensions are lambdas over lists.')
nums = [0, 1, 2, 3, 4]
squares = []
for x in nums:
    squares.append(x ** 2)
print(squares)   # Prints [0, 1, 4, 9, 16]

nums = [0, 1, 2, 3, 4]
squares = [x ** 2 for x in nums]
print(squares)   # Prints [0, 1, 4, 9, 16]

# In[1]:
print('They can also contain conditions...')
nums = [0, 1, 2, 3, 4]
even_squares = [x ** 2 for x in nums if x % 2 == 0]
print(even_squares)  # Prints "[0, 4, 16]"

# In[1]:
print('Dictionaries are hashtables.')
d = {'cat': 'cute', 'dog': 'furry'}  # Create a new dictionary with some data
print(d['cat'])       # Get an entry from a dictionary; prints "cute"
print('cat' in d)     # Check if a dictionary has a given key; prints "True"
d['fish'] = 'wet'     # Set an entry in a dictionary
print(d['fish'])      # Prints "wet"
# print(d['monkey'])  # KeyError: 'monkey' not a key of d
print(d.get('monkey', 'N/A'))  # Get an element with a default; prints "N/A"
print(d.get('fish', 'N/A'))    # Get an element with a default; prints "wet"
del d['fish']         # Remove an element from a dictionary
print(d.get('fish', 'N/A')) # "fish" is no longer a key; prints "N/A"

# In[1]:
d = {'person': 2, 'cat': 4, 'spider': 8}
for animal in d:
    legs = d[animal]
    print('A %s has %d legs' % (animal, legs))
# Prints "A person has 2 legs", "A cat has 4 legs", "A spider has 8 legs"

# In[1]:
d = {'person': 2, 'cat': 4, 'spider': 8}
for animal, legs in d.items():
    print('A %s has %d legs' % (animal, legs))
# Prints "A person has 2 legs", "A cat has 4 legs", "A spider has 8 legs"

# In[1]:
# Build a dictionary based on a list and lambda function
nums = [0, 1, 2, 3, 4]
even_num_to_square = {x: x ** 2 for x in nums if x % 2 == 0}
print(even_num_to_square)  # Prints "{0: 0, 2: 4, 4: 16}"

# In[1]:
# Sets are lists without order
print('Sets are unordered list of elements.')
animals = {'cat', 'dog'}
print('cat' in animals)   # Check if an element is in a set; prints "True"
print('fish' in animals)  # prints "False"
animals.add('fish')       # Add an element to a set
print('fish' in animals)  # Prints "True"
print(len(animals))       # Number of elements in a set; prints "3"
animals.add('cat')        # Adding an element that is already in the set does nothing
print(len(animals))       # Prints "3"
animals.remove('cat')     # Remove an element from a set
print(len(animals))       # Prints "2"


# In[1]:
# Creates a set from a list of numbers generated by a lambda.
from math import sqrt
nums = {int(sqrt(x)) for x in range(30)}
print(nums)  # Prints "{0, 1, 2, 3, 4, 5}"


# In[1]:
print('Tuples are like un-indexed lists and they can be used as element key')
d = {(x, x + 1): x for x in range(10)}  # Create a dictionary with tuple keys
t = (5, 6)        # Create a tuple
print(type(t))    # Prints "<class 'tuple'>"
print(d)
print(d[t])       # Prints "5"
print(d[(1, 2)])  # Prints "1"


# In[1]:
# More Functions
def helloguys(name='Joe Doe', loud=False):
    if loud:
        print('HELLO, %s!' % name.upper())
    else:
        print('Hello, %s' % name)

helloguys('Bob') # Prints "Hello, Bob"
helloguys('Jenny', loud=True)  # Prints "HELLO, FRED!"

# In[1]:
# Objects
class Greeter(object):

    # Constructor
    def __init__(self, name):
        self.name = name  # Create an instance variable

    # Instance method
    def greet(self, loud=False):
        if loud:
            print('HELLO, %s!' % self.name.upper())
        else:
            print('Hello, %s' % self.name)

g = Greeter('Fred')  # Construct an instance of the Greeter class
g.greet()            # Call an instance method; prints "Hello, Fred"
g.greet(loud=True)   # Call an instance method; prints "HELLO, FRED!"

# In[1]:
# Check python system.
import sys

print ('Input line parameters')
print (sys.argv)

print ('Byte order:'+str(sys.byteorder))
print (sys.exec_prefix)
print (sys.executable)
print (sys.path)
print (sys.version_info)
print (sys.platform)

print (sys.argv[0])
print ('Modules =======')
print (sys.modules)

# In[1]:
# Sets check for existence so they can use to identify duplicates
def unique(elements):
    if len(elements)==len(set(elements)):
        print("All elements are unique")
    else:
        print("List has duplicates")
unique([1,2,3,4,5]) # All elements are unique
unique([1,1,2,3,4,5])  # No


# In[1]:
# Getting the histogram of elements.
from collections import Counter
elements = [1, 2, 3, 2, 4, 3, 2, 3]
count = Counter(elements)
print(count) # {2: 3, 3: 3, 1: 1, 4: 1}

# In[1]:
# Getting the most frequent element
def most_frequent(elements):
    return max(set(elements), key = elements.count)
numbers = [1, 2, 3, 2, 4, 3, 1, 3]
most_frequent(numbers) # 3

# In[1]:
# Recursion
def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)

print ('Quicksort es un algoritmo bastante eficiente para ordenar elementos.')
print ([3,6,8,10,1,2,1])
print(quicksort([3,6,8,10,1,2,1]))

# %%
