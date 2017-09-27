'''
Tricks to remember:
- x & (x-1) drops lowest set bit
'''

'''
4.1 Computing Parity
'''

# O(n) time complexity
def parity1(x):
    starting = x
    result = 0
    count = 0
    while x:
        result ^= x & 1
        x >>= 1
        count += 1
    print("parity1({}) # of iterations: {}".format(starting, count))
    return result

'''
x & (x-1) drops lowest set bit
this can lower time complexity in avg case
Using parity1, input 0xb10000000 would require 8 iterations of the while
loop.
If we dropped the lowest bit on every iteration, this input
would require just one iteration to calculate the parity.
'''

def parity2(x):
    starting = x
    result = 0
    count = 0
    while x:
        x &= x - 1
        result ^= 1
        count += 1
    print("parity2({}) # of iterations: {}".format(starting, count))
    return result
