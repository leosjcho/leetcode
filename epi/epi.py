'''
4.1 Computing Parity
'''

def parity(x):
    result = 0
    while x:
        result ^= x & 1
        x >>= 1
    return result
