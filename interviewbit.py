# @param X : list of integers
# @param Y : list of integers
# Points are represented by (X[i], Y[i])
# @return an integer
def coverPoints(self, X, Y):
    if len(X) < 2:
        return 0
    steps = 0
    for i in xrange(len(X)-1):
        dx = abs(X[i+1] - X[i])
        dy = abs(Y[i+1] - Y[i])
        steps += min(dx, dy)
        steps += max(dx, dy) - min(dx, dy)
    return steps
