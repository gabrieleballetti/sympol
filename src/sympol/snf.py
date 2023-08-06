from sympy import Matrix

# Due to sympy having a compact implementation of the Smith normal form
# algorithm, we reimplement it here in a version which returns the
# transformation matrices S and T other than the diagonal matrix D.
# This is forked from the code of https://github.com/corbinmcneill/SNF
# (GPLv3), and adapted to sympy matrices. The original code was written
# for a more general setting, this is specialized to the case of integers.
# Might be slow, but probably not the bottleneck for us.


def extended_gcd(a, b):
    x0 = 1
    x1 = 0
    y0 = 0
    y1 = 1
    while b != 0:
        tempa = a
        tempb = b
        q = tempa // tempb
        a = tempb
        b = tempa % tempb
        tempx0 = x0
        x0 = x1
        x1 = tempx0 - q * x0
        tempy0 = y0
        y0 = y1
        y1 = tempy0 - q * y0
    return [a, x0, y0]


# Perform a "column-swap". Here we modify the matrix J by swapping the
# columns of index i and j. We adjust the matrix T to make sure the
# overall relation of S*A*T = J continues to hold.
def cSwap(i, j, J, T):
    # perform the column swap to J
    for k in range(J.rows):
        temp = J[k, i]
        J[k, i] = J[k, j]
        J[k, j] = temp

    # adjust the T matrix
    adjustment = Matrix.eye(T.rows)
    adjustment[i, i] = 0
    adjustment[j, j] = 0
    adjustment[i, j] = 1
    adjustment[j, i] = 1
    T = T * adjustment

    return J, T


# Perform a "column-wise linear combination" operation. Here we set the k
# column of the matrix J to be a * the k column plus b times the j
# column. We update the T matrix to ensure the relationship S*A*T = J
# continues to hold.
def cLC(k, i, j, a, b, J, T, gcd=None):
    # perform the linear column application to J
    if gcd is None or a in [1, -1]:
        c = 0
        d = 1
    else:
        c = -J[k, j] // gcd  # pragma: no cover
        d = J[k, i] // gcd  # pragma: no cover

    for k in range(J.rows):
        temp = J[k, i]
        J[k, i] = a * J[k, i] + b * J[k, j]
        J[k, j] = c * temp + d * J[k, j]

    # adjust the T matrix
    adjustment = Matrix.eye(T.rows)
    adjustment[i, i] = a
    if i != j:
        adjustment[j, i] = b
        adjustment[i, j] = c
        adjustment[j, j] = d

    T = T * adjustment

    return J, T


# Perform a "row-swap". Here we modify the matrix J by swapping the rows
# of index i and j. We adjust the matrix S to make sure the overall
# relation of S*A*T = J continues to hold.
def rSwap(i, j, J, S):
    # perform the row swap to J
    for k in range(J.cols):
        temp = J[i, k]
        J[i, k] = J[j, k]
        J[j, k] = temp

    # adjust the S matrix
    adjustment = Matrix.eye(S.rows)
    adjustment[i, j] = 1
    adjustment[j, i] = 1
    adjustment[i, i] = 0
    adjustment[j, j] = 0
    S = adjustment * S

    return J, S


# Perform a "row-wise linear combination" operation. Here we set the k
# row of the matrix J to be a * the i row plus b times the j row. We
# update the S matrix to ensure the relationship S*A*T = J continues to
# hold.
def rLC(k, i, j, a, b, J, S, gcd=None):
    if gcd is None or a in [1, -1]:
        c = 0
        d = 1
    else:
        c = -J[j, k] // gcd  # pragma: no cover
        d = J[i, k] // gcd  # pragma: no cover

    # perform the linear column application to J
    for k in range(J.cols):
        temp = J[i, k]
        J[i, k] = a * J[i, k] + b * J[j, k]
        J[j, k] = c * temp + d * J[j, k]

    # adjust the S matrix
    adjustment = Matrix.eye(S.rows)
    adjustment[i, i] = a
    if i != j:
        adjustment[i, j] = b
        adjustment[j, i] = c
        adjustment[j, j] = d
    S = adjustment * S

    return J, S


def smith_normal_form(A: Matrix):
    """
    Returns the Smith normal form of the matrix A, i.e. the diagonal matrix D
    such that D = U * A * V, where U and V are integer unimodular matrices.
    """
    J = A.copy()
    S = Matrix.eye(A.rows)
    T = Matrix.eye(A.cols)

    # The heart of snf starts here
    for i in range(min(J.rows, J.cols)):
        # if the top-left element of the subarray is 0 we need to
        # perform row/column swaps to move in a different value
        if J[i, i] == 0:
            # we search for a nonzero entry in the submatrix to replace the
            # zero element with.
            foundReplacement = False
            j = i
            k = i
            for j in range(i, J.rows):
                if foundReplacement:
                    break
                for k in range(i, J.cols):
                    if J[j, k] != 0:
                        foundReplacement = True
                        break

            # if there are no non-zero values left to swap in, the
            # algorithm is complete
            if not foundReplacement:
                break

            # perform the swap
            else:
                J, S = rSwap(i, j, J, S)
                J, T = cSwap(i, k, J, T)

        # now we should not have a zero in the top-left position
        # of the submatrix

        # make the top-left submatrix element be the gcd of all the
        # elements in the same row or the same column
        doneIteration = False
        while not doneIteration:
            if J[i, i] in [1, -1]:
                break
            doneIteration = True
            for j in range(i + 1, J.rows):
                gcd, x, y = extended_gcd(J[i, i], J[j, i])
                if J[i, i] in [gcd, -gcd]:
                    pass
                elif J[j, i] in [gcd, -gcd]:
                    J, S = rSwap(i, j, J, S)
                    doneIteration = False
                else:
                    J, S = rLC(i, i, j, x, y, J, S, gcd)
                    doneIteration = False
            for j in range(i + 1, J.cols):
                gcd, x, y = extended_gcd(J[i, i], J[i, j])
                if J[i, i] in [gcd, -gcd]:
                    pass
                elif J[i, j] in [gcd, -gcd]:
                    J, T = cSwap(i, j, J, T)  # pragma: no cover
                    doneIteration = False  # pragma: no cover
                else:
                    J, T = cLC(i, i, j, x, y, J, T, gcd)
                    doneIteration = False

        # use the gcd to make all elements int the ith row and the ith
        # column zero by row and column linear combinations
        doneZeroing = False
        while not doneZeroing:
            doneZeroing = True
            for j in range(i + 1, J.rows):
                if J[j, i] != 0:
                    J, S = rLC(i, j, i, 1, -J[j, i] // J[i, i], J, S)
                    if J[j, i] != 0:
                        doneZeroing = False  # pragma: no cover
            for j in range(i + 1, J.cols):
                if J[i, j] != 0:
                    J, T = cLC(i, j, i, 1, -J[i, j] // J[i, i], J, T)
                    if J[i, j] != 0:
                        doneZeroing = False  # pragma: no cover

    # At this point J is diagonalized. Me simply need to make sure
    # that every diagonal element divides the element after it
    for i in range(min(J.cols, J.rows) - 1):
        # If the next diagonal element is 0, then all following diagonal
        # elements # will be 0. Therefore J is in Smith normal form
        # and we return
        if J[i + 1, i + 1] == 0:
            return J, S, T
        gcd, x, y = extended_gcd(J[i, i], J[i + 1, i + 1])

        # if the ith diagonal element is already the gcd of of the the
        # ith and the (i+1)th diagonal elements, they are correct and
        # we can advance. If they are not we should change the ith
        # element to be the gcd by row operations while maintaining
        # that J is diagonal
        if gcd == J[i + 1, i + 1]:
            J, T = cSwap(i, i + 1, J, T)
            J, S = rSwap(i, i + 1, J, S)

        elif gcd != J[i, i]:
            J, S = rLC(i, i, i + 1, 1, 1, J, S)
            J, T = cLC(i, i, i + 1, x, y, J, T, gcd)
            J, T = cLC(i, i + 1, i, 1, -J[i, i + 1] // J[i, i], J, T)
            J, S = rLC(i, i + 1, i, 1, -J[i + 1, i] // J[i, i], J, S)

    # The original algorithm was in a more general setting where solutions up to a
    # unit are enough. In our case we might end up with possibly negative diagonal
    # elements. We force them to be positive, and update S and T accordingly.
    for i in range(min(J.cols, J.rows)):
        if J[i, i] < 0:
            J, S = rLC(i, i, i, -1, 0, J, S)

    return J, S, T
