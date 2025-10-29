# N-Queen Variation: Blocked Cells
def is_safe(q_position, row, col, blocked):
    if blocked and (row, col) in blocked:
        return False

    for i in range(row):
        if q_position[i] == col:
            return False

    for i in range(row):
        if abs(q_position[i] - col) == abs(i - row):
            return False
    return True

def n_Queens(n, blocked, q_position, row, solutions):
    if row == n:
        solutions.append(q_position[:])
        return

    for column in range(n):
        if is_safe(q_position, row, column, blocked):
            q_position[row] = column
            n_Queens(n, blocked, q_position, row+1, solutions)
            q_position[row] = -1

def standard_N_Queens(n, q_position, row, solutions):
    if row == n:
        solutions.append(q_position[:])
        return
    for column in range(n):
        if is_safe(q_position, row, column, None):
            q_position[row] = column
            standard_N_Queens(n, q_position, row+1, solutions)
            q_position[row] = -1

def print_solutions(solutions, n, blocked):
    for index, sol in enumerate(solutions):
        print(f"Solution {index+1}:")
        board = [['.' for _ in range(n)] for _ in range(n)]

        for r, c in blocked:
            board[r][c] = 'X'

        for r in range(n):
            board[r][sol[r]] = 'Q'
        for row in board:
            print(' '.join(row))
        print()

n = 5
blocked = [(0, 2), (2, 3), (3, 0)]
q_position = [-1] * n
solutions = []
n_Queens(n, blocked, q_position, 0, solutions)
print(f"Total solutions for N={n} with blocked cells {blocked}: {len(solutions)}")
print_solutions(solutions, n, blocked)

standard_solutions = []
queen_position_standard = [-1] * n
standard_N_Queens(n, queen_position_standard, 0, standard_solutions)
print(f"Standard N-Queens for N={n} has {len(standard_solutions)} solutions.")
print(f"With blocked cells, the number of solutions is reduced to {len(solutions)}.")
print("The blocked cells restrict possible placements, potentially eliminating some valid configurations that would otherwise be solutions.")
