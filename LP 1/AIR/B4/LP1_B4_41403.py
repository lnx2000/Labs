def printOutput(board, N):
    for i in range(N):
        for j in range(N):
            print (board[i][j], end = " ")
        print()
  
def isSafe(board, row, col):
  
    for i in range(col):
        if board[row][i] == 1:
            return False
  
    # upper left diagonal
    for i, j in zip(range(row, -1, -1), 
                    range(col, -1, -1)):
        if board[i][j] == 1:
            return False
  
    # lower left diagonal
    for i, j in zip(range(row, N, 1), 
                    range(col, -1, -1)):
        if board[i][j] == 1:
            return False
  
    return True
  
def solve2(board, col, N):
      
    # If all queens are placed
    if col >= N:
        return True
    # all rows one by one
    for i in range(N):
  
        if isSafe(board, i, col):
              
            board[i][col] = 1
  
            if solve2(board, col + 1, N) == True:
                return True
            board[i][col] = 0
  
    return False
  
def solve(N):
    board = [[0 for _ in range(N)] for __ in range(N)]
    print("Before")
    printOutput(board, N)
    if solve2(board, 0, N) == False:
        print ("Solution does not exist")
        return False
    print("After")
    printOutput(board, N)
    return True

if __name__ == '__main__':
	N = int(input())
	solve(N)
