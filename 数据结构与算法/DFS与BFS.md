#### DFS
```cpp
void DFS(int i ,int j, tensor A)
{
    if (i <0 || i > (A.size[0] - 1) || j<0 || j > (A.size[1] - 1)) {
        return;
    }
    A[i][j] = 0;
    DFS(i + 1, j);
    DFS(i - 1, j);
    DFS(i , j + 1);
    DFS(i , j - 1);
}

int main()
{
    int cnt =0;
    for (int i = 0; i<a; ++i)
        for (int j = 0; j < b; ++j) {
            if (A[i][j] == 1) {
                 DFS(i, j);
                 ++cnt;
        }
    }
}
```