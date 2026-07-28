#include <iostream>

int reduceSum(const int* arr, int n) {
    int res = 0;
    for (int i = 0; i < n; i += 1) {
        res += arr[i];
    }
    return res;
}

int main() {
    const int n = 256;
    int* arr = new int[n];
    
    for (int i = 0; i < n; i += 1) {
        arr[i] = i * 2 + 3;
    }

    int result = reduceSum(arr, n);

    std::cout << result << '\n';

    delete[] arr;

    return 0;
}