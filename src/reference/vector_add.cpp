#include <iostream>

void vectorAdd(const float* first, const float* second, float* third, int n) {
    for (int i = 0; i < n; i += 1) {
        third[i] = first[i] + second[i];
    }
}

int main() {
    const int n = 100;
    float* a = new float[n];
    float* b = new float[n];
    float* c = new float[n];

    for (int i = 0; i < n; i += 1) {
        a[i] = i;
        b[i] = i * 2;
    }

    vectorAdd(a, b, c, n);

    for (int i = 0; i < n; i += 1) {
        std::cout << c[i] << '\n';
    }

    delete[] a;
    delete[] b;
    delete[] c;
    return 0;
}