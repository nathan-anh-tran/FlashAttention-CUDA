#include <iostream>

struct Matrix {
        int rows;
        int cols;
        float* ptr;
};

Matrix allocateMatrix(int rows, int cols) {
    Matrix m;
    m.rows = rows;
    m.cols = cols;
    m.ptr = new float[rows * cols];
    return m;
}

void freeMatrix(Matrix m) {
    delete[] m.ptr;
}

void addMatrix(const Matrix a, const Matrix b, Matrix res) {
    for (int r = 0; r < a.rows; r += 1) {
        for (int c = 0; c < a.cols; c += 1) {
            int i = r + a.cols * c;
            res.ptr[i] = a.ptr[i] + b.ptr[i];
        }
    }
}

int main() {
    int rows = 10;
    int cols = 20;

    Matrix a = allocateMatrix(10, 20);
    Matrix b = allocateMatrix(10, 20);
    
    for (int i = 0; i < rows * cols; i += 1) {
        a.ptr[i] = i * 3;
        b.ptr[i] = i * 5;
    }
    Matrix c = allocateMatrix(10, 20);

    addMatrix(a, b, c);

    freeMatrix(a);
    freeMatrix(b);
    freeMatrix(c);

    return 0;
}