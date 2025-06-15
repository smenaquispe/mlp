// Mat.h
#ifndef MAT_H
#define MAT_H

#include <vector>
#include <iostream>
#include <cmath>
#include <random>

using namespace std;

class Mat
{
public:
    vector<vector<double>> data;
    size_t rows, cols;

    Mat(size_t r, size_t c, bool randInit = false) : rows(r), cols(c)
    {
        data.resize(r, vector<double>(c, 0.0));
        if (randInit)
        {
            random_device rd;
            mt19937 gen(rd());
            normal_distribution<double> dist(0.0, sqrt(2.0 / c));
            for (size_t i = 0; i < r; ++i)
                for (size_t j = 0; j < c; ++j)
                    data[i][j] = dist(gen);
        }
    }

    vector<double> &operator[](size_t i)
    {
        return data[i];
    }

    const vector<double> &operator[](size_t i) const
    {
        return data[i];
    }

    // Multiplica la matriz actual por un vector columna (activaciones)
    vector<double> dot(const vector<double> &vec) const
    {
        vector<double> result(rows, 0.0);
        for (size_t i = 0; i < rows; ++i)
            for (size_t j = 0; j < cols; ++j)
                result[i] += data[i][j] * vec[j];
        return result;
    }

    // Transpuesta
    Mat transpose() const
    {
        Mat t(cols, rows);
        for (size_t i = 0; i < rows; ++i)
            for (size_t j = 0; j < cols; ++j)
                t[j][i] = data[i][j];
        return t;
    }

    // Inicialización con ceros
    static Mat zeros(size_t r, size_t c)
    {
        return Mat(r, c, false);
    }

    auto begin() { return data.begin(); }
    auto end() { return data.end(); }
    auto begin() const { return data.begin(); }
    auto end() const { return data.end(); }

    Mat operator+(const Mat &other);
    Mat operator*(double scalar);
    Mat hadamard(const Mat &other); // producto elemento a elemento
};

#endif