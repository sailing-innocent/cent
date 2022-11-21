#pragma once

#include <vector>
#include <cmath>
#include <iostream>

class Polynomial {
public:
    Polynomial() = default;
    explicit Polynomial(int _order) {
        mOrder = _order;
        mParams.resize(_order+1);
        for (auto i = 0; i <= mOrder; i++) {
            mParams[i] = 0.0;
        }
    }
    double forward(double x) {
        double res = 0.0;
        for (auto i = 0; i <= mOrder; i++) {
            res += std::pow(x,i) * mParams[i];
        }
        return res;
    }
    float operator()(float x) {
        return static_cast<float>(forward(static_cast<double>(x)));
    }
    bool setParam(int _order, double _param) {
        mParams[_order] = _param;
        return true;
    }
    friend std::ostream& operator<<(std::ostream& os, Polynomial poly) {
        os << poly.params()[0];
        for (auto i = 1; i <= poly.order(); i++) {
            os << "+" << poly.params()[i] << "x^" << i;
        }
        os << std::endl;
        return os;
    }
    const int order() const { return mOrder; }
    std::vector<double>& params() { return mParams; } 
protected:
    int mOrder;
    std::vector<double> mParams;
};