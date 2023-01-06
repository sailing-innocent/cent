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
    float forward(float x) {
        float res = 0.0;
        for (auto i = 0; i <= mOrder; i++) {
            res += std::pow(x,i) * mParams[i];
        }
        return res;
    }
    float operator()(float x) {
        return static_cast<float>(forward(static_cast<float>(x)));
    }
    bool setParam(int _order, float _param) {
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
    std::vector<float>& params() { return mParams; } 

    std::vector<float> ESM_dir(std::vector<float>& samples_x, std::vector<float>& samples_y, int samples_N) {
        std::vector<float> res;
        for (auto n = 0; n <= mOrder; n++) {
            float dEdw = 0.0;
            for (auto j = 0; j < samples_N; j++) {
                dEdw += (forward(samples_x[j]) - samples_y[j]) * std::pow(samples_x[j], n);
            }
            res.push_back(dEdw);
            // std::cout << dEdw << ",";
        }
        // std::cout << std::endl;
        return res;
    }
protected:
    int mOrder;
    std::vector<float> mParams;
};