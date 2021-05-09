// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.
// C++ 17, structured binding, proper operator new (alignment)
#include <random>
#include <fstream>
#include <vector>
#include <cmath>
#include <ratio>
#include <Eigen/Dense>
#include <Eigen/Sparse>

constexpr std::size_t NUM_PAR = 96;
constexpr double RADIUS = 1. / 4;
constexpr double STIFF_CONST = 128;
constexpr double TIME_NORMALIZER = 1 << 10;
constexpr double time_step = 1. / TIME_NORMALIZER;
typedef std::ratio<1, 100> convergence;

template <typename T>
T clip(double f) {
    if (f > std::numeric_limits<T>::max())
        return std::numeric_limits<T>::max();
    else if (f < std::numeric_limits<T>::min())
        return std::numeric_limits<T>::min();
    return static_cast<T>(f);
}

double spring_force(double a, double b) {
    // by b, on a
    if (auto dist = std::abs(a - b); dist < RADIUS * 2)
        if (a > b)
            return (STIFF_CONST / 2) * (2 * RADIUS - dist);
        else if (a == b)
            throw;
        else
            return -(STIFF_CONST / 2) * (2 * RADIUS - dist);
    return 0;
}

double spring_derivative(double a, double b) {
    // perturbs a
    if (auto dist = std::abs(a - b); dist < RADIUS * 2)
        if (a == b)
            throw;
        else
            return -STIFF_CONST / 2;
    return 0;
}

typedef Eigen::Matrix<double, NUM_PAR, 1> Vector;

Vector spring_force(const Vector& X) {
    Eigen::Matrix<double, NUM_PAR, Eigen::Dynamic> ret(NUM_PAR, NUM_PAR);
    ret.diagonal().setConstant(0);
    for (std::size_t i = 0; i != X.size(); ++i)
        for (std::size_t j = i + 1; j != X.size(); ++j) {
            ret(i, j) = spring_force(X(i), X(j));
            ret(j, i) = -ret(i, j);
        }
    //return ret.rowwise().sum();
    return ret.colwise().sum().transpose();
}

Eigen::Matrix<double, NUM_PAR, Eigen::Dynamic> spring_derivative(const Vector& X) {
    Eigen::Matrix<double, NUM_PAR, Eigen::Dynamic> ret(NUM_PAR, NUM_PAR);
    for (std::size_t i = 0; i != X.size(); ++i)
        for (std::size_t j = i + 1; j != X.size(); ++j) {
            ret(i, j) = spring_derivative(X(i), X(j));
            ret(j, i) = ret(i, j); // double negation
        }
    for (std::size_t i = 0; i != X.size(); ++i) {
        ret(i, i) = 0;
        for (std::size_t j = 0; j != X.size(); ++j)
            if (j != i)
                ret(i, i) -= ret(i, j);
    }
    return ret;
}

template <typename T>
std::enable_if_t<
    (T::ColsAtCompileTime > 1) || T::ColsAtCompileTime == Eigen::Dynamic,
    Eigen::SparseMatrix<double>
>
spring_force(const T& _X) {
    Eigen::Matrix<double, NUM_PAR, Eigen::Dynamic> X = _X;
    Eigen::SparseMatrix<double> ret(NUM_PAR, X.cols());
    for (Eigen::Index i = 0; i != X.cols(); ++i)
        ret.col(i) = spring_force(X.col(i)).sparseView();
    return ret;
}

struct record_t {
    unsigned t = 0;
    Vector X;
    Vector V;
};

std::vector<record_t>* ptr;

unsigned next_interaction(const Vector& X, const Vector& V) {
    int min = std::numeric_limits<int>::max();
    assert(X.size() == V.size());
    for (Eigen::Index i = 0; i != X.size(); ++i) {
        for (auto rhs = i + 1; rhs != X.size(); ++rhs) {
            double t = (X[rhs] - X[i]) / (V[i] - V[rhs]);
            if (t >= 0) {
                double t_thres = (2 * RADIUS) / std::abs(V[i] - V[rhs]);
                if (t > t_thres) {
                    min = std::min(min, std::max(1, clip<int>((t - t_thres) * TIME_NORMALIZER)));
                    // bounds min by 1 from below,
                    // which relies on the assumption that is sufficiently large to cover the spring compression
                    continue;
                }
                else if (t < time_step) {
                    std::printf("Warning: Time = %u, \\Delta x = %f; \\Delta v = %f", ptr->back().t, X[rhs] - X[i], V[i] - V[rhs]);
                    if (X[rhs] != X[i])
                        std::printf(".\nError: Minimal time step too large.\n");
                    else
                        std::puts(", rigid body collision."); // FIXME: double, nearly impossible
                }
                return 1;
            }
        }
    }
    return min;
}

struct dump_on_exit {
    std::ofstream outfile;
    const std::vector<record_t>& history;
    dump_on_exit(std::vector<record_t>& history, const char* fn)
        : history(history), outfile(fn, std::ios::binary) {}
    ~dump_on_exit() {
        for (auto& [t, X, V] : history) {
            outfile.write(reinterpret_cast<const char*>(&t), sizeof(unsigned));
            outfile.write(reinterpret_cast<const char*>(X.data()), sizeof(double) * NUM_PAR);
            outfile.write(reinterpret_cast<const char*>(V.data()), sizeof(double) * NUM_PAR);
        }
    }
};

int main(int argc, char** argv) {
    constexpr unsigned T_END = (1 << 7) * TIME_NORMALIZER;

    std::mt19937_64 gen1(std::random_device{}());
    std::mt19937_64 gen2(std::random_device{}());
    std::uniform_real_distribution<> dis_d(-0.5, 0.5);
    std::uniform_real_distribution<> dis_v(-1, 1);
    record_t particles;
    for (std::size_t n = 0; n != NUM_PAR; ++n) {
        particles.X[n] = n + dis_d(gen1);
        particles.V[n] = dis_v(gen2);
    }

    std::vector<record_t> history;
    ptr = &history;
    dump_on_exit _{ history, argv[1] };
    history.reserve(1024);
    history.push_back(particles);

    Eigen::DiagonalMatrix<double, NUM_PAR> _identity;
    _identity.setIdentity();
    Eigen::SparseMatrix<double> identity(_identity);
    for (particles.t = 1; particles.t < T_END; ++particles.t) {
        auto forces = spring_force(particles.X);
        if (forces.any()) {
            // Backward Euler
            record_t draft = particles;
            for (std::size_t i = 0; i != 1; ++i) {
                Eigen::SparseMatrix<double> Jacobian{ spring_derivative(draft.X).sparseView() };
                Eigen::MatrixXd Hold = Jacobian;
                Eigen::SparseMatrix<double> quarter = identity - time_step * time_step * Jacobian;
                Eigen::SimplicialLLT<Eigen::SparseMatrix<double>> solver(quarter);
                draft.V = particles.V + spring_force(draft.X) * time_step;
                draft.X = solver.solve(particles.X) + time_step * solver.solve(particles.V);
                draft.V = time_step * solver.solve(Jacobian * particles.X) + solver.solve(particles.V);
                //std::printf("%f, ", (draft.X - (particles.X + (particles.V + forces * time_step) * time_step)).norm() * 1000);
            }
            particles.X = draft.X;
            particles.V = draft.V;
            //std::puts("\b\n#################");
        }
        else {
            if (particles.t > 1)
                std::printf("Warning: Unnecessary key step at time = %u.\n", particles.t);
            particles.X += particles.V * time_step;
        }
        history.push_back(particles);

        unsigned n_steps = next_interaction(particles.X, particles.V) - 1;
        if (n_steps) {
            std::printf("Info: skipping %u.\n", n_steps);
            n_steps = std::min(T_END - particles.t, n_steps);
            particles.X += particles.V * (time_step * n_steps);
            particles.t += n_steps;
            history.push_back(particles);
        }
    }
}
