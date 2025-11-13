#pragma once
#include <polyfem/Common.hpp>
#include <polyfem/utils/Types.hpp>
#include <polyfem/solver/NLProblem.hpp>


#include <ipc/collision_mesh.hpp>
#include <ipc/utils/logger.hpp>
#include <ipc/broad_phase/broad_phase.hpp>
#include <ipc/collisions/normal/normal_collisions.hpp>

#include <ipc/broad_phase/brute_force.hpp>
#include <ipc/broad_phase/bvh.hpp>
#include <ipc/broad_phase/hash_grid.hpp>
#include <ipc/broad_phase/spatial_hash.hpp>
#include <ipc/broad_phase/sweep_and_prune.hpp>
#include <ipc/broad_phase/sweep_and_tiniest_queue.hpp>

namespace minimizer {
    class ContactConstraint {
    public:
        const double dhat_;
        const double dmin_ = 0;
        const ipc::CollisionMesh& collision_mesh_;
        const ipc::BroadPhaseMethod broad_phase_method_;
        const std::shared_ptr<ipc::BroadPhase> broad_phase_;
        const Eigen::MatrixXd V0;
        mutable ipc::Candidates candidates_;

        ContactConstraint(const ipc::CollisionMesh& collision_mesh,
            const Eigen::VectorXd& x0,
            const double dhat,
            const ipc::BroadPhaseMethod broad_phase_method);

        void update_candidates(const Eigen::MatrixXd& displaced_surface) const;
        Eigen::MatrixXd compute_displaced_surface(const Eigen::VectorXd& x) const;
        double value(const Eigen::VectorXd& x) const;
        void first_derivative(const Eigen::VectorXd& x, Eigen::VectorXd& gradv) const;
    };

    struct OptimizationFunctions {
        ContactConstraint& c;      
        polyfem::solver::NLProblem& objFunc;
    };

    int projected_newton(polyfem::solver::NLProblem& objFunc, Eigen::VectorXd& x, const ipc::CollisionMesh& collision_mesh, const double dhat);

    int sqp(polyfem::solver::NLProblem& objFunc, Eigen::VectorXd& x, const ipc::CollisionMesh& collision_mesh, const double dhat);

    void finite_difference_constraint(polyfem::solver::NLProblem& objFunc, Eigen::VectorXd& x, const ipc::CollisionMesh& collision_mesh, const double dhat);
}