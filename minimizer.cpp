#include "minimizer.h"

#include <polyfem/State.hpp>
#include <polyfem/OptState.hpp>

#include <polyfem/utils/JSONUtils.hpp>
#include <polyfem/utils/Logger.hpp>
#include <polyfem/io/YamlToJson.hpp>

#include <polyfem/solver/forms/FrictionForm.hpp>
#include <polyfem/solver/ALSolver.hpp>
#include <polyfem/utils/Timer.hpp>

#include <ipc/utils/local_to_global.hpp>

// Include ALGLIB
#include "stdafx.h"
#include "optimization.h"

using namespace polyfem;
using namespace solver;
using namespace alglib;

namespace minimizer {
	ContactConstraint::ContactConstraint(const ipc::CollisionMesh& collision_mesh,
		const Eigen::VectorXd& x0,
		const double dhat,
		const ipc::BroadPhaseMethod broad_phase_method)
		:collision_mesh_(collision_mesh),
		dhat_(dhat),
		broad_phase_method_(broad_phase_method),
		broad_phase_(ipc::build_broad_phase(broad_phase_method)),
		V0(compute_displaced_surface(x0))
	{}

	Eigen::MatrixXd ContactConstraint::compute_displaced_surface(const Eigen::VectorXd& x) const
	{
		return collision_mesh_.displace_vertices(utils::unflatten(x, collision_mesh_.dim()));
	}

	void ContactConstraint::update_candidates(const Eigen::MatrixXd& displaced_surface) const
	{
		candidates_.build(
			collision_mesh_, V0, displaced_surface, dhat_/2);
	}

	double ContactConstraint::value(const Eigen::VectorXd& x) const {
		Eigen::MatrixXd V = compute_displaced_surface(x);
		assert(V.rows() == collision_mesh_.num_vertices());
		update_candidates(V);

		const size_t num_vertices = collision_mesh_.num_vertices();
		double c = 0.0;

		if (candidates_.empty())
		{
			return c;
		}

		const Eigen::MatrixXi& E = collision_mesh_.edges();
		const Eigen::MatrixXi& F = collision_mesh_.faces();

		for (size_t i = 0; i < candidates_.size(); i++)
		{
			double toi = 0;
			bool is_colliding = candidates_[i].ccd(candidates_[i].dof(V0, E, F), candidates_[i].dof(V, E, F), toi);
			if(is_colliding) {
				c += -sqrt(candidates_[i].compute_distance(candidates_[i].dof(V, E, F)));
			}
		}

		return c;
	}

	void ContactConstraint::first_derivative(const Eigen::VectorXd& x, Eigen::VectorXd& gradv_full) const {
		Eigen::MatrixXd V = compute_displaced_surface(x);
		assert(V.rows() == collision_mesh_.num_vertices());
		update_candidates(V);

		const size_t num_vertices = collision_mesh_.num_vertices();
		double c = 0.0;

		if (candidates_.empty())
		{
			return;
		}

		const Eigen::MatrixXi& E = collision_mesh_.edges();
		const Eigen::MatrixXi& F = collision_mesh_.faces();

		for (size_t i = 0; i < candidates_.size(); i++)
		{
			double toi = 0;
			bool is_colliding = candidates_[i].ccd(candidates_[i].dof(V0, E, F), candidates_[i].dof(V, E, F), toi);
			if (is_colliding) {
				double d = sqrt(candidates_[i].compute_distance(candidates_[i].dof(V, E, F)));
				auto v_ids = candidates_[i].vertex_ids(E, F);
				ipc::local_gradient_to_global_gradient(-candidates_[i].compute_distance_gradient(candidates_[i].dof(V, E, F))/(2*d), v_ids, collision_mesh_.dim(), gradv_full);
			}
		}

		return;
	}

	int projected_newton(NLProblem& objFunc, Eigen::VectorXd& x, const ipc::CollisionMesh& collision_mesh, const double dhat) {
		constexpr double Inf = std::numeric_limits<double>::infinity();
		Eigen::VectorXd grad = Eigen::VectorXd::Zero(x.rows());
		Eigen::VectorXd delta_x = Eigen::VectorXd::Zero(x.rows());
		Eigen::SparseMatrix<double> hessian;
		auto linear_solver = polysolve::linear::Solver::create("Eigen::PardisoLDLT", "");

		do
		{
			double energy = objFunc(x);
			objFunc.gradient(x, grad);
			objFunc.set_project_to_psd(true);
			objFunc.hessian(x, hessian);
			linear_solver->analyze_pattern(hessian, hessian.rows());
			try
			{
				linear_solver->factorize(hessian);
			}
			catch (const std::runtime_error& err)
			{
				// warn if using gradient descent
				logger().debug("Unable to factorize Hessian: \"{}\"", err.what());

				// Eigen::saveMarket(hessian, "problematic_hessian.mtx");
				return std::nan("");
			}

			linear_solver->solve(-grad, delta_x); // H Δx = -g
			double alpha = 1;
			{
				Eigen::VectorXd new_x = x + alpha * delta_x;
				//objFunc.max_step_size(x, new_x);
				int c_iter = 0;
				for (; c_iter < 25; c_iter++) {
					objFunc.line_search_begin(x, new_x);
					if (!objFunc.is_step_collision_free(x, new_x) || !std::isfinite(objFunc(new_x)))
					{
						alpha *= 0.5;
						new_x = x + alpha * delta_x;
					}
					else
					{
						objFunc.line_search_end();
						break;
					}
					objFunc.line_search_end();
				}
				if (c_iter == 25)
					logger().error("Collision Line search failed to find a valid finite energy step");

				//alpha = objFunc.max_step_size(x, new_x);
				int ls_iter = 0;
				for (; ls_iter < 25; ls_iter++) {
					new_x = x + alpha * delta_x;

					objFunc.solution_changed(new_x);
					const double new_energy = objFunc(new_x);
					if (new_energy - energy < 0)
						break;
					alpha = 0.5 * alpha;
				}

				if (ls_iter == 25)
					logger().error("Line search failed to find descent step");
			}

			{
				Eigen::VectorXd x1 = x + alpha * delta_x;
				if (objFunc.after_line_search_custom_operation(x, x1))
					objFunc.solution_changed(x1);
				x = x1;
			}
			if (grad.lpNorm<Eigen::Infinity>() < 1e-6)
				break;
		} while (true);
		return EXIT_SUCCESS;
	}

	// --- OBJECTIVE FUNCTION ---
	void nlcfunc_jac(const real_1d_array& x_, real_1d_array& fi_, real_2d_array& jac_, void* ptr)
	{
		OptimizationFunctions* func = static_cast<OptimizationFunctions*>(ptr);
		// 1. MAP ALGLIB INPUT TO EIGEN (Zero-copy read)
		// .getcontent() gives the raw pointer, .length() gives size
		const Eigen::Map<const Eigen::VectorXd> x(x_.getcontent(), x_.length());

		// 2. MAP ALGLIB OUTPUT TO EIGEN (Zero-copy write)
		Eigen::Map<Eigen::VectorXd> fi(fi_.getcontent(), fi_.length());

		// 2. MAP Jacobian TO EIGEN (Zero-copy write)
		Eigen::VectorXd grad_f = Eigen::VectorXd::Zero(x_.length(), 1);
		Eigen::VectorXd grad_v = Eigen::VectorXd::Zero(func->c.collision_mesh_.num_vertices()*3, 1);

		fi[0] = func->objFunc(x);
		fi[1] = func->c.value(func->objFunc.reduced_to_full(x));

		func->objFunc.gradient(x, grad_f);
		func->c.first_derivative(func->objFunc.reduced_to_full(x), grad_v);
		grad_v = func->objFunc.full_to_reduced_grad(grad_v);

		std::copy(grad_f.begin(), grad_f.end(), jac_[0]);
		std::copy(grad_v.begin(), grad_v.end(), jac_[1]);
	}

	int sqp(NLProblem& objFunc, Eigen::VectorXd& x, const ipc::CollisionMesh& collision_mesh, const double dhat) {
		ContactConstraint c = ContactConstraint(collision_mesh, objFunc.reduced_to_full(x), dhat, ipc::BroadPhaseMethod::HASH_GRID);
		OptimizationFunctions funcs = { c, objFunc };
		// Problem sizes
		int n_vars = x.size();

		// Initial guess using ALGLIB format
		real_1d_array alg_x;
		alg_x.setcontent(n_vars, x.data());

		// Setup Solver
		minnlcstate state;
		minnlcreport rep;

		minnlccreate(n_vars, alg_x, state);
		minnlcsetalgosqp(state);

		real_1d_array nl = "[0]";
		real_1d_array nu = "[inf]";
		minnlcsetnlc2(state, nl, nu);
		minnlcsetcond(state, 0.000001, 0);

		//real_1d_array test_x = alg_x;
		//real_1d_array test_v;
		//real_2d_array test_jac;
		//test_v.setlength(2);
		//test_jac.setlength(2, x.size());
		//for (int i = 1; i < x.size(); i+=3) {
		//	test_x[i] -= 0.1;
		//}
		//nlcfunc_jac(test_x, test_v, test_jac, &funcs);
		//printf("Test constraint f: %s\n", test_v.tostring(2).c_str());
		//printf("Test constraint Jac: %s\n", test_jac.tostring(2).c_str());

		// Optimize
		alglib::minnlcoptimize(state, nlcfunc_jac, NULL, &funcs);

		// Get results
		minnlcresults(state, alg_x, rep);

		// Map final result to Eigen for printing/further use
		Eigen::Map<Eigen::VectorXd> x_final(alg_x.getcontent(), n_vars);
		x = x_final;

		printf("Optimization status: %d\n", int(rep.terminationtype));
		printf("Optimized X: %s\n", alg_x.tostring(2).c_str()); 

		return EXIT_SUCCESS;
	}

	void finite_difference_constraint(NLProblem& objFunc, Eigen::VectorXd& x, const ipc::CollisionMesh& collision_mesh, const double dhat) {
		double eps = 1e-6;
		ContactConstraint c = ContactConstraint(collision_mesh, objFunc.reduced_to_full(x), dhat, ipc::BroadPhaseMethod::HASH_GRID);
		Eigen::VectorXd x_new = x ;
		for (int i = 1; i < x.size(); i+=3) {
			x_new[i] -= 0.01;
		}

		Eigen::VectorXd grad = Eigen::VectorXd::Zero(c.collision_mesh_.num_vertices() * 3, 1);
		c.first_derivative(objFunc.reduced_to_full(x_new), grad);
		grad = objFunc.full_to_reduced_grad(grad);

		double f = c.value(objFunc.reduced_to_full(x_new));
		Eigen::VectorXd grad_fd = Eigen::VectorXd::Zero(x.size(), 1);
		for (int i = 0; i < x.size(); i++) {
			Eigen::VectorXd x_e = x_new;
			x_e[i] += eps;
			double f_e0 = c.value(objFunc.reduced_to_full(x_e));
			x_e[i] -= eps * 2;
			double f_e1 = c.value(objFunc.reduced_to_full(x_e));
			grad_fd[i] = (f_e0 - f_e1) / (2 * eps);
		}
		std::cout << "grad_diff_norm: " << (grad - grad_fd).norm() << std::endl;
		std::cout << "grad_diff: " << (grad - grad_fd).transpose() << std::endl;
		std::cout << "grad: " << grad.transpose() << std::endl;
		std::cout << "grad_fd: " << grad_fd.transpose() << std::endl;

	}
}