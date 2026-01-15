#include "mex.hpp"
#include "mexAdapter.hpp"

#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <streambuf>
#include <mutex>
#include <map>
#include <algorithm> 
#include <cmath> // for std::isfinite

// ==============================================================================
// POLYFEM INCLUDES
// ==============================================================================
#include <polyfem/State.hpp>
#include <polyfem/utils/Logger.hpp>
#include <polyfem/utils/Timer.hpp> // For POLYFEM_SCOPED_TIMER
#include <polyfem/solver/forms/FrictionForm.hpp>
#include <polyfem/solver/forms/BarrierContactForm.hpp>
#include <polyfem/solver/ALSolver.hpp>
#include <polyfem/solver/NLProblem.hpp>
#include <polysolve/nonlinear/Solver.hpp>
#include <polysolve/nonlinear/PostStepData.hpp>

#include <ipc/collision_mesh.hpp>
#include <ipc/candidates/candidates.hpp>
#include <ipc/candidates/collision_stencil.hpp>
#include <ipc/broad_phase/sweep_and_prune.hpp>
#include <ipc/ccd/tight_inclusion_ccd.hpp>
#include <ipc/utils/local_to_global.hpp>

#include <tbb/blocked_range.h>
#include <tbb/combinable.h>
#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_for_each.h>
#include <tbb/parallel_reduce.h>

// DOUBLECCD
#include <doubleCCD/doubleccd.hpp>

// Third-party
#include <igl/Timer.h>
#include <jse/jse.h>

// SPDLOG HEADERS
#include <spdlog/spdlog.h>
#include <spdlog/sinks/base_sink.h>
#include <spdlog/details/null_mutex.h>

using namespace polyfem;
using namespace matlab::data;
using namespace matlab::mex;

//#define USE_PERSISTANT_CANDIDATES

// ==============================================================================
// HELPER: Logger Sink
// ==============================================================================
template<typename Mutex>
class MatlabSink : public spdlog::sinks::base_sink<Mutex>
{
    std::shared_ptr<matlab::engine::MATLABEngine> matlabPtr;
    ArrayFactory factory;

public:
    explicit MatlabSink(std::shared_ptr<matlab::engine::MATLABEngine> ptr) : matlabPtr(ptr) {}

protected:
    void sink_it_(const spdlog::details::log_msg& msg) override {
        spdlog::memory_buf_t formatted;
        spdlog::sinks::base_sink<Mutex>::formatter_->format(msg, formatted);
        std::string log_str(formatted.data(), formatted.size());

        matlabPtr->feval(u"fprintf", 0, std::vector<Array>({
            factory.createScalar("%s"),
            factory.createScalar(log_str)
            }));
    }

    void flush_() override {
        //matlabPtr->feval(u"drawnow", 0, std::vector<Array>({ factory.createScalar("limitrate") }));
    }
};
using MatlabSink_mt = MatlabSink<std::mutex>;

// ==============================================================================
// HELPER: Stream Buffer
// ==============================================================================
class MatlabStreamBuf : public std::streambuf {
    std::shared_ptr<matlab::engine::MATLABEngine> matlabPtr;
    ArrayFactory factory;

public:
    MatlabStreamBuf(std::shared_ptr<matlab::engine::MATLABEngine> ptr) : matlabPtr(ptr) {}

protected:
    virtual std::streamsize xsputn(const char* s, std::streamsize n) override {
        matlabPtr->feval(u"fprintf", 0, std::vector<Array>({
            factory.createScalar("%s"),
            factory.createScalar(std::string(s, n))
            }));
        return n;
    }
    virtual int overflow(int c) override {
        if (c != EOF) {
            char ch = static_cast<char>(c);
            matlabPtr->feval(u"fprintf", 0, std::vector<Array>({
                factory.createScalar("%s"),
                factory.createScalar(std::string(1, ch))
                }));
        }
        return c;
    }
};

// ==============================================================================
// HELPER: Time Statistics
// ==============================================================================
struct SimStatistics {
    igl::Timer timer;
    double total_time = 0;
    double init_time = 0;
    double objective_time = 0;
	double collision_time = 0;
    double extra_factorizing_time = 0;
    int func_eval = 0;
    int hessian_eval = 0;
    int ccd_count = 0;
};

// ==============================================================================
//  APPLY CONSTRAINTS CONDITIONS
// ==============================================================================
Eigen::VectorXd prepare(polyfem::State& state, Eigen::MatrixXd& sol, int t) {
    using namespace polyfem::solver; // For ALSolver, NLProblem

    const double t0 = state.args["time"]["t0"];
    const int time_steps = state.args["time"]["time_steps"];
    const double dt = state.args["time"]["dt"];
    polyfem::logger().info("{}/{}  t={}", t, time_steps, t0 + dt * t);
    assert(state.solve_data.nl_problem != nullptr);
    NLProblem& nl_problem = *(state.solve_data.nl_problem);

    assert(sol.size() == state.rhs.size());

    if (nl_problem.uses_lagging())
    {
        POLYFEM_SCOPED_TIMER("Initializing lagging");
        nl_problem.init_lagging(sol);
        polyfem::logger().info("Lagging iteration 1:");
    }

    // ---------------------------------------------------------------------

    // Save the subsolve sequence for debugging
    int subsolve_count = 0;
    state.save_subsolve(subsolve_count, t, sol, Eigen::MatrixXd());

    // ---------------------------------------------------------------------

    ALSolver al_solver(
        state.solve_data.al_form,
        state.args["solver"]["augmented_lagrangian"]["initial_weight"],
        state.args["solver"]["augmented_lagrangian"]["scaling"],
        state.args["solver"]["augmented_lagrangian"]["max_weight"],
        state.args["solver"]["augmented_lagrangian"]["eta"],
        [&](const Eigen::VectorXd& x) {
            state.solve_data.update_barrier_stiffness(sol);
        });

    al_solver.post_subsolve = [&](const double al_weight) {
        state.stats.solver_info.push_back(
            { {"type", al_weight > 0 ? "al" : "rc"},
                {"t", t} });
        if (al_weight > 0)
            state.stats.solver_info.back()["weight"] = al_weight;
        state.save_subsolve(++subsolve_count, t, sol, Eigen::MatrixXd());
        };

    al_solver.solve_al(nl_problem, sol,
        state.args["solver"]["augmented_lagrangian"]["nonlinear"], state.args["solver"]["linear"], state.units.characteristic_length());

    assert(sol.size() == nl_problem.full_size());

    Eigen::VectorXd tmp_sol = nl_problem.full_to_reduced(sol);
    nl_problem.use_reduced_size();
    nl_problem.line_search_begin(sol, tmp_sol);

    if (!std::isfinite(nl_problem.value(tmp_sol))
        || !nl_problem.is_step_valid(sol, tmp_sol)
        || !nl_problem.is_step_collision_free(sol, tmp_sol))
        polyfem::log_and_throw_error("Failed to apply constraints conditions; solve with augmented lagrangian first!");
    nl_problem.line_search_end();

    // --------------------------------------------------------------------
    // Perform one final solve with the DBC projected out
    polyfem::logger().debug("Successfully applied constraints conditions; solving in reduced space");
    nl_problem.init(sol);
    state.solve_data.update_barrier_stiffness(sol);
	return tmp_sol;
}

// ==============================================================================
//  SAVE RESULTS
// ==============================================================================
void save(polyfem::State& state, Eigen::MatrixXd& sol, int t) {
    using namespace polyfem::solver; // For ALSolver, NLProblem

    const double t0 = state.args["time"]["t0"];
    const int time_steps = state.args["time"]["time_steps"];
    const double dt = state.args["time"]["dt"];

    if (state.args["space"]["advanced"]["count_flipped_els_continuous"])
    {
        const auto invalidList = utils::count_invalid(state.mesh->dimension(), state.bases, state.geom_bases(), sol);
        polyfem::logger().debug("Flipped elements (cnt {}) : {}", invalidList.size(), invalidList);
    }

    state.save_timestep(t0 + dt * t, t + static_cast<int>(t0 / dt), t0, dt, sol, Eigen::MatrixXd());

    {
        POLYFEM_SCOPED_TIMER("Update quantities");
        state.solve_data.time_integrator->update_quantities(sol);
        state.solve_data.nl_problem->update_quantities(t0 + (t + 1) * dt, sol);
        state.solve_data.update_dt();
        state.solve_data.update_barrier_stiffness(sol);
    }

    if (state.time_callback)
        state.time_callback(t, time_steps, t0 + dt * t, t0 + dt * time_steps);

    const std::string& state_path = state.resolve_output_path(fmt::format(state.args["output"]["data"]["state"], t + static_cast<int>(t0 / dt)));
    if (!state_path.empty())
        state.solve_data.time_integrator->save_state(state_path);

    state.save_restart_json(t0, dt, t);

}

// ==============================================================================
// THE MEX FUNCTION
// ==============================================================================
class MexFunction : public matlab::mex::Function {
    ArrayFactory factory;
    std::shared_ptr<matlab::engine::MATLABEngine> matlabPtr = getEngine();
    static Eigen::MatrixXd V_prev;
    static Eigen::VectorXd sol_ls_0;
    static Eigen::VectorXd sol_ls_1;
	static std::vector<double> t_list;
    static SimStatistics sim_stat;
    static std::unique_ptr<ipc::Candidates> candidates;

public:
    void operator()(ArgumentList outputs, ArgumentList inputs) {

        // 1. Redirect Output using C++ API Buffer
        MatlabStreamBuf buffer(matlabPtr);
        std::streambuf* oldOut = std::cout.rdbuf(&buffer);
        std::streambuf* oldErr = std::cerr.rdbuf(&buffer);

        // 2. Setup Logger using C++ API Sink
        auto matlab_sink = std::make_shared<MatlabSink_mt>(matlabPtr);
        matlab_sink->set_pattern("[%^%l%$] %v");
        spdlog::logger& p_logger = polyfem::logger();
        p_logger.sinks().clear();
        p_logger.sinks().push_back(matlab_sink);
        p_logger.set_level(spdlog::level::info);
        p_logger.flush_on(spdlog::level::info);
        //std::cout << "Logger name: " << p_logger.name() << "Current level: " << (int)p_logger.level() << std::endl;

        try {
            checkArguments(inputs);

            CharArray cmdArr = inputs[0];
            std::string cmd = cmdArr.toAscii();

            TypedArray<uint64_t> handleArr = inputs[1];
            uint64_t ptr_val = handleArr[0];
            if (ptr_val == 0) throw std::runtime_error("Received null state pointer.");
            polyfem::State* state = reinterpret_cast<polyfem::State*>(ptr_val);

            // --------------------------------------------------------------
            // COMMAND: PREPARE
            // Usage: [sol, sol_reduced] = polyfem_problem_mex('prepare', handle, sol, t)
            // --------------------------------------------------------------
            if (cmd == "prepare") {
                Eigen::MatrixXd sol;
                int t;
                double inflation_radius;

                // Accept Inputs 
                if (inputs.size() == 5 && inputs[2].getType() == ArrayType::DOUBLE) {
                    matlabToEigen(inputs[2], sol);
                    t = getInt(inputs[3]);
					inflation_radius = inputs[4][0];
                }
                else {
                    throw std::runtime_error("Invalid input arguments. Usage: [sol, sol_reduced] = polyfem_problem_mex('prepare', handle, sol, t, inflation_radius)");
                }
                sim_stat.timer.start();
                igl::Timer init_timer;
                init_timer.start();
				Eigen::VectorXd sol_reduced = prepare(*state, sol, t);
                V_prev = state->collision_mesh.displace_vertices(utils::unflatten(sol, state->collision_mesh.dim()));
				sol_ls_0 = sol_reduced;
				sol_ls_1 = sol_reduced;
                if (candidates) {
                    candidates.reset();
                    mexUnlock();
                }
                candidates = std::make_unique<ipc::Candidates>();
                init_timer.stop();
                sim_stat.init_time += init_timer.getElapsedTime();

#ifdef  USE_PERSISTANT_CANDIDATES
                try {
                    assert(state->solve_data.nl_problem != nullptr);
                    polyfem::solver::NLProblem& nl_problem = *(state->solve_data.nl_problem);
                    const ipc::CollisionMesh& collision_mesh = state->collision_mesh;
                    Eigen::MatrixXd V0 = collision_mesh.displace_vertices(utils::unflatten(sol, collision_mesh.dim()));
                    assert(V0.rows() == collision_mesh.num_vertices());
                    candidates->build(collision_mesh, V0, inflation_radius);
                    printToMatlab("inflation radius: " + std::to_string(inflation_radius) + "\n");
                    printToMatlab("candidate size: " + std::to_string(candidates->size()) + "\n");
                }
                catch (const std::exception& e) {
                    candidates.reset();
                    throw std::runtime_error("Build candiates failed: " + std::string(e.what()));
                    return;
                }

                mexLock();
#endif

				outputs[0] = eigenToMatlab(sol);
				outputs[1] = eigenToMatlab(sol_reduced);
            }
            // --------------------------------------------------------------
			// COMMAND: EVAL_F
            // Usage: [f,g,h] = polyfem_problem_mex('eval_f', handle, sol_reduced)
            // --------------------------------------------------------------
            else if (cmd == "eval_f") {
                Eigen::MatrixXd sol;
                Eigen::VectorXd tmp_sol;
                // Accept Inputs 
                if (inputs.size() == 4 && inputs[2].getType() == ArrayType::DOUBLE && inputs[3].getType() == ArrayType::DOUBLE) {
                    matlabToEigen(inputs[2], sol);
                    matlabToEigen(inputs[3], tmp_sol);
                }
                else {
                    throw std::runtime_error("Invalid input arguments. Usage: [f,g] = polyfem_problem_mex('eval_c', handle, sol_reduced)");
                }
                igl::Timer timer;
                // Start the timer
                timer.start();

                assert(state.solve_data.nl_problem != nullptr);
                polyfem::solver::NLProblem& nl_problem = *(state->solve_data.nl_problem);
                double f = 0.0;
                Eigen::VectorXd grad;
				nl_problem.solution_changed(tmp_sol);
                //bool is_step_valid = true;

				igl::Timer collision_timer;
				collision_timer.start();
                const ipc::CollisionMesh& collision_mesh = state->collision_mesh;
                Eigen::MatrixXd V = collision_mesh.displace_vertices(utils::unflatten(nl_problem.reduced_to_full(tmp_sol), collision_mesh.dim()));
                bool is_step_valid = !ipc::has_intersections(collision_mesh, V, std::make_shared<ipc::SweepAndPrune>());

                //if(is_step_valid)
                //    is_step_valid = nl_problem.is_step_collision_free(nl_problem.full_to_reduced(sol), tmp_sol);

                //logger().debug("1: Is step valid? {}", is_step_valid);
                /*
                for (auto form : nl_problem.forms()) {
                    std::shared_ptr<polyfem::solver::ContactForm> contact_form = std::dynamic_pointer_cast<polyfem::solver::ContactForm>(form);
                    if (contact_form != nullptr) {
                        logger().debug("Barrier Stiffness: {:e}, is project to psd: {}", contact_form->barrier_stiffness(), contact_form->is_project_to_psd());
                    }
                    std::shared_ptr<polyfem::solver::BarrierContactForm> barrier_contact_form = std::dynamic_pointer_cast<polyfem::solver::BarrierContactForm>(form);
                    if (barrier_contact_form != nullptr) {
                        logger().debug("Minimum distance: {:e}", barrier_contact_form->collision_set().compute_minimum_distance(collision_mesh, barrier_contact_form->compute_displaced_surface(tmp_sol)));
                    }
                }
                */

                if(is_step_valid)
                {
                    ipc::Candidates& collisions = *candidates;
                    double dhat = polyfem::Units::convert(state->args["contact"]["dhat"], state->units.length());
                    polyfem::solver::NLProblem& nl_problem = *(state->solve_data.nl_problem);
                    const ipc::CollisionMesh& collision_mesh = state->collision_mesh;
                    Eigen::MatrixXd V = collision_mesh.displace_vertices(utils::unflatten(nl_problem.reduced_to_full(tmp_sol), collision_mesh.dim()));
                    Eigen::MatrixXd V0 = collision_mesh.displace_vertices(utils::unflatten(sol, collision_mesh.dim()));
                    const int ndof = collision_mesh.num_vertices() * collision_mesh.dim();
                    assert(V.rows() == collision_mesh.num_vertices());
                    assert(V0.rows() == collision_mesh.num_vertices());
                    if(outputs.size() > 2)
                        collisions.build(collision_mesh, V_prev, V, dhat / 2, std::make_shared<ipc::SweepAndPrune>());
                    if (!collisions.empty()) {
                        const Eigen::MatrixXi& E = collision_mesh.edges();
                        const Eigen::MatrixXi& F = collision_mesh.faces();
                        //Eigen::MatrixXd V_ls_0 = collision_mesh.displace_vertices(utils::unflatten(nl_problem.reduced_to_full(sol_ls_0), collision_mesh.dim()));
                        //printToMatlab("Mesh statistics, vertices number: " + std::to_string(V.size() / 3) + ", edge number: " + std::to_string(E.size() / 2) + ", face number : " + std::to_string(F.size() / 3) + '\n');
                        /*
                        for (size_t i = 0; i < collisions.size(); i++) {
                            double toi = 0;
                            bool is_colliding = false;
                            ipc::VectorMax12d dof = collisions[i].dof(V, E, F);
                            ipc::VectorMax12d dof0 = collisions[i].dof(V0, E, F);
                            ipc::VectorMax12d dof_prev = collisions[i].dof(V_prev, E, F);
                            double d_prev = sqrt(collisions[i].compute_distance(dof_prev));
                            double d = sqrt(collisions[i].compute_distance(dof));
                            ipc::VectorMax12d local_grad_prev = collisions[i].compute_distance_gradient(dof_prev);
                            ipc::VectorMax12d local_grad = collisions[i].compute_distance_gradient(dof);
                            double c_approx = -d_prev + local_grad_prev.dot(dof - dof_prev) / (2 * d_prev);

                            ////printToMatlab("ccd candidate: " + std::to_string(i) + ", dof0: " + eigenToString(collisions[i].dof(V_ls_0, E, F).transpose()) + ", dof : " + eigenToString(dof.transpose()) + "\n");
                            igl::Timer ccd_timer;
                            ccd_timer.start();
                            if(is_linesearch){
                                is_colliding = (t_ls > t_list[i]);
                            }
                            else {
                                is_colliding = collisions[i].ccd(dof0, dof, toi);
                                //t_list.push_back(toi);
                            }
                            ccd_timer.stop();
                            double ccd_time = ccd_timer.getElapsedTimeInMicroSec();
                            //printToMatlab("tight inclusion ccd_time : " + std::to_string(ccd_time) + "us \n");
                            //if (abs(d - abs(c_approx)) < 1e-3 && d > 1e-2) {
                            //    is_colliding = !std::signbit(c_approx);
                            //    //printToMatlab("ccd candidate " + std::to_string(i) + ", d0: " + std::to_string(d0) + ", d: " + std::to_string(d) + ", c_approx: " + std::to_string(c_approx) + '\n');
                            //}
                            //else
                            ccd_timer.start();
                            {
                                if (i < collisions.vv_candidates.size() + collisions.ev_candidates.size())
                                    continue;
                                else if (i < collisions.vv_candidates.size() + collisions.ev_candidates.size() + collisions.ee_candidates.size())
                                    is_colliding = doubleccd_edge_edge(dof0, dof);
                                else if (i < collisions.size())
                                    is_colliding = doubleccd_vertex_face(dof0, dof);
                                else
                                    polyfem::log_and_throw_error("Index out of candidates range.");
                                ccd_count++;
                            }
                            ccd_timer.stop();
                            ccd_time = ccd_timer.getElapsedTimeInMicroSec();
                           // printToMatlab("double ccd_time : " + std::to_string(ccd_time) + "us \n");
                            if (is_colliding) {
                                is_step_valid = false;
                                break;
                            }
                        }
                        */
                        //
                        //igl::Timer ccd_timer;
                        //ccd_timer.start();
                        sim_stat.ccd_count++;
                        tbb::parallel_for(
                            tbb::blocked_range<size_t>(0, collisions.size()),
                            [&](tbb::blocked_range<size_t> r) {
                                for (size_t i = r.begin(); i < r.end(); i++) {
                                    double toi = 0;
                                    bool is_colliding = false;
                                    ipc::VectorMax12d dof = collisions[i].dof(V, E, F);
                                    ipc::VectorMax12d dof0 = collisions[i].dof(V_prev, E, F);

                                    is_colliding = collisions[i].ccd(dof0, dof, toi);

                                    if(is_colliding)
                                    {
                                        if (i < collisions.vv_candidates.size() + collisions.ev_candidates.size())
                                            is_colliding = is_colliding;
                                        else if (i < collisions.vv_candidates.size() + collisions.ev_candidates.size() + collisions.ee_candidates.size())
                                            is_colliding = doubleccd_edge_edge(dof0, dof);
                                        else if (i < collisions.size())
                                            is_colliding = doubleccd_vertex_face(dof0, dof);
                                        else
                                            polyfem::log_and_throw_error("Index out of candidates range.");
                                    }

                                    if (is_colliding) {
                                        is_step_valid = false;
                                        return;
                                    }
                                }
                            });
                        //ccd_timer.stop();
                        //double ccd_time = ccd_timer.getElapsedTime();
                        //printToMatlab("double ccd_time : " + std::to_string(ccd_time) + "s \n");
                        //printToMatlab("Candidate size: " + std::to_string(collisions.size()) + ",ccd size:" + std::to_string(ccd_count) +'\n');

                        //ccd_timer.start();
                        //collisions.compute_collision_free_stepsize(collision_mesh, V0, V);
                        //tbb::parallel_for(
                        //    tbb::blocked_range<size_t>(0, collisions.size()),
                        //    [&](tbb::blocked_range<size_t> r) {
                        //        for (size_t i = r.begin(); i < r.end(); i++) {
                        //            const ipc::CollisionStencil& candidate = collisions[i];
                        //            ipc::VectorMax12d dof = collisions[i].dof(V, E, F);
                        //            ipc::VectorMax12d dof0 = collisions[i].dof(V0, E, F);
                        //            double toi = std::numeric_limits<double>::infinity(); // output
                        //            const bool are_colliding = candidate.ccd(
                        //                dof0,
                        //                dof, //
                        //                toi);

                        //        }
                        //    });
                        //ccd_timer.stop();
                        //ccd_time = ccd_timer.getElapsedTime();
                        //printToMatlab("tight inclusion ccd_time : " + std::to_string(ccd_time) + "s \n");
                        //
                        //ccd_timer.start();
                        //ipc::has_intersections(collision_mesh, V, std::make_shared<ipc::SweepAndPrune>());
                        //ccd_timer.stop();
                        //ccd_time = ccd_timer.getElapsedTime();
                        //printToMatlab("dcd time : " + std::to_string(ccd_time) + "s \n");
                        
                    }
                }
                if (is_step_valid)
                    V_prev = V;
				collision_timer.stop();
				sim_stat.collision_time += collision_timer.getElapsedTime();
                sim_stat.func_eval++;

                if (outputs.size() > 0) {
                    f = nl_problem(tmp_sol);
                    if (!is_step_valid) {
                        //f = std::numeric_limits<double>::infinity();
                        f = std::numeric_limits<double>::quiet_NaN();
                    }
                    outputs[0] = factory.createScalar<double>(f);
                }
                if (outputs.size() > 1) {
                    nl_problem.gradient(tmp_sol, grad);
                    outputs[1] = eigenToMatlab(grad);
                }
                if (outputs.size() > 2) {
                    //Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
                    sim_stat.hessian_eval++;
                    Eigen::SparseMatrix<double> hessian;
                    if (is_step_valid) {
                        //logger().debug("Grad Norm: {:g}", grad.norm());
                        if (grad.lpNorm<Eigen::Infinity>() > 1e-0) 
                        {
                            try {
                                nl_problem.set_project_to_psd(true);
                                nl_problem.hessian(tmp_sol, hessian);
                            }
                            catch (const std::runtime_error& e) {
                                // Code to handle the specific exception type
                                logger().info("Invalid hessian, return 0 hessian");
                                hessian.setZero();
                            }
                        }
                        else 
                        {
                            logger().info("Switch to unprojected hessian");
                            nl_problem.set_project_to_psd(false);
                            nl_problem.hessian(tmp_sol, hessian);

                            igl::Timer factorization_timer;
                            factorization_timer.start();
                            //Eigen::CholmodSupernodalLLT<Eigen::SparseMatrix<double>> solver(hessian);
                            try {
                                auto solver = polysolve::linear::Solver::create("Eigen::CholmodSupernodalLLT", "");
                                //auto solver = polysolve::linear::Solver::create("Eigen::PardisoLDLT", "");
                                solver->analyze_pattern(hessian, hessian.rows());
                                solver->factorize(hessian);
                            }
                            catch (const std::runtime_error& e) {
                                logger().info("hessian is not positive definite");
                                try {
                                    nl_problem.set_project_to_psd(true);
                                    nl_problem.hessian(tmp_sol, hessian);
                                }
                                catch (const std::runtime_error& e) {
                                    // Code to handle the specific exception type
                                    logger().info("Invalid hessian, return 0 hessian");
                                    hessian.setZero();
                                }
							}
                            factorization_timer.stop();
                            sim_stat.extra_factorizing_time += factorization_timer.getElapsedTime();
                            //if (params["solver_info"] != "Success") {
                            //    logger().info("hessian is not positive definite");
                            //    hessian.setZero();
                            //    try {
                            //        nl_problem.set_project_to_psd(true);
                            //        nl_problem.hessian(tmp_sol, hessian);
                            //    }
                            //    catch (const std::runtime_error& e) {
                            //        // Code to handle the specific exception type
                            //        logger().info("Invalid hessian, return 0 hessian");
                            //        hessian.setZero();
                            //    }
                            //}
                        }
                        //nl_problem.set_project_to_psd(false);
                    }
                    outputs[2] = eigenSparseToMatlab(hessian);
                    json solver_info;
                    nl_problem.post_step(polysolve::nonlinear::PostStepData(1, solver_info, tmp_sol, grad));
                }

                // Stop the timer
                timer.stop();

                // Get the elapsed time in seconds (as a double)
                double elapsed_time = timer.getElapsedTime();
                sim_stat.objective_time += elapsed_time;
                //printToMatlab("Time for eval_f : " + std::to_string(elapsed_time) + " seconds\n");
            }
            // --------------------------------------------------------------
            // COMMAND: EVAL_C
            // Usage: [f,g, h] = polyfem_problem_mex('eval_c', handle, sol, sol_reduced)
            // --------------------------------------------------------------
            else if (cmd == "eval_c") {
                Eigen::VectorXd tmp_sol;
                Eigen::MatrixXd sol;

                // Accept Inputs 
                if (inputs.size() == 4 && inputs[2].getType() == ArrayType::DOUBLE && inputs[3].getType() == ArrayType::DOUBLE) {
                    matlabToEigen(inputs[2], sol);
                    matlabToEigen(inputs[3], tmp_sol);
                }
                else {
                    throw std::runtime_error("Invalid input arguments. Usage: [f,g,h] = polyfem_problem_mex('eval_c', handle, sol_reduced)");
                }

                igl::Timer timer;
                // Start the timer
                timer.start();
                //printToMatlab("Start eval_c... \n");
#ifdef USE_PERSISTANT_CANDIDATES
                polyfem::solver::NLProblem& nl_problem = *(state->solve_data.nl_problem);
                const ipc::CollisionMesh& collision_mesh = state->collision_mesh;
                Eigen::MatrixXd V = collision_mesh.displace_vertices(utils::unflatten(nl_problem.reduced_to_full(tmp_sol), collision_mesh.dim()));
                Eigen::MatrixXd V0 = collision_mesh.displace_vertices(utils::unflatten(sol, collision_mesh.dim()));
                assert(V.rows() == collision_mesh.num_vertices());
                assert(V0.rows() == collision_mesh.num_vertices());
                const int ndof = collision_mesh.num_vertices() * collision_mesh.dim();
                const int nc = candidates->size();
                const Eigen::MatrixXi& E = collision_mesh.edges();
                const Eigen::MatrixXi& F = collision_mesh.faces();
                Eigen::VectorXd c_parallel = Eigen::VectorXd::Zero(nc, 1);
                Eigen::MatrixXd grad_parallel = Eigen::MatrixXd::Zero(tmp_sol.size(), nc);
                std::vector<Eigen::SparseMatrix<double>> hessian_parallel(nc);
				ipc::Candidates& collisions = *candidates;
                
                // A. Parallel Computation
                tbb::parallel_for(tbb::blocked_range<size_t>(0, nc),
                    [&](const tbb::blocked_range<size_t>& r) {
                        for (size_t i = r.begin(); i != r.end(); i++) {
                            double toi = 0;
                            ipc::VectorMax12d dof = collisions[i].dof(V, E, F);
                            ipc::VectorMax12d dof0 = collisions[i].dof(V0, E, F);
                            double d = sqrt(collisions[i].compute_distance(dof));
                            if (!collisions[i].ccd(dof0, dof, toi)) {
                                d = -d;
                            }
                            if (d > 0) {
                                c_parallel[i] = d + 5e-4;
                                if (outputs.size() > 1) {
                                    auto v_ids = collisions[i].vertex_ids(E, F);
                                    Eigen::MatrixXd grad_full = Eigen::MatrixXd::Zero(ndof, 1);
                                    ipc::local_gradient_to_global_gradient(collisions[i].compute_distance_gradient(dof) / (2 * d), v_ids, collision_mesh.dim(), grad_full);
                                    grad_parallel.col(i) = nl_problem.full_to_reduced_grad(grad_full);
                                }
                                if (outputs.size() > 2) {
                                    // Compute Local Hessian & Gradient (needed for barrier Hessian)
                                    ipc::VectorMax12d local_grad = collisions[i].compute_distance_gradient(dof);
                                    ipc::MatrixMax12d local_hess = collisions[i].compute_distance_hessian(dof);
                                    std::vector<Eigen::Triplet<double>> hessian_triplets;
                                    local_hess = local_hess / (2 * d) - (local_grad * local_grad.transpose()) / (4 * std::pow(d, 3));

                                    auto v_ids = collisions[i].vertex_ids(E, F);
                                    ipc::local_hessian_to_global_triplets(
                                        ipc::project_to_psd(local_hess, ipc::PSDProjectionMethod::NONE),
                                        v_ids, collision_mesh.dim(), hessian_triplets
                                    );
                                    hessian_parallel[i].resize(ndof, ndof);
                                    hessian_parallel[i].setFromTriplets(hessian_triplets.begin(), hessian_triplets.end());
                                    nl_problem.full_hessian_to_reduced_hessian(hessian_parallel[i]);
                                }
                            }
                            else {
                                hessian_parallel[i].resize(ndof, ndof);
                                nl_problem.full_hessian_to_reduced_hessian(hessian_parallel[i]);
                            }
                        }
                    }
                );
                if (outputs.size() > 0)
                    outputs[0] = eigenToMatlab(c_parallel);
                if (outputs.size() > 1) {
                    outputs[1] = eigenToMatlab(grad_parallel);
                }
                if (outputs.size() > 2) {
                    CellArray cellContainer = factory.createCellArray({ 1, static_cast<unsigned long long>(nc) });
                    for (size_t i = 0; i < nc; i++) {
                        cellContainer[i] = eigenSparseToMatlab(hessian_parallel[i]);
                    }
                    outputs[2] = cellContainer;
                }
#else
                //ipc::Candidates candidates;
				ipc::Candidates& collisions = *candidates;
                double dhat = polyfem::Units::convert(state->args["contact"]["dhat"], state->units.length());
                polyfem::solver::NLProblem& nl_problem = *(state->solve_data.nl_problem);
                const ipc::CollisionMesh& collision_mesh = state->collision_mesh;
                Eigen::MatrixXd V = collision_mesh.displace_vertices(utils::unflatten(nl_problem.reduced_to_full(tmp_sol), collision_mesh.dim()));
                Eigen::MatrixXd V0 = collision_mesh.displace_vertices(utils::unflatten(sol, collision_mesh.dim()));
                const int ndof = collision_mesh.num_vertices() * collision_mesh.dim();
                assert(V.rows() == collision_mesh.num_vertices());
                assert(V0.rows() == collision_mesh.num_vertices());
                double c_parallel = 0;
                Eigen::VectorXd grad_parallel = Eigen::VectorXd::Zero(tmp_sol.size());
                Eigen::SparseMatrix<double> hessian_parallel;
                std::vector<Eigen::Triplet<double>> constraint_hessian_triplets_parallel;

                Eigen::VectorXd dsol = tmp_sol - sol_ls_0;
                Eigen::VectorXd ls_direction = (sol_ls_1 - sol_ls_0).normalized();
                bool is_linesearch = (abs(ls_direction.dot(dsol) - dsol.norm()) < 1e-8) && ((sol_ls_1 - sol_ls_0).norm() > 1e-8);
                //printToMatlab("Is current step in line search? " + std::to_string(is_linesearch) + ", (tmp_sol - sol_ls_0).norm: " + std::to_string(dsol.norm()) + ", (sol_ls_1 - sol_ls_0).norm: " + std::to_string((sol_ls_1 - sol_ls_0).norm()) + ", ls_direction.dot(dsol): " + std::to_string(ls_direction.dot(dsol)) + '\n');
                printToMatlab("Is current step in line search? " + std::to_string(is_linesearch) + ", tmp_sol: (" + eigenToString(tmp_sol.head(3).transpose()) + "), sol_ls_1: (" + eigenToString(sol_ls_1.head(3).transpose()) + "), sol_ls_0: (" + eigenToString(sol_ls_0.head(3).transpose()) + ")\n");
                double t_ls = 1.0;
                //is_linesearch = false;
                if (is_linesearch) 
					t_ls = dsol.norm() / (sol_ls_1 - sol_ls_0).norm();
                else{
                    collisions.clear();
                    collisions.build(collision_mesh, V0, V, dhat / 2, std::make_shared<ipc::SweepAndPrune>());
                    sol_ls_0 = sol_ls_1;
					sol_ls_1 = tmp_sol;
                    t_list.clear();
                }
                /*
                //printToMatlab("Candidate size : " + std::to_string(candidates.size()) + " \n");
				if (candidates.size() > 5000) {
                    printToMatlab("V0: " + eigenToString(V0.topLeftCorner(16,3).transpose()) + "\n");
                    printToMatlab("V: " + eigenToString(V.topLeftCorner(16, 3).transpose()) + "\n");
                    printToMatlab("tmp_sol: " + eigenToString(tmp_sol.transpose()) + "\n");
                }
                */
                int ccd_count = 0;
                if (!collisions.empty()) {
                    using TripletVector = std::vector<Eigen::Triplet<double>>;
                    tbb::combinable<double> c_storage;
                    tbb::combinable<Eigen::VectorXd> grad_storage(Eigen::VectorXd::Zero(tmp_sol.size()));
                    tbb::enumerable_thread_specific<TripletVector> hess_storage;
                    const Eigen::MatrixXi& E = collision_mesh.edges();
                    const Eigen::MatrixXi& F = collision_mesh.faces();
					ipc::TightInclusionCCD tight_ccd(1e-6, 1e8, 1.0);
                    Eigen::MatrixXd V_ls_0 = collision_mesh.displace_vertices(utils::unflatten(nl_problem.reduced_to_full(sol_ls_0), collision_mesh.dim()));
                    //printToMatlab("Mesh statistics, vertices number: " + std::to_string(V.size() / 3) + ", edge number: " + std::to_string(E.size() / 2) + ", face number : " + std::to_string(F.size() / 3) + '\n');

                    for (size_t i = 0; i < collisions.size(); i++) {
                        double toi = 0;
                        bool is_colliding = false;
                        ipc::VectorMax12d dof = collisions[i].dof(V, E, F);
                        ipc::VectorMax12d dof0 = collisions[i].dof(V0, E, F);
                        ipc::VectorMax12d dof_prev = collisions[i].dof(V_prev, E, F);
						double d_prev = sqrt(collisions[i].compute_distance(dof_prev));
                        double d = sqrt(collisions[i].compute_distance(dof));
                        ipc::VectorMax12d local_grad_prev = collisions[i].compute_distance_gradient(dof_prev);
                        ipc::VectorMax12d local_grad = collisions[i].compute_distance_gradient(dof);
						double c_approx = -d_prev + local_grad_prev.dot(dof - dof_prev) / (2 * d_prev);
                        if (d < 1e-6) {
                            std::stringstream d_string;
                            d_string << std::scientific << d;
                            printToMatlab("ccd candidate: " + std::to_string(i) + ", dof0: " + eigenToString(collisions[i].dof(V_ls_0, E, F).transpose()) + ", dof : " + eigenToString(dof.transpose()) + "\n");
                            printToMatlab("d: " + d_string.str() + ", grad: " + eigenToString(local_grad.transpose()) + "\n");
                        }

                        /*
                        printToMatlab("ccd candidate: " + std::to_string(i) + ", dof0: " + eigenToString(collisions[i].dof(V_ls_0, E, F).transpose()) + ", dof : " + eigenToString(dof.transpose()) + "\n");
                        igl::Timer ccd_timer;
                        ccd_timer.start();
                        if(is_linesearch){
                            is_colliding = (t_ls > t_list[i]);
                        }
                        else {
                            is_colliding = collisions[i].ccd(collisions[i].dof(V_ls_0, E, F), dof, toi, 0.0, 1.0, tight_ccd);
                            t_list.push_back(toi);
						}
                        ccd_timer.stop();
                        double ccd_time = ccd_timer.getElapsedTime();
                        printToMatlab("ccd_time : " + std::to_string(ccd_time) + "s \n");

                        
                        bool ccd_is_colliding = collisions[i].ccd(dof0, dof, toi, 0.0, 1.0, tight_ccd);
						if (is_colliding != ccd_is_colliding)
                            printToMatlab("Warning: CCD result mismatch at candidate " + std::to_string(i) + ", is_linesearch: " + std::to_string(is_linesearch) + ", t_ls: " + std::to_string(t_ls) + ", t_list[i]: " + std::to_string(t_list[i]) + ", toi: " + std::to_string(toi) + '\n');
						is_colliding = ccd_is_colliding;
                        */

                        if (abs(d - abs(c_approx)) < 1e-3 && d > 1e-2) {
                            is_colliding = !std::signbit(c_approx);
                            //printToMatlab("ccd candidate " + std::to_string(i) + ", d0: " + std::to_string(d0) + ", d: " + std::to_string(d) + ", c_approx: " + std::to_string(c_approx) + '\n');
                        }
                        else 
                        //if(d < 1e-5)
                        {
                            if (i < collisions.vv_candidates.size() + collisions.ev_candidates.size())
                                continue;
                            else if (i < collisions.vv_candidates.size() + collisions.ev_candidates.size() + collisions.ee_candidates.size())
                                is_colliding = doubleccd_edge_edge(dof0, dof);
                            else if (i < collisions.size())
                                is_colliding = doubleccd_vertex_face(dof0, dof);
                            else
                                polyfem::log_and_throw_error("Index out of candidates range.");
                            ccd_count++;
                        }

                        if (!is_colliding)
                            d = -d;

                        if (d > -1e-4) {
                            c_parallel += d + 1e-4;
                            if (outputs.size() > 1) {
                                auto v_ids = collisions[i].vertex_ids(E, F);
                                Eigen::MatrixXd grad_full = Eigen::MatrixXd::Zero(ndof, 1);
                                ipc::local_gradient_to_global_gradient(local_grad / (2 * d), v_ids, collision_mesh.dim(), grad_full);
                                grad_parallel += nl_problem.full_to_reduced_grad(grad_full);
                            }
                            if (outputs.size() > 2) {
                                // Compute Local Hessian & Gradient (needed for barrier Hessian)
                                ipc::MatrixMax12d local_hess = collisions[i].compute_distance_hessian(dof);
                                local_hess = local_hess / (2 * d) - (local_grad * local_grad.transpose()) / (4 * std::pow(d, 3));

                                auto v_ids = collisions[i].vertex_ids(E, F);
                                ipc::local_hessian_to_global_triplets(
                                    ipc::project_to_psd(local_hess, ipc::PSDProjectionMethod::CLAMP),
                                    v_ids, collision_mesh.dim(), constraint_hessian_triplets_parallel
                                );
                            }
                        }
					}
                    /*
                    // A. Parallel Computation
                    tbb::parallel_for(tbb::blocked_range<size_t>(0, candidates.size()),
                        [&](const tbb::blocked_range<size_t>& r) {
                            for (size_t i = r.begin(); i != r.end(); i++) {
                                double toi = 0;
								bool is_colliding = false;
                                ipc::VectorMax12d dof = candidates[i].dof(V, E, F);
                                ipc::VectorMax12d dof0 = candidates[i].dof(V0, E, F);
                                double d = sqrt(candidates[i].compute_distance(dof));
								d = (dof0 - dof).norm();

                                if (i < candidates.vv_candidates.size() + candidates.ev_candidates.size()) {
                                    is_colliding = candidates[i].ccd(dof0, dof, toi);
                                }
                                else if (i < candidates.vv_candidates.size() + candidates.ev_candidates.size() + candidates.ee_candidates.size()) {
                                    is_colliding = doubleccd_edge_edge(dof0, dof);
                                }
                                else if (i < candidates.size()) {
                                    is_colliding = doubleccd_vertex_face(dof0, dof);
                                }
                                else
                                {
                                    polyfem::log_and_throw_error("Index out of candidates range.");
								}

                                if (!is_colliding) {
                                    d = -d;
                                }
                                if (d >= -1-5) {
                                    c_storage.local() = d;
                                    if (outputs.size() > 1) {
                                        auto v_ids = candidates[i].vertex_ids(E, F);
                                        Eigen::MatrixXd grad_full = Eigen::MatrixXd::Zero(ndof, 1);
                                        ipc::local_gradient_to_global_gradient(candidates[i].compute_distance_gradient(dof) / (2 * d), v_ids, collision_mesh.dim(), grad_full);
                                        grad_storage.local() = nl_problem.full_to_reduced_grad(grad_full);
                                    }
                                    if (outputs.size() > 2) {
                                        // Compute Local Hessian & Gradient (needed for barrier Hessian)
                                        ipc::VectorMax12d local_grad = candidates[i].compute_distance_gradient(dof);
                                        ipc::MatrixMax12d local_hess = candidates[i].compute_distance_hessian(dof);
                                        local_hess = local_hess / (2 * d) - (local_grad * local_grad.transpose()) / (4 * std::pow(d, 3));

                                        auto v_ids = candidates[i].vertex_ids(E, F);
                                        ipc::local_hessian_to_global_triplets(
                                            ipc::project_to_psd(local_hess, ipc::PSDProjectionMethod::NONE),
                                            v_ids, collision_mesh.dim(), hess_storage.local()
                                        );
                                    }
                                }

                            }
                        }
                    );

                    // B. Fast Assembly (Offset + Copy)
					c_parallel = c_storage.combine(std::plus<double>());
                    grad_parallel = grad_storage.combine(
                        [](const Eigen::VectorXd& a, const Eigen::VectorXd& b) {
                            return a + b;
                        }
					);

                    // 1. Calculate offsets
                    std::vector<size_t> offsets(hess_storage.size());
                    size_t total_triplets = 0;
                    size_t idx = 0;
                    for (const auto& local : hess_storage) {
                        offsets[idx++] = total_triplets;
                        total_triplets += local.size();
                    }

                    // 2. Single allocation
                    size_t start_offset = constraint_hessian_triplets_parallel.size();
                    constraint_hessian_triplets_parallel.resize(start_offset + total_triplets);

                    // 3. Parallel Copy
                    tbb::parallel_for(size_t(0), hess_storage.size(), [&](size_t i) {
                        auto it = hess_storage.begin();
                        std::advance(it, i);
                        const auto& local = *it;

                        std::copy(
                            local.begin(),
                            local.end(),
                            constraint_hessian_triplets_parallel.begin() + start_offset + offsets[i]
                        );
                        });
                    */
                }
                
				if (c_parallel <= 0)
                    V_prev = V;

                if (outputs.size() > 0)
                    outputs[0] = factory.createScalar<double>(c_parallel);
                if (outputs.size() > 1) {
                    outputs[1] = eigenToMatlab(grad_parallel);
                }
                if (outputs.size() > 2) {
                    hessian_parallel.resize(ndof, ndof);
                    hessian_parallel.setFromTriplets(constraint_hessian_triplets_parallel.begin(), constraint_hessian_triplets_parallel.end());
                    nl_problem.full_hessian_to_reduced_hessian(hessian_parallel);

                    CellArray cellContainer = factory.createCellArray({ 1, 1 });
                    cellContainer[0] = eigenSparseToMatlab(hessian_parallel);
                    outputs[2] = cellContainer;
                }
#endif
                // Stop the timer
                timer.stop();

                // Get the elapsed time in seconds (as a double)
                double elapsed_time = timer.getElapsedTime();
                if (elapsed_time > 1) {
                    printToMatlab("Total Time for eval_c : " + std::to_string(elapsed_time) + " seconds, Contact Number:" + std::to_string(collisions.size()) + ", double CCD Number : " + std::to_string(ccd_count) +'\n');
                }

                /*
                for (size_t i = 0; i < collisions.size(); i++) {
                    double toi = 0;
                    ipc::VectorMax12d dof = collisions[i].dof(V, E, F);
                    ipc::VectorMax12d dof0 = collisions[i].dof(V0, E, F);
                    double d = sqrt(collisions[i].compute_distance(dof));
                    if (!collisions[i].ccd(dof0, dof, toi)) {
                        d = -d;
                    }

                    //if (abs(d) < 1e-6) {
                    //    c_parallel[i] = d;
                    //    if (outputs.size() > 1) {
                    //        auto v_ids = collisions[i].vertex_ids(E, F);
                    //        Eigen::MatrixXd grad_full = Eigen::MatrixXd::Zero(ndof, 1);
                    //        ipc::VectorMax12d local_grad = dof - dof0;
                    //        for (size_t j = 0; j < local_grad.size(); j += 3) {
                    //            local_grad.segment<3>(j).normalize();
                    //        }
                    //        ipc::local_gradient_to_global_gradient(local_grad, v_ids, collision_mesh.dim(), grad_full);
                    //        grad_parallel.col(i) = nl_problem.full_to_reduced_grad(grad_full);
                    //        printToMatlab("Collision" + std::to_string(i) + " d: " + std::to_string(c_parallel[i]) + ", gradient: " + eigenToString(grad_parallel.col(i).transpose()) + "\n");
                    //    }
                    //    if (outputs.size() > 2) {
                    //        ipc::VectorMax12d local_vector = dof - dof0;
                    //        ipc::MatrixMax12d local_hess = collisions[i].compute_distance_hessian(dof);
                    //        local_hess.setZero();
                    //        for (size_t j = 0; j < local_hess.cols(); j += 3) {
                    //            double vector_norm = local_vector.segment<3>(j).norm();
                    //            local_hess.block<3,3>(j,j) = (Eigen::Matrix3d::Identity() / vector_norm - (local_vector.segment<3>(j) * local_vector.segment<3>(j).transpose()) / pow(vector_norm, 3));
                    //        }
                    //        std::vector<Eigen::Triplet<double>> hessian_triplets;


                    //        auto v_ids = collisions[i].vertex_ids(E, F);
                    //        ipc::local_hessian_to_global_triplets(
                    //            ipc::project_to_psd(local_hess, ipc::PSDProjectionMethod::NONE),
                    //            v_ids, collision_mesh.dim(), hessian_triplets
                    //        );
                    //        hessian_parallel[i].resize(ndof, ndof);
                    //        hessian_parallel[i].setFromTriplets(hessian_triplets.begin(), hessian_triplets.end());
                    //        nl_problem.full_hessian_to_reduced_hessian(hessian_parallel[i]);
                    //    }
                    //    continue;
                    //}

                    if (d > 0) {
                        c_parallel[i] = d + 5e-4;
                        if (outputs.size() > 1) {
                            auto v_ids = collisions[i].vertex_ids(E, F);
                            Eigen::MatrixXd grad_full = Eigen::MatrixXd::Zero(ndof, 1);
                            ipc::local_gradient_to_global_gradient(collisions[i].compute_distance_gradient(dof) / (2 * d), v_ids, collision_mesh.dim(), grad_full);
                            //ipc::local_gradient_to_global_gradient(sign * collisions[i].compute_distance_gradient(dof), v_ids, collision_mesh.dim(), grad_full);
                            grad_parallel.col(i) = nl_problem.full_to_reduced_grad(grad_full);
                        }
                        if (outputs.size() > 2) {
                            // Compute Local Hessian & Gradient (needed for barrier Hessian)
                            ipc::VectorMax12d local_grad = collisions[i].compute_distance_gradient(dof);
                            ipc::MatrixMax12d local_hess = collisions[i].compute_distance_hessian(dof);
                            std::vector<Eigen::Triplet<double>> hessian_triplets;
                            //if (i == 0)
                            //    printToMatlab("Collision 0 dof: " + eigenToString(local_hess) + "\n");
                            //local_hess = sign * local_hess;
                            local_hess = local_hess / (2 * d) - (local_grad * local_grad.transpose()) / (4 * std::pow(d, 3));

                            auto v_ids = collisions[i].vertex_ids(E, F);
                            ipc::local_hessian_to_global_triplets(
                                ipc::project_to_psd(local_hess, ipc::PSDProjectionMethod::NONE),
                                v_ids, collision_mesh.dim(), hessian_triplets
                            );
                            hessian_parallel[i].resize(ndof, ndof);
                            hessian_parallel[i].setFromTriplets(hessian_triplets.begin(), hessian_triplets.end());
                            nl_problem.full_hessian_to_reduced_hessian(hessian_parallel[i]);
                        }
                    }
                    else {
                        hessian_parallel[i].resize(ndof, ndof);
                        nl_problem.full_hessian_to_reduced_hessian(hessian_parallel[i]);
					}
                }
                if (outputs.size() > 0)
                    outputs[0] = eigenToMatlab(c_parallel);
                if (outputs.size() > 1) {
                    outputs[1] = eigenToMatlab(grad_parallel);
                }
				if (outputs.size() > 2) {
                    CellArray cellContainer = factory.createCellArray({ 1, static_cast<unsigned long long>(nc) });
					for (size_t i = 0; i < nc; i++) {
                        cellContainer[i] = eigenSparseToMatlab(hessian_parallel[i]);
                    }
                    outputs[2] = cellContainer;
                }
                
                // Stop the timer
                timer.stop();

                // Get the elapsed time in seconds (as a double)
                double elapsed_time = timer.getElapsedTime();
                if (elapsed_time > 1)
                    printToMatlab("Time for eval_c : " + std::to_string(elapsed_time) + " seconds\n");
                */

                /*
                double c = 0.0;
                Eigen::VectorXd grad_full = Eigen::VectorXd::Zero(collision_mesh.num_vertices() * 3, 1);
                Eigen::SparseMatrix<double> hessian;
                std::vector<Eigen::Triplet<double>> hessian_triplets;

                if (!candidates.empty())
                {
                    const Eigen::MatrixXi& E = collision_mesh.edges();
                    const Eigen::MatrixXi& F = collision_mesh.faces();

                    
                    for (size_t i = 0; i < candidates.size(); i++)
                    {
                        double toi = 0;
                        bool is_colliding = candidates[i].ccd(candidates[i].dof(V0, E, F), candidates[i].dof(V, E, F), toi);
                        if (is_colliding) {
                            double d = sqrt(candidates[i].compute_distance(candidates[i].dof(V, E, F)));
                            c += d + 1e-3;
                            if (outputs.size() > 1) {
                                auto v_ids = candidates[i].vertex_ids(E, F);
                                ipc::local_gradient_to_global_gradient(candidates[i].compute_distance_gradient(candidates[i].dof(V, E, F)) / (2 * d), v_ids, collision_mesh.dim(), grad_full);
                            }
                            if (outputs.size() > 2) {
                                auto v_ids = candidates[i].vertex_ids(E, F);
                                ipc::VectorMax12d local_grad = candidates[i].compute_distance_gradient(candidates[i].dof(V, E, F));
                                ipc::MatrixMax12d local_hess = candidates[i].compute_distance_hessian(candidates[i].dof(V, E, F));
                                local_hess = local_hess / (2 * d) - (local_grad * local_grad.transpose()) / (4 * std::pow(d, 3));
                                ipc::local_hessian_to_global_triplets(ipc::project_to_psd(local_hess, ipc::PSDProjectionMethod::ABS), v_ids, collision_mesh.dim(), hessian_triplets);
                                //ipc::local_hessian_to_global_triplets(local_hess, v_ids, collision_mesh.dim(), hessian_triplets);
                            }
                        }
                    }
                    
                }

                //printToMatlab("thread num: " + std::to_string(tbb::this_task_arena::max_concurrency()) + '\n');
                //if (abs(c - c_parallel) > 1e-8)
                //    printToMatlab("c_diff: " + std::to_string(abs(c - c_parallel)) + '\n');
                //if ((grad_full - grad_parallel).norm() > 1e-8)
                //    printToMatlab("grad_diff_norm: " + std::to_string((grad_full - grad_parallel).norm()) + '\n');

                if (outputs.size() > 0)
                    outputs[0] = factory.createScalar<double>(c_parallel);
                if (outputs.size() > 1) {
                    Eigen::VectorXd grad = nl_problem.full_to_reduced_grad(grad_parallel);
                    outputs[1] = eigenToMatlab(grad);
                }
                */
            }
            // --------------------------------------------------------------
			// COMMAND: EVAL_H_MULT
            // Usage: Hv = polyfem_problem_mex('eval_h_mult', handle, sol, sol_reduced, lambda, v)
            // --------------------------------------------------------------
            else if (cmd == "eval_h_mult") {
                Eigen::MatrixXd sol;
                Eigen::VectorXd tmp_sol;
                Eigen::VectorXd l_ineq, l_eq, l_lb, l_ub;
                Eigen::VectorXd v;
                igl::Timer timer;
                double prepare_time = 0;
                double f_hessian_time = 0;
                double collision_time = 0;
                double c_hessian_time = 0;
				double assembly_time = 0;
                // Start the timer
                timer.start();

                double dhat = polyfem::Units::convert(state->args["contact"]["dhat"], state->units.length());
				//std::vector<Eigen::Triplet<double>> constraint_hessian_triplets;
                //std::vector<Eigen::Triplet<double>> constraint_hessian_triplets_parallel;

                // Accept Inputs 
                if (inputs.size() == 6 &&
                    inputs[2].getType() == ArrayType::DOUBLE &&
                    inputs[3].getType() == ArrayType::DOUBLE &&
					inputs[4].getType() == ArrayType::STRUCT &&
                    inputs[5].getType() == ArrayType::DOUBLE)
                     {
                    matlabToEigen(inputs[2], sol);
                    matlabToEigen(inputs[3], tmp_sol);
                    StructArray lambda = inputs[4];
                    extractLambdaField(lambda, "ineqnonlin", l_ineq);
                    extractLambdaField(lambda, "eqnonlin", l_eq);
                    extractLambdaField(lambda, "lower", l_lb);
                    extractLambdaField(lambda, "upper", l_ub);
                    matlabToEigen(inputs[5], v);
                }
                else {
                    throw std::runtime_error("Usage: Hv = polyfem_problem_mex('eval_h_mult', handle, sol, sol_reduced, lambda, v)");
                }

                assert(state->solve_data.nl_problem != nullptr);
                polyfem::solver::NLProblem& nl_problem = *(state->solve_data.nl_problem);
                Eigen::SparseMatrix<double> hessian_f;
                Eigen::VectorXd Hv;
                //tbb::combinable<Eigen::VectorXd> Hc_v(Eigen::VectorXd::Zero(tmp_sol.size()));

                timer.stop();
                prepare_time = timer.getElapsedTime();
				timer.start();

                nl_problem.hessian(tmp_sol, hessian_f);
                Hv = hessian_f * v;

				timer.stop();
				f_hessian_time = timer.getElapsedTime();

#ifdef USE_PERSISTANT_CANDIDATES
                if (candidates->size() > 1) {
                    const ipc::CollisionMesh& collision_mesh = state->collision_mesh;
                    Eigen::MatrixXd V = collision_mesh.displace_vertices(utils::unflatten(nl_problem.reduced_to_full(tmp_sol), collision_mesh.dim()));
                    Eigen::MatrixXd V0 = collision_mesh.displace_vertices(utils::unflatten(sol, collision_mesh.dim()));
                    assert(V.rows() == collision_mesh.num_vertices());
                    assert(V0.rows() == collision_mesh.num_vertices());
                    const Eigen::MatrixXi& E = collision_mesh.edges();
                    const Eigen::MatrixXi& F = collision_mesh.faces();
                    ipc::Candidates& collisions = *candidates;
                    const int ndof = collision_mesh.num_vertices() * collision_mesh.dim();
                    /*
                    // A. Parallel Computation
                    tbb::parallel_for(tbb::blocked_range<size_t>(0, collisions.size()),
                        [&](const tbb::blocked_range<size_t>& r) {
                            for (size_t i = r.begin(); i != r.end(); i++) {
                                double d;
                                double toi = 0;
                                auto dof = collisions[i].dof(V, E, F);
                                // Re-run CCD check for the Hessian phase
                                if (collisions[i].ccd(collisions[i].dof(V0, E, F), dof, toi)) 
                                    d = sqrt(collisions[i].compute_distance(dof));
                                else
									d = -sqrt(collisions[i].compute_distance(dof));

                                // Compute Local Hessian & Gradient (needed for barrier Hessian)
                                ipc::VectorMax12d local_grad = collisions[i].compute_distance_gradient(dof);
                                ipc::MatrixMax12d local_hess = collisions[i].compute_distance_hessian(dof);
                                std::vector<Eigen::Triplet<double>> hessian_triplets;
                                Eigen::SparseMatrix<double> hessian_c;

                                // Barrier formula
                                local_hess = local_hess / (2 * d) - (local_grad * local_grad.transpose()) / (4 * std::pow(d, 3));

                                auto v_ids = collisions[i].vertex_ids(E, F);
                                ipc::local_hessian_to_global_triplets(
                                    ipc::project_to_psd(local_hess, ipc::PSDProjectionMethod::ABS),
                                    v_ids, collision_mesh.dim(), hessian_triplets
                                );
                                hessian_c.resize(ndof, ndof);
                                hessian_c.setFromTriplets(hessian_triplets.begin(), hessian_triplets.end());
                                nl_problem.full_hessian_to_reduced_hessian(hessian_c);
								Hc_v.local() = l_ineq[i] * (hessian_c * v);
                            }
                        }
                    );
					Hv += Hc_v.combine([](const Eigen::VectorXd& a, const Eigen::VectorXd& b) { return a + b; });
                    */
                    for (size_t i = 0; i < collisions.size(); i++) {
                        double toi = 0;
                        ipc::VectorMax12d dof = collisions[i].dof(V, E, F);
                        ipc::VectorMax12d dof0 = collisions[i].dof(V0, E, F);
                        double d = sqrt(collisions[i].compute_distance(dof));
                        // Re-run CCD check for the Hessian phase
                        timer.start();
                        if (!collisions[i].ccd(dof0, dof, toi))
                            d = -d;
                        timer.stop();
                        double tmp_time = timer.getElapsedTime();
                        //if (tmp_time > 1e-1)
                        //	printToMatlab("collision detection time for hessian of constraint " + std::to_string(i) + ": " + std::to_string(tmp_time) + " seconds\n");
                        collision_time += tmp_time;

                        timer.start();
                        //if (abs(d) < 1e-6) {
                        //    ipc::VectorMax12d local_vector = dof - dof0;
                        //    ipc::MatrixMax12d local_hess = collisions[i].compute_distance_hessian(dof);
                        //    local_hess.setZero();
                        //    for (size_t j = 0; j < local_hess.cols(); j += 3) {
                        //        double vector_norm = local_vector.segment<3>(j).norm();
                        //        local_hess.block<3, 3>(j, j) = (Eigen::Matrix3d::Identity() / vector_norm - (local_vector.segment<3>(j) * local_vector.segment<3>(j).transpose()) / pow(vector_norm, 3));
                        //    }
                        //    std::vector<Eigen::Triplet<double>> hessian_triplets;
                        //    Eigen::SparseMatrix<double> hessian_c;

                        //    auto v_ids = collisions[i].vertex_ids(E, F);
                        //    ipc::local_hessian_to_global_triplets(
                        //        ipc::project_to_psd(local_hess, ipc::PSDProjectionMethod::NONE),
                        //        v_ids, collision_mesh.dim(), hessian_triplets
                        //    );
                        //    hessian_c.resize(ndof, ndof);
                        //    hessian_c.setFromTriplets(hessian_triplets.begin(), hessian_triplets.end());
                        //    nl_problem.full_hessian_to_reduced_hessian(hessian_c);
                        //    Hv += l_ineq[i] * (hessian_c * v);
                        //    continue;
                        //}

                        // Compute Local Hessian & Gradient (needed for barrier Hessian)
                        ipc::VectorMax12d local_grad = collisions[i].compute_distance_gradient(dof);
                        ipc::MatrixMax12d local_hess = collisions[i].compute_distance_hessian(dof);
                        std::vector<Eigen::Triplet<double>> hessian_triplets;
                        Eigen::SparseMatrix<double> hessian_c;
                        //local_hess = sign * local_hess;
                        local_hess = local_hess / (2 * d) - (local_grad * local_grad.transpose()) / (4 * std::pow(d, 3));

                        auto v_ids = collisions[i].vertex_ids(E, F);
                        ipc::local_hessian_to_global_triplets(
                            ipc::project_to_psd(local_hess, ipc::PSDProjectionMethod::ABS),
                            v_ids, collision_mesh.dim(), hessian_triplets
                        );
                        hessian_c.resize(ndof, ndof);
                        hessian_c.setFromTriplets(hessian_triplets.begin(), hessian_triplets.end());
                        nl_problem.full_hessian_to_reduced_hessian(hessian_c);
                        Hv += l_ineq[i] * (hessian_c * v);

						timer.stop();
						assembly_time += timer.getElapsedTime();
                    }
					c_hessian_time = collision_time + assembly_time;
                }

                outputs[0] = eigenToMatlab(Hv);

                if (prepare_time + f_hessian_time + c_hessian_time > 1) {
                    printToMatlab("Time for eval_h_mult: " + std::to_string(prepare_time + f_hessian_time + c_hessian_time) + " seconds\n");
                    printToMatlab("Prepare Time: " + std::to_string(prepare_time) + " seconds\n");
                    printToMatlab("f Hessian Time: " + std::to_string(f_hessian_time) + " seconds\n");
                    printToMatlab("c Hessian Time: " + std::to_string(c_hessian_time) + " = " + std::to_string(collision_time) + '+' + std::to_string(assembly_time) + " seconds\n");
                }
#else
                outputs[0] = eigenToMatlab(Hv);
#endif
                /*
				if (l_ineq.size() < 1)
                    throw std::runtime_error("Invalid input arguments. Lambda for nonlinear inequality constraint can't be empty.");

                assert(state.solve_data.nl_problem != nullptr);
                polyfem::solver::NLProblem& nl_problem = *(state->solve_data.nl_problem);
                Eigen::SparseMatrix<double> hessian_f;
                Eigen::SparseMatrix<double> hessian_c;
                Eigen::SparseMatrix<double> hessian_c_parallel;
                Eigen::VectorXd Hv;

                const ipc::CollisionMesh& collision_mesh = state->collision_mesh;
                Eigen::MatrixXd V = collision_mesh.displace_vertices(utils::unflatten(nl_problem.reduced_to_full(tmp_sol), collision_mesh.dim()));
                Eigen::MatrixXd V0 = collision_mesh.displace_vertices(utils::unflatten(sol, collision_mesh.dim()));
                assert(V.rows() == collision_mesh.num_vertices());
                assert(V0.rows() == collision_mesh.num_vertices());
                candidates.build(collision_mesh, V0, V, dhat / 2);


                if (!candidates.empty()) {
                    using TripletVector = std::vector<Eigen::Triplet<double>>;
                    tbb::enumerable_thread_specific<TripletVector> hess_storage;
                    const Eigen::MatrixXi& E = collision_mesh.edges();
                    const Eigen::MatrixXi& F = collision_mesh.faces();

                    // A. Parallel Computation
                    tbb::parallel_for(tbb::blocked_range<size_t>(0, candidates.size()),
                        [&](const tbb::blocked_range<size_t>& r) {

                            TripletVector& local_triplets = hess_storage.local();

                            for (size_t i = r.begin(); i != r.end(); i++) {
                                double toi = 0;
                                // Re-run CCD check for the Hessian phase
                                if (candidates[i].ccd(candidates[i].dof(V0, E, F), candidates[i].dof(V, E, F), toi)) {

                                    auto dof = candidates[i].dof(V, E, F);
                                    double d = sqrt(candidates[i].compute_distance(dof));

                                    // Compute Local Hessian & Gradient (needed for barrier Hessian)
                                    ipc::VectorMax12d local_grad = candidates[i].compute_distance_gradient(dof);
                                    ipc::MatrixMax12d local_hess = candidates[i].compute_distance_hessian(dof);

                                    // Barrier formula
                                    local_hess = local_hess / (2 * d) - (local_grad * local_grad.transpose()) / (4 * std::pow(d, 3));

                                    auto v_ids = candidates[i].vertex_ids(E, F);
                                    ipc::local_hessian_to_global_triplets(
                                        ipc::project_to_psd(local_hess, ipc::PSDProjectionMethod::ABS),
                                        v_ids, collision_mesh.dim(), local_triplets
                                    );
                                }
                            }
                        }
                    );

                    // B. Fast Assembly (Offset + Copy)

                    // 1. Calculate offsets
                    std::vector<size_t> offsets(hess_storage.size());
                    size_t total_triplets = 0;
                    size_t idx = 0;
                    for (const auto& local : hess_storage) {
                        offsets[idx++] = total_triplets;
                        total_triplets += local.size();
                    }

                    // 2. Single allocation
                    size_t start_offset = constraint_hessian_triplets_parallel.size();
                    constraint_hessian_triplets_parallel.resize(start_offset + total_triplets);

                    // 3. Parallel Copy
                    tbb::parallel_for(size_t(0), hess_storage.size(), [&](size_t i) {
                        auto it = hess_storage.begin();
                        std::advance(it, i);
                        const auto& local = *it;

                        std::copy(
                            local.begin(),
                            local.end(),
                            constraint_hessian_triplets_parallel.begin() + start_offset + offsets[i]
                        );
                        });
                }

                hessian_c_parallel.resize(collision_mesh.num_vertices() * 3, collision_mesh.num_vertices() * 3);
                hessian_c_parallel.setFromTriplets(constraint_hessian_triplets_parallel.begin(), constraint_hessian_triplets_parallel.end());
                nl_problem.full_hessian_to_reduced_hessian(hessian_c_parallel);
                */

    //            if (!candidates.empty())
    //            {
    //                const Eigen::MatrixXi& E = collision_mesh.edges();
    //                const Eigen::MatrixXi& F = collision_mesh.faces();

    //                for (size_t i = 0; i < candidates.size(); i++)
    //                {
    //                    double toi = 0;
    //                    bool is_colliding = candidates[i].ccd(candidates[i].dof(V0, E, F), candidates[i].dof(V, E, F), toi);
    //                    if (is_colliding) {
    //                        double d = sqrt(candidates[i].compute_distance(candidates[i].dof(V, E, F)));
    //                        auto v_ids = candidates[i].vertex_ids(E, F);
    //                        ipc::VectorMax12d local_grad = candidates[i].compute_distance_gradient(candidates[i].dof(V, E, F));
    //                        ipc::MatrixMax12d local_hess = candidates[i].compute_distance_hessian(candidates[i].dof(V, E, F));
    //                        local_hess = local_hess / (2 * d) - (local_grad * local_grad.transpose()) / (4 * std::pow(d, 3));
    //                        ipc::local_hessian_to_global_triplets(ipc::project_to_psd(local_hess, ipc::PSDProjectionMethod::ABS), v_ids, collision_mesh.dim(), constraint_hessian_triplets);
    //                        //ipc::local_hessian_to_global_triplets(local_hess, v_ids, collision_mesh.dim(), constraint_hessian_triplets);
    //                    }
    //                }
    //            }

    //            hessian_c.resize(collision_mesh.num_vertices() * 3, collision_mesh.num_vertices() * 3);
				//hessian_c.setFromTriplets(constraint_hessian_triplets.begin(), constraint_hessian_triplets.end());
    //            nl_problem.full_hessian_to_reduced_hessian(hessian_c);
                //if((hessian_c - hessian_c_parallel).norm() > 1e-6)
                //    polyfem::logger().info("hessian and hessian_parallel are different, norm: {}", (hessian_c - hessian_c_parallel).norm());

                //if (constraint_hessian_triplets_parallel.size() > 0)
                //    Hv =  l_ineq[0] * hessian_c_parallel * v;
                //else {
                //    nl_problem.hessian(tmp_sol, hessian_f);
                //    Hv = hessian_f * v;
                //}

                //nl_problem.hessian(tmp_sol, hessian_f);
                //Hv = (hessian_f + l_ineq[0] * hessian_c_parallel) * v;
                //outputs[0] = eigenToMatlab(Hv);
            }
            // --------------------------------------------------------------
            // COMMAND: SOLVE
            // [sol] = polyfem_problem_mex('solve', handle, sol_reduced)
            // --------------------------------------------------------------
            else if (cmd == "solve") {
                Eigen::VectorXd tmp_sol;
                // Accept Inputs 
                if (inputs.size() == 3 && inputs[2].getType() == ArrayType::DOUBLE) {
                    matlabToEigen(inputs[2], tmp_sol);
                }
                else {
                    throw std::runtime_error("Invalid input arguments. Usage: sol_reduced = polyfem_problem_mex('solve', handle, sol_reduced)");
                }
                igl::Timer solver_timer;
                solver_timer.start();
                assert(state.solve_data.nl_problem != nullptr);
                polyfem::solver::NLProblem& nl_problem = *(state->solve_data.nl_problem);
                try
                {
                    const auto scale = nl_problem.normalize_forms();
                    auto nl_solver = polysolve::nonlinear::Solver::create(
                        state->args["solver"]["nonlinear"],
                        state->args["solver"]["linear"],
                        state->units.characteristic_length() * scale, polyfem::logger());
                    polyfem::logger().debug("Using nl solver.");
                    nl_solver->minimize(nl_problem, tmp_sol);
                    const polysolve::json& solver_info = nl_solver->info();
                    int iterations = solver_info["iterations"].get<int>();
                    sim_stat.hessian_eval += iterations;
                    //minimize(nl_problem, tmp_sol, state->args["solver"]["nonlinear"], state->args["solver"]["linear"], state->units.characteristic_length()* scale, state->collision_mesh);
                }
                catch (const std::runtime_error& e)
                {
                    throw e;
                }
                solver_timer.stop();
                sim_stat.objective_time += solver_timer.getElapsedTime();
                if (outputs.size() > 0)
                    outputs[0] = eigenToMatlab(tmp_sol);
            }
            // --------------------------------------------------------------
            // COMMAND: POST SOLVE
            // [sol] = polyfem_problem_mex('post_solve', handle, sol_reduced, project_hessian)
            // --------------------------------------------------------------
            else if (cmd == "post_solve") {
                Eigen::VectorXd tmp_sol;
                bool project_hessian;
                // Accept Inputs 
                if (inputs.size() == 4 && inputs[2].getType() == ArrayType::DOUBLE && inputs[3].getType() == ArrayType::LOGICAL) {
                    matlabToEigen(inputs[2], tmp_sol);
                    TypedArray<bool> val = inputs[3];
                    project_hessian = val[0];
                }
                else {
                    throw std::runtime_error("Invalid input arguments. Usage: sol_reduced = polyfem_problem_mex('solve', handle, sol_reduced)");
                }
                polyfem::solver::NLProblem& nl_problem = *(state->solve_data.nl_problem);
                nl_problem.set_project_to_psd(project_hessian);
                json solver_info;
                Eigen::VectorXd grad;
                nl_problem.gradient(tmp_sol, grad);
                nl_problem.post_step(polysolve::nonlinear::PostStepData(1, solver_info, tmp_sol, grad));
            }
            // --------------------------------------------------------------
            // COMMAND: SAVE
            // [sol] = polyfem_problem_mex('save', handle, sol_reduced, t)
            // --------------------------------------------------------------
            else if (cmd == "save") {
                Eigen::VectorXd tmp_sol;
                Eigen::MatrixXd sol;
                int t;
                // Accept Inputs 
                if (inputs.size() == 4 && inputs[2].getType() == ArrayType::DOUBLE) {
                    matlabToEigen(inputs[2], tmp_sol);

                    t = getInt(inputs[3]);
                }
                else {
                    throw std::runtime_error("Invalid input arguments. Usage: sol_reduced = polyfem_problem_mex('save', handle, sol_reduced, t)");
                }

                assert(state.solve_data.nl_problem != nullptr);
                polyfem::solver::NLProblem& nl_problem = *(state->solve_data.nl_problem);
                nl_problem.finish();
                sol = nl_problem.reduced_to_full(tmp_sol);
				save(*state, sol, t);

                sim_stat.timer.stop();
                sim_stat.total_time += sim_stat.timer.getElapsedTime();
                logger().info("Current total time: {}s, initial time: {}s, objective funcioin time: {}s, extra factorization time: {}s, collision time: {}s.", sim_stat.total_time, sim_stat.init_time, sim_stat.objective_time, sim_stat.extra_factorizing_time, sim_stat.collision_time);
                logger().info("Current total function evaluations: {}, total hessian evaluations: {}, ccd count: {}", sim_stat.func_eval, sim_stat.hessian_eval, sim_stat.ccd_count);

                if(candidates) {
                    candidates.reset();
                    mexUnlock();
				}

				if (outputs.size() > 0)
				    outputs[0] = eigenToMatlab(sol);
            }
            else {
                throw std::runtime_error("Unknown command: " + cmd);
            }

        }
        catch (const std::exception& e) {
            printToMatlab("Error in polyfem_solve_mex: " + std::string(e.what()) + "\n");
        }

        std::cout.rdbuf(oldOut);
        std::cerr.rdbuf(oldErr);
    }

private:
    void printToMatlab(const std::string& msg) {
        matlabPtr->feval(u"fprintf", 0, std::vector<Array>({
            factory.createScalar("%s"),
            factory.createScalar(msg)
            }));
    }

    void checkArguments(ArgumentList inputs) {
        if (inputs.size() < 2) {
            throw std::runtime_error("Usage: polyfem_solve_mex('command', handle)");
        }
        if (inputs[0].getType() != ArrayType::CHAR) {
            throw std::runtime_error("First argument must be a command string.");
        }
        if (inputs[1].getType() != ArrayType::UINT64) {
            throw std::runtime_error("Second argument must be a uint64 state handle.");
        }
    }
    // Helper to safely extract int from MATLAB array (which handles both Double and Int inputs)
    int getInt(Array in) {
        if (in.getType() == ArrayType::DOUBLE) {
            TypedArray<double> val = in;
            return static_cast<int>(val[0]);
        }
        else if (in.getType() == ArrayType::INT32) {
            TypedArray<int32_t> val = in;
            return static_cast<int>(val[0]);
        }
        else if (in.getType() == ArrayType::UINT32) {
            TypedArray<uint32_t> val = in;
            return static_cast<int>(val[0]);
        }
        else if (in.getType() == ArrayType::INT64) {
            TypedArray<int64_t> val = in;
            return static_cast<int>(val[0]);
        }
        return 0;
    }

    // Helper: MATLAB TypedArray -> Eigen::MatrixXd
    void matlabToEigen(const TypedArray<double>& in, Eigen::MatrixXd& out) {
        auto dims = in.getDimensions();
        out.resize(dims[0], dims[1]);
        std::copy(in.begin(), in.end(), out.data());
    }

    // Helper: MATLAB TypedArray -> Eigen::MatrixXd
    void matlabToEigen(const TypedArray<double>& in, Eigen::VectorXd& out) {
        auto dims = in.getDimensions();
        out.resize(dims[0]);
        std::copy(in.begin(), in.end(), out.data());
    }

    // --------------------------------------------------------------------------
    // HELPER: Extract Field from MATLAB Struct -> Eigen Vector
    // --------------------------------------------------------------------------
    void extractLambdaField(const StructArray & s, const std::string & field, Eigen::VectorXd & out) {
        // MATLAB StructArray can handle field lookup
        // Fix: Removing 'const' here to allow calling begin() on the Range object if needed
        auto fields = s.getFieldNames();
        bool found = false;
        for (const auto& f : fields) { if (f == field) found = true; }

        if (found) {
            Array val = s[0][field];
            // Only extract if it's double and non-empty
            if (val.getType() == ArrayType::DOUBLE && val.getNumberOfElements() > 0) {
                matlabToEigen(val, out);
            }
        }
    }

    // Helper: Eigen::MatrixXd -> MATLAB TypedArray
    TypedArray<double> eigenToMatlab(const Eigen::MatrixXd& in) {
        TypedArray<double> out = factory.createArray<double>({ (size_t)in.rows(), (size_t)in.cols() });
        std::copy(in.data(), in.data() + in.size(), out.begin());
        return out;
    }

    // Helper: Eigen::VectorXd -> MATLAB Array
    TypedArray<double> eigenToMatlab(const Eigen::VectorXd& in) {
        TypedArray<double> out = factory.createArray<double>({ (size_t)in.size(), 1 });
        std::copy(in.data(), in.data() + in.size(), out.begin());
        return out;
    }

    // --------------------------------------------------------------------------
    // HELPER: Eigen::SparseMatrix -> MATLAB SparseArray
    // --------------------------------------------------------------------------
    SparseArray<double> eigenSparseToMatlab(const Eigen::SparseMatrix<double>& in) {
        if (!in.isCompressed()) {
            printToMatlab("Input matrix should be sparse compressed form.\n");
        }

        size_t rows = in.rows();
        size_t cols = in.cols();
        size_t nnz = in.nonZeros();

        buffer_ptr_t<double> data_buff = factory.createBuffer<double>(nnz);
        const double* eigen_val_ptr = in.valuePtr();
        std::copy(eigen_val_ptr, eigen_val_ptr + nnz, data_buff.get());

        buffer_ptr_t<size_t> rows_buff = factory.createBuffer<size_t>(nnz);
        size_t* rows_ptr = rows_buff.get();
        const int* eigen_inner_ptr = in.innerIndexPtr();
        for (size_t i = 0; i < nnz; ++i) {
            rows_ptr[i] = static_cast<size_t>(eigen_inner_ptr[i]);
        }

        buffer_ptr_t<size_t> cols_buff = factory.createBuffer<size_t>(nnz);
        size_t* cols_ptr = cols_buff.get();
        const int* eigen_outer_ptr = in.outerIndexPtr();
        // Iterate over each column 'j'
        for (size_t j = 0; j < cols; ++j) {
            int start = eigen_outer_ptr[j];
            int end = eigen_outer_ptr[j + 1];

            // Fill the column index 'j' for all non-zeros in this column
            for (int k = start; k < end; ++k) {
                cols_ptr[k] = j;
            }
        }

        return factory.createSparseArray(
            { rows, cols },
            nnz,
            std::move(data_buff),
            std::move(rows_buff),
            std::move(cols_buff)
        );
    }

    template <typename Derived>
    static std::string eigenToString(const Eigen::MatrixBase<Derived>& mat) {
        std::stringstream ss;
        ss << mat;
        return ss.str();
    }

    // --------------------------------------------------------------------------
    // HELPER: Double CCD: Vertex face
    // --------------------------------------------------------------------------
    bool doubleccd_vertex_face(
        Eigen::ConstRef<ipc::VectorMax12d> vertices_t0,
        Eigen::ConstRef<ipc::VectorMax12d> vertices_t1) const
    {
        assert(vertices_t0.size() == 12 && vertices_t1.size() == 12);
        doubleccd::vf_pair dt, dtshift;
        dt.x0 = vertices_t0.segment<3>(0);
        dt.x1 = vertices_t0.segment<3>(3);
        dt.x2 = vertices_t0.segment<3>(6);
        dt.x3 = vertices_t0.segment<3>(9);

        dt.x0b = vertices_t1.segment<3>(0);
        dt.x1b = vertices_t1.segment<3>(3);
        dt.x2b = vertices_t1.segment<3>(6);
        dt.x3b = vertices_t1.segment<3>(9);
		double time;
        double err = shift_vertex_face(dt, dtshift, time);
        dt = dtshift;
        return doubleccd::vertexFaceCCD(
            dt.x0, dt.x1, dt.x2, dt.x3, dt.x0b, dt.x1b, dt.x2b, dt.x3b
        );
    }

    // --------------------------------------------------------------------------
    // HELPER: Double CCD: Vertex face
    // --------------------------------------------------------------------------
    bool doubleccd_edge_edge(
        Eigen::ConstRef<ipc::VectorMax12d> vertices_t0,
        Eigen::ConstRef<ipc::VectorMax12d> vertices_t1) const
    {
        assert(vertices_t0.size() == 12 && vertices_t1.size() == 12);
        doubleccd::ee_pair dt, dtshift;
        dt.a0 = vertices_t0.segment<3>(0);
        dt.a1 = vertices_t0.segment<3>(3);
        dt.b0 = vertices_t0.segment<3>(6);
        dt.b1 = vertices_t0.segment<3>(9);

        dt.a0b = vertices_t1.segment<3>(0);
        dt.a1b = vertices_t1.segment<3>(3);
        dt.b0b = vertices_t1.segment<3>(6);
        dt.b1b = vertices_t1.segment<3>(9);
        double time;
        double err = shift_edge_edge(dt, dtshift, time);
        dt = dtshift;
        return doubleccd::edgeEdgeCCD(
            dt.a0, dt.a1, dt.b0, dt.b1, dt.a0b, dt.a1b, dt.b0b, dt.b1b
        );
    }

    // --------------------------------------------------------------------------
    // HELPER: IPC minimizer
    // --------------------------------------------------------------------------
    int minimize(polyfem::solver::NLProblem& objFunc, Eigen::VectorXd& x, const json& solver_params_in, const json& linear_params, const double characteristic_length, ipc::CollisionMesh collision_mesh) {
        constexpr double NaN = std::numeric_limits<double>::quiet_NaN();
        constexpr double Inf = std::numeric_limits<double>::infinity();
        json solver_params = solver_params_in; // mutable copy

        json rules;
        jse::JSE jse;

        jse.strict = true;
        const std::string input_spec = "C:/Users/zyxiong/.cache/CPM/polysolve/4f00/nonlinear-solver-spec.json";
        std::ifstream file(input_spec);

        if (file.is_open())
            file >> rules;
        else
            polysolve::log_and_throw_error(logger(), "unable to open {} rules", input_spec);

        const bool valid_input = jse.verify_json(solver_params, rules);

        if (!valid_input)
            polysolve::log_and_throw_error(logger(), "invalid input json:\n{}", jse.log2str());

        solver_params = jse.inject_defaults(solver_params, rules);
        polysolve::nonlinear::Criteria m_stop;
        polysolve::nonlinear::Criteria m_current;
        polysolve::nonlinear::Status m_status;
        m_current.reset();

        m_stop.xDelta = solver_params["x_delta"];
        m_stop.fDelta = solver_params["advanced"]["f_delta"];
        m_stop.gradNorm = solver_params["grad_norm"];
        m_stop.firstGradNorm = solver_params["first_grad_norm_tol"];
        m_stop.xDeltaDotGrad = -solver_params["advanced"]["derivative_along_delta_x_tol"].get<double>();

        // Make these relative to the characteristic length
        logger().debug("Using a characteristic length of {:g}", characteristic_length);
        m_stop.xDelta *= characteristic_length;
        m_stop.fDelta *= characteristic_length;
        m_stop.gradNorm *= characteristic_length;
        m_stop.firstGradNorm *= characteristic_length;
        // m_stop.xDeltaDotGrad *= characteristic_length;

        m_stop.iterations = solver_params["max_iterations"];
        bool allow_out_of_iterations = solver_params["allow_out_of_iterations"];

        m_stop.fDeltaCount = solver_params["advanced"]["f_delta_step_tol"];

        std::shared_ptr<polysolve::nonlinear::line_search::LineSearch> m_line_search;
        m_line_search = polysolve::nonlinear::line_search::LineSearch::create(solver_params, logger());
        json solver_info = json();
        solver_info["line_search"] = solver_params["line_search"]["method"];
        solver_info["iterations"] = 0;
        m_line_search->use_grad_norm_tol = solver_params["line_search"]["use_grad_norm_tol"];
        m_line_search->use_grad_norm_tol *= characteristic_length;
        Eigen::VectorXd grad = Eigen::VectorXd::Zero(x.rows());
        Eigen::VectorXd delta_x = Eigen::VectorXd::Zero(x.rows());
        double old_energy = NaN;
        objFunc.solution_changed(x);
        solver_info["energy"] = objFunc(x);
        logger().debug(
            "Starting {} with {} solve f_0={:g}. Stopping criteria: iters={:d} Delta f={:g} Grad Norm={:g} Delta x={:g} Delta x Dot Grad={:g}",
            "Newton Solver", m_line_search->name(), objFunc(x), m_stop.iterations, m_stop.fDelta, m_stop.gradNorm, m_stop.xDelta, m_stop.xDeltaDotGrad);
        objFunc.post_step(polysolve::nonlinear::PostStepData(m_current.iterations, solver_info, x, grad));

        auto linear_solver = polysolve::linear::Solver::create(linear_params, logger());
        igl::Timer stop_watch;
        stop_watch.start();
        do
        {
            m_line_search->set_is_final_strategy(false);

            double energy = objFunc(x);
            m_current.fDelta = std::abs(old_energy - energy);

            objFunc.gradient(x, grad);
            m_current.gradNorm = grad.norm();

            // Check convergence without these values to avoid impossible linear solves.
            m_current.xDelta = NaN;
            m_current.xDeltaDotGrad = NaN;
            m_status = checkConvergence(m_stop, m_current);
            if (m_status != polysolve::nonlinear::Status::Continue)
                break;

            Eigen::SparseMatrix<double> hessian;
            objFunc.set_project_to_psd(false);
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
            const double residual = (hessian * delta_x + grad).norm();

            if (std::isnan(residual) || residual > 1e-5 * characteristic_length) {
                logger().debug("Switch to projected Newton Solver");
                m_line_search->set_is_final_strategy(true);
                hessian.setZero();
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
                    logger().debug("Unable to factorize projected Hessian: \"{}\"", err.what());
                    // Eigen::saveMarket(hessian, "problematic_hessian.mtx");
                    return std::nan("");
                }
                linear_solver->solve(-grad, delta_x); // H Δx = -g
            }

            m_current.xDelta = delta_x.norm();
            m_current.xDeltaDotGrad = delta_x.dot(grad);

            if (m_current.gradNorm != 0 && m_current.xDeltaDotGrad >= 0)
            {
                logger().debug("Switch to projected Newton Solver");
                m_line_search->set_is_final_strategy(true);
                hessian.setZero();
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
                    logger().debug("Unable to factorize projected Hessian: \"{}\"", err.what());
                    // Eigen::saveMarket(hessian, "problematic_hessian.mtx");
                    return std::nan("");
                }
                linear_solver->solve(-grad, delta_x); // H Δx = -g
            }

            m_status = checkConvergence(m_stop, m_current);

            if (m_status != polysolve::nonlinear::Status::Continue)
                break;

            logger().debug(
                "[{}][{}] pre LS iter={:d} f={:g} Grad Norm={:g} Deltax norm: {:.16g}",
                "Newton", m_line_search->name(),
                m_current.iterations, energy, m_current.gradNorm, delta_x.norm());

            double rate;
            rate = m_line_search->line_search(x, delta_x, objFunc);
            Eigen::VectorXd x1 = x + rate * delta_x;

            /*
            {
                Eigen::MatrixXd xs;
                Eigen::VectorXd fxs(1);
                matlab::data::TypedArray<bool> fileExists = matlabPtr->feval(u"isfile", factory.createScalar("data_IPC.mat"));
                logger().debug("is file exist? {fileExists[0]}");

                if (fileExists[0]) {
                    // A. Load existing data
                    matlabPtr->eval(u"load('data_IPC.mat', 'xs');");
                    matlab::data::TypedArray<double> xs_matlab = matlabPtr->getVariable(u"xs");
                    matlabToEigen(xs_matlab, xs);
                    xs.conservativeResize(Eigen::NoChange, xs.cols() + 1);
                    xs.col(xs.cols() - 1) = x;

                    matlabPtr->eval(u"load('data_IPC.mat', 'fxs');");
                    matlab::data::TypedArray<double> fxs_matlab = matlabPtr->getVariable(u"fxs");
                    matlabToEigen(fxs_matlab, fxs);
                    fxs.conservativeResize(fxs.size() + 1);
                    fxs(fxs.size() - 1) = energy;

                }
                else {
                    xs = x;
                    fxs << energy;
                }

                matlabPtr->setVariable(u"xs", eigenToMatlab(xs));
                matlabPtr->setVariable(u"fxs", eigenToMatlab(fxs));
                matlabPtr->eval(u"save('data_IPC.mat', 'xs', 'fxs');");
            }
            */

            if (objFunc.after_line_search_custom_operation(x, x1))
                objFunc.solution_changed(x1);
            x = x1;
            old_energy = energy;

            const double step = (rate * delta_x).norm();
            solver_info["energy"] = energy;
            solver_info["iterations"] = m_current.iterations;
            objFunc.post_step(polysolve::nonlinear::PostStepData(m_current.iterations, solver_info, x, grad));
            for (auto form : objFunc.forms()) {
                std::shared_ptr<polyfem::solver::ContactForm> contact_form = std::dynamic_pointer_cast<polyfem::solver::ContactForm>(form);
                if (contact_form != nullptr) {
                    logger().debug("Barrier Stiffness: {:e}", contact_form->barrier_stiffness());
                }
                std::shared_ptr<polyfem::solver::BarrierContactForm> barrier_contact_form = std::dynamic_pointer_cast<polyfem::solver::BarrierContactForm>(form);
                if (barrier_contact_form != nullptr) {
                    logger().debug("Minimum distance: {:e}", barrier_contact_form->collision_set().compute_minimum_distance(collision_mesh, barrier_contact_form->compute_displaced_surface(x)));
                }
            }
            logger().debug(
                "[{}][{}] Current criteria: iters={:d} Delta f={:g} Grad Norm={:g} Delta x={:g} Delta x Dot Grad={:g}",
                "Newton", m_line_search->name(), m_current.iterations, m_current.fDelta, m_current.gradNorm, m_current.xDelta, m_current.xDeltaDotGrad);
            if (objFunc.stop(x))
            {
                m_status = polysolve::nonlinear::Status::ObjectiveCustomStop;
                logger().debug("[{}][{}] Objective decided to stop", "Newton", m_line_search->name());
            }
            m_current.fDeltaCount = (m_current.fDelta < m_stop.fDelta) ? (m_current.fDeltaCount + 1) : 0;
            if (++m_current.iterations >= m_stop.iterations)
                m_status = polysolve::nonlinear::Status::IterationLimit;

        } while (objFunc.callback(m_current, x) && (m_status == polysolve::nonlinear::Status::Continue));

        if (!allow_out_of_iterations && m_status == polysolve::nonlinear::Status::IterationLimit)
            polysolve::log_and_throw_error(logger(), "[{}][{}] Reached iteration limit (limit={})", "Newton", m_line_search->name(), m_stop.iterations);

        double tot_time = stop_watch.getElapsedTimeInSec();
        const bool succeeded = m_status == polysolve::nonlinear::Status::GradNormTolerance;
        logger().log(
            succeeded ? spdlog::level::info : spdlog::level::err,
            "[{}][{}] Finished: {} took {:g}s. Stopped criteria: iters={:d} Delta f={:g} Grad Norm={:g} Delta x={:g} Delta x Dot Grad={:g}",
            "Newton", m_line_search->name(), status_message(m_status), tot_time, m_current.iterations, m_current.fDelta, m_current.gradNorm, m_current.xDelta, m_current.xDeltaDotGrad);

        return EXIT_SUCCESS;
    }
};

Eigen::MatrixXd MexFunction::V_prev;
Eigen::VectorXd MexFunction::sol_ls_0;
Eigen::VectorXd MexFunction::sol_ls_1;
std::vector<double> MexFunction::t_list;
SimStatistics MexFunction::sim_stat;
std::unique_ptr<ipc::Candidates> MexFunction::candidates = nullptr;
