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
//#include <doubleCCD/doubleccd.hpp>

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
    igl::Timer init_timer;
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
    //polyfem::logger().info("Avaliable nonlinear solvers: ");
    //for (std::string& solver_name : polysolve::nonlinear::Solver::available_solvers()) {
    //    std::cout << solver_name << std::endl;
    //}
    //polyfem::logger().info("Avaliable linear solvers: ");
    //for (std::string& solver_name : polysolve::linear::Solver::available_solvers()) {
    //    std::cout << solver_name << std::endl;
    //}

    //polyfem::logger().info("Augmented Lagrangian Solver:[{}], [{}]", state.args["solver"]["augmented_lagrangian"]["nonlinear"].dump(), state.args["solver"]["linear"].dump());
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
    static double bc_al_weight;
    static double bc_initial_error;
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
        p_logger.set_level(spdlog::level::debug);
        p_logger.flush_on(spdlog::level::debug);
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
            // COMMAND: INIT
            // Usage: polyfem_problem_mex('init_bc', handle, sol, t)
            // --------------------------------------------------------------
            if (cmd == "init") {
                Eigen::MatrixXd sol;
                int t;

                // Accept Inputs 
                if (inputs.size() == 4 && inputs[2].getType() == ArrayType::DOUBLE) {
                    matlabToEigen(inputs[2], sol);
                    t = getInt(inputs[3]);
                }
                else {
                    throw std::runtime_error("Invalid input arguments. Usage: polyfem_problem_mex('init', handle, sol, t)");
                }
                //sim_stat.timer.start();
                //sim_stat.init_timer.start();
                using namespace polyfem::solver; // For ALSolver, NLProblem

                const double t0 = state->args["time"]["t0"];
                const int time_steps = state->args["time"]["time_steps"];
                const double dt = state->args["time"]["dt"];
                polyfem::logger().info("{}/{}  t={}", t, time_steps, t0 + dt * t);
                assert(state.solve_data.nl_problem != nullptr);
                NLProblem& nl_problem = *(state->solve_data.nl_problem);

                assert(sol.size() == state.rhs.size());

                if (nl_problem.uses_lagging())
                {
                    POLYFEM_SCOPED_TIMER("Initializing lagging");
                    nl_problem.init_lagging(sol);
                    polyfem::logger().info("Lagging iteration 1:");
                }

                // --------------------------------------------------------------------

                const Eigen::VectorXd initial_sol = sol;
                Eigen::VectorXd tmp_sol = nl_problem.full_to_reduced(sol);
                assert(tmp_sol.size() == nl_problem.reduced_size());

                // --------------------------------------------------------------------
                const std::vector<std::shared_ptr<polyfem::solver::AugmentedLagrangianForm>>& alagr_forms = state->solve_data.al_form;
                bc_al_weight = state->args["solver"]["augmented_lagrangian"]["initial_weight"];
                int al_steps = 0;

                bc_initial_error = 0;
                for (const auto& f : alagr_forms)
                    bc_initial_error += f->compute_error(sol);

                for (auto& f : alagr_forms)
                    f->set_initial_weight(bc_al_weight);

                double current_error = 0;
                for (const auto& f : alagr_forms)
                    current_error += f->compute_error(sol);

                logger().debug("Initial error = {}", current_error);
                logger().debug("fvalue = {}, is_step_valid = {}, is_step_collision_free = {}", nl_problem.value(tmp_sol), nl_problem.is_step_valid(sol, tmp_sol), nl_problem.is_step_collision_free(sol, tmp_sol));
                V_prev = state->collision_mesh.displace_vertices(utils::unflatten(sol, state->collision_mesh.dim()));
                sol_ls_0 = sol;
                sol_ls_1 = sol;
            }
            // --------------------------------------------------------------
            // COMMAND: SOLVE_BC
            // Usage: [sol] = polyfem_problem_mex('init_bc', handle, sol)
            // --------------------------------------------------------------
            else if (cmd == "solve_bc") {
                Eigen::MatrixXd sol;
                int t;
                // Accept Inputs 
                if (inputs.size() == 4 && inputs[2].getType() == ArrayType::DOUBLE) {
                    matlabToEigen(inputs[2], sol);
                    t = getInt(inputs[3]);
                }
                else {
                    throw std::runtime_error("Invalid input arguments. Usage: [sol] = polyfem_problem_mex('solve_bc', handle, sol, t)");
                }

                const double t0 = state->args["time"]["t0"];
                const int time_steps = state->args["time"]["time_steps"];
                const double dt = state->args["time"]["dt"];
                polyfem::logger().info("{}/{}  t={}", t, time_steps, t0 + dt * t);
                assert(state.solve_data.nl_problem != nullptr);
                polyfem::solver::NLProblem& nl_problem = *(state->solve_data.nl_problem);
                double eta_tol = state->args["solver"]["augmented_lagrangian"]["eta"];
                double max_weight = state->args["solver"]["augmented_lagrangian"]["max_weight"];
                double scaling = state->args["solver"]["augmented_lagrangian"]["scaling"];

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
                state->save_subsolve(subsolve_count, t, sol, Eigen::MatrixXd());

                // ---------------------------------------------------------------------
                //{
                //    polyfem::solver::ALSolver al_solver(
                //        state->solve_data.al_form,
                //        state->args["solver"]["augmented_lagrangian"]["initial_weight"],
                //        state->args["solver"]["augmented_lagrangian"]["scaling"],
                //        state->args["solver"]["augmented_lagrangian"]["max_weight"],
                //        state->args["solver"]["augmented_lagrangian"]["eta"],
                //        [&](const Eigen::VectorXd& x) {
                //            state->solve_data.update_barrier_stiffness(sol);
                //        });

                //    al_solver.post_subsolve = [&](const double al_weight) {
                //        state->stats.solver_info.push_back(
                //            { {"type", al_weight > 0 ? "al" : "rc"},
                //                {"t", t} });
                //        if (al_weight > 0)
                //            state->stats.solver_info.back()["weight"] = al_weight;
                //        state->save_subsolve(++subsolve_count, t, sol, Eigen::MatrixXd());
                //        };

                //    al_solver.solve_al(nl_problem, sol,
                //        state->args["solver"]["augmented_lagrangian"]["nonlinear"], state->args["solver"]["linear"], state->units.characteristic_length());
                //}

                {
                    assert(sol.size() == nl_problem.full_size());

                    const Eigen::VectorXd initial_sol = sol;
                    Eigen::VectorXd tmp_sol = nl_problem.full_to_reduced(sol);
                    assert(tmp_sol.size() == nl_problem.reduced_size());

                    // --------------------------------------------------------------------
                    const std::vector<std::shared_ptr<polyfem::solver::AugmentedLagrangianForm>>& alagr_forms = state->solve_data.al_form;
                    bc_al_weight = state->args["solver"]["augmented_lagrangian"]["initial_weight"];
                    int al_steps = 0;

                    bc_initial_error = 0;
                    for (const auto& f : alagr_forms)
                        bc_initial_error += f->compute_error(sol);

                    nl_problem.use_reduced_size();
                    nl_problem.line_search_begin(sol, tmp_sol);

                    for (auto& f : alagr_forms)
                        f->set_initial_weight(bc_al_weight);

                    double current_error = 0;
                    for (const auto& f : alagr_forms)
                        current_error += f->compute_error(sol);

                    logger().debug("Initial error = {}", current_error);
                    logger().debug("fvalue = {}, is_step_valid = {}, is_step_collision_free = {}", nl_problem.value(tmp_sol), nl_problem.is_step_valid(sol, tmp_sol), nl_problem.is_step_collision_free(sol, tmp_sol));
                    while (!std::isfinite(nl_problem.value(tmp_sol))
                        || !nl_problem.is_step_valid(sol, tmp_sol)
                        || !nl_problem.is_step_collision_free(sol, tmp_sol))
                    {
                        nl_problem.line_search_end();

                        nl_problem.use_full_size();
                        logger().info("Solving AL Problem with weight {}", bc_al_weight);

                        nl_problem.init(sol);
                        //update_barrier_stiffness(sol);
                        state->solve_data.update_barrier_stiffness(sol);
                        tmp_sol = sol;

                        try
                        {
                            const auto scale = nl_problem.normalize_forms();
                            auto nl_solver = polysolve::nonlinear::Solver::create(
                                state->args["solver"]["augmented_lagrangian"]["nonlinear"], state->args["solver"]["linear"], state->units.characteristic_length() * scale, logger());

                            nl_solver->minimize(nl_problem, tmp_sol);
                            nl_problem.finish();
                        }
                        catch (const std::runtime_error& e)
                        {
                            std::string err_msg = e.what();
                            // if the nonlinear solve fails due to invalid energy at the current solution, changing the weights would not help
                            if (err_msg.find("f(x) is nan or inf; stopping") != std::string::npos)
                                log_and_throw_error("Failed to solve with AL; f(x) is nan or inf");
                            if (err_msg.find("Reached iteration limit") != std::string::npos)
                                log_and_throw_error("Reached iteration limit in AL");
                        }

                        sol = tmp_sol;

                        current_error = 0;
                        for (const auto& f : alagr_forms)
                            current_error += f->compute_error(sol);
                        logger().debug("Current error = {}", current_error);
                        const double eta = 1 - sqrt(current_error / bc_initial_error);

                        logger().debug("Current eta = {}", eta);

                        if (eta < 0)
                        {
                            logger().debug("Higher error than initial, increase weight and revert to previous solution");
                            sol = initial_sol;
                        }

                        nl_problem.use_reduced_size();
                        tmp_sol = nl_problem.full_to_reduced(sol);
                        nl_problem.line_search_begin(sol, tmp_sol);
                        logger().debug("eta tol = {}, max_al_weight = {}, scale = {}", eta_tol, max_weight, scaling);
                        if (eta < eta_tol && bc_al_weight < max_weight)
                            bc_al_weight *= scaling;

                        for (auto& f : alagr_forms)
                            f->update_lagrangian(sol, bc_al_weight);
                        logger().debug("fvalue = {}, is_step_valid = {}, is_step_collision_free = {}", nl_problem.value(tmp_sol), nl_problem.is_step_valid(sol, tmp_sol), nl_problem.is_step_collision_free(sol, tmp_sol));
                        state->solve_data.update_barrier_stiffness(sol);
                        ++al_steps;
                    }
                    nl_problem.line_search_end();
                }

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
                state->solve_data.update_barrier_stiffness(sol);
                
                outputs[0] = eigenToMatlab(tmp_sol);
            }
            // --------------------------------------------------------------
            // COMMAND: CHECK_CONVERGENCE_BC
            // Usage: [is_converged] = polyfem_problem_mex('check_convergence_bc', handle, sol)
            // --------------------------------------------------------------
            else if (cmd == "check_convergence_bc") {
                Eigen::MatrixXd sol;

                // Accept Inputs 
                if (inputs.size() == 3 && inputs[2].getType() == ArrayType::DOUBLE) {
                    matlabToEigen(inputs[2], sol);
                }
                else {
                    throw std::runtime_error("Invalid input arguments. Usage: [is_converged] = polyfem_problem_mex('check_convergence_bc', handle, sol)");
                }
                polyfem::solver::NLProblem& nl_problem = *(state->solve_data.nl_problem);
                const std::vector<std::shared_ptr<polyfem::solver::AugmentedLagrangianForm>>& alagr_forms = state->solve_data.al_form;
                double current_error = 0;

                for (const auto& f : alagr_forms)
                    current_error += f->compute_error(sol);
                logger().debug("Current error = {}", current_error);
                const double eta = 1 - sqrt(current_error / bc_initial_error);

                nl_problem.use_reduced_size();
                Eigen::VectorXd tmp_sol = nl_problem.full_to_reduced(sol);
                nl_problem.line_search_begin(sol, tmp_sol);
                bool is_bc_converged = (std::isfinite(nl_problem.value(tmp_sol))
                    && nl_problem.is_step_valid(sol, tmp_sol)
                    && nl_problem.is_step_collision_free(sol, tmp_sol) && eta > 0.999);
                nl_problem.line_search_end();

                nl_problem.use_full_size();
                logger().info("Solving AL Problem with weight {}", bc_al_weight);

                nl_problem.init(sol);
                //update_barrier_stiffness(sol);
                state->solve_data.update_barrier_stiffness(sol);
                outputs[0] = factory.createScalar<bool>(is_bc_converged);
            }
            // --------------------------------------------------------------
            // COMMAND: UPDATE_AL_WEIGHT_BC
            // Usage: polyfem_problem_mex('check_convergence_bc', handle, sol)
            // --------------------------------------------------------------
            else if (cmd == "update_al_weight_bc") {
                Eigen::MatrixXd sol;

                // Accept Inputs 
                if (inputs.size() == 3 && inputs[2].getType() == ArrayType::DOUBLE) {
                    matlabToEigen(inputs[2], sol);
                }
                else {
                    throw std::runtime_error("Invalid input arguments. Usage: polyfem_problem_mex('update_al_weight_bc', handle, sol)");
                }
                polyfem::solver::NLProblem& nl_problem = *(state->solve_data.nl_problem);
                //sol = tmp_sol;

                //current_error = 0;
                //for (const auto& f : alagr_forms)
                //    current_error += f->compute_error(sol);
                //logger().debug("Current error = {}", current_error);
                //const double eta = 1 - sqrt(current_error / bc_initial_error);

                //logger().debug("Current eta = {}", eta);

                //if (eta < 0)
                //{
                //    logger().debug("Higher error than initial, increase weight and revert to previous solution");
                //    sol = initial_sol;
                //}

                //nl_problem.use_reduced_size();
                //tmp_sol = nl_problem.full_to_reduced(sol);
                //nl_problem.line_search_begin(sol, tmp_sol);
                //logger().debug("eta tol = {}, max_al_weight = {}, scale = {}", eta_tol, max_weight, scaling);
                //if (eta < eta_tol && bc_al_weight < max_weight)
                //    bc_al_weight *= scaling;

                //for (auto& f : alagr_forms)
                //    f->update_lagrangian(sol, bc_al_weight);
                //logger().debug("fvalue = {}, is_step_valid = {}, is_step_collision_free = {}", nl_problem.value(tmp_sol), nl_problem.is_step_valid(sol, tmp_sol), nl_problem.is_step_collision_free(sol, tmp_sol));
                //state->solve_data.update_barrier_stiffness(sol);
                const std::vector<std::shared_ptr<polyfem::solver::AugmentedLagrangianForm>>& alagr_forms = state->solve_data.al_form;
                Eigen::VectorXd tmp_sol = nl_problem.full_to_reduced(sol);
                double eta_tol = state->args["solver"]["augmented_lagrangian"]["eta"];
                double max_weight = state->args["solver"]["augmented_lagrangian"]["max_weight"];
                double scaling = state->args["solver"]["augmented_lagrangian"]["scaling"];
                double current_error = 0;

                for (const auto& f : alagr_forms)
                    current_error += f->compute_error(sol);
                logger().debug("Current error = {}", current_error);
                const double eta = 1 - sqrt(current_error / bc_initial_error);

                logger().debug("Current eta = {}", eta);

                if (eta < 0)
                {
                    logger().info("Higher error than initial, increase weight and revert to previous solution");
                    sol = sol_ls_0;
                }

                if (eta < eta_tol && bc_al_weight <max_weight)
                    bc_al_weight *= scaling;

                for (auto& f : alagr_forms)
                    f->update_lagrangian(sol, bc_al_weight);
                logger().debug("fvalue = {}, is_step_valid = {}, is_step_collision_free = {}", nl_problem.value(tmp_sol), nl_problem.is_step_valid(sol, tmp_sol), nl_problem.is_step_collision_free(sol, tmp_sol));
                state->solve_data.update_barrier_stiffness(sol);
            }
            // --------------------------------------------------------------
            // COMMAND: FIX_BC
            // Usage: [sol_reduced] = polyfem_problem_mex('fix_bc', handle, sol, t)
            // --------------------------------------------------------------
            else if (cmd == "fix_bc") {
                Eigen::MatrixXd sol;

                // Accept Inputs 
                if (inputs.size() == 3 && inputs[2].getType() == ArrayType::DOUBLE) {
                    matlabToEigen(inputs[2], sol);
                }
                else {
                    throw std::runtime_error("Invalid input arguments. Usage: [sol_reduced] = polyfem_problem_mex('fix_bc', handle, sol)");
                }

                polyfem::solver::NLProblem& nl_problem = *(state->solve_data.nl_problem);
                nl_problem.use_reduced_size();
                Eigen::VectorXd tmp_sol = nl_problem.full_to_reduced(sol);
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
                state->solve_data.update_barrier_stiffness(sol);

                V_prev = state->collision_mesh.displace_vertices(utils::unflatten(sol, state->collision_mesh.dim()));
                sol_ls_0 = tmp_sol;
                sol_ls_1 = tmp_sol;
                if (candidates) {
                    candidates.reset();
                    mexUnlock();
                }
                candidates = std::make_unique<ipc::Candidates>();
                //sim_stat.init_timer.stop();
                //sim_stat.init_time += sim_stat.init_timer.getElapsedTime();
                outputs[0] = eigenToMatlab(tmp_sol);
            }

            // --------------------------------------------------------------
            // COMMAND: PREPARE
            // Usage: [sol, sol_reduced] = polyfem_problem_mex('prepare', handle, sol, t)
            // --------------------------------------------------------------
            else if (cmd == "prepare") {
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
                //Eigen::VectorXd tmp_sol;
                // Accept Inputs 
                if (inputs.size() == 3 && inputs[2].getType() == ArrayType::DOUBLE) {
                    matlabToEigen(inputs[2], sol);
                    //matlabToEigen(inputs[3], tmp_sol);
                }
                else {
                    throw std::runtime_error("Invalid input arguments. Usage: [f,g] = polyfem_problem_mex('eval_c', handle, sol)");
                }
                igl::Timer timer;
                // Start the timer
                timer.start();

                assert(state.solve_data.nl_problem != nullptr);
                polyfem::solver::NLProblem& nl_problem = *(state->solve_data.nl_problem);
                double f = 0.0;
                Eigen::VectorXd grad;
				nl_problem.solution_changed(sol);
                //bool is_step_valid = true;

				igl::Timer collision_timer;
				collision_timer.start();
                const ipc::CollisionMesh& collision_mesh = state->collision_mesh;
                
                Eigen::MatrixXd V;
                if(sol.size() == nl_problem.full_size())
                    V = collision_mesh.displace_vertices(utils::unflatten(sol, collision_mesh.dim()));
                else
                    V = collision_mesh.displace_vertices(utils::unflatten(nl_problem.reduced_to_full(sol), collision_mesh.dim()));
                bool is_step_valid = !ipc::has_intersections(collision_mesh, V, std::make_shared<ipc::SweepAndPrune>());
                //bool is_step_valid = true;

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
                    const ipc::CollisionMesh& collision_mesh = state->collision_mesh;
                    const int ndof = collision_mesh.num_vertices() * collision_mesh.dim();
                    Eigen::VectorXd dsol = sol - sol_ls_0;
                    Eigen::VectorXd ls_direction = (sol_ls_1 - sol_ls_0).normalized();
                    bool is_linesearch = (abs(ls_direction.dot(dsol) - dsol.norm()) < 1e-9) && ((sol_ls_1 - sol_ls_0).norm() > 1e-9);
                    double t = 1.0;

                    //if (is_linesearch) {
                    //    t = dsol.norm() / (sol_ls_1 - sol_ls_0).norm();
                    //}
                    //else {
                    //    collisions.build(collision_mesh, V_prev, V, dhat / 2, std::make_shared<ipc::SweepAndPrune>());
                    //    sol_ls_1 = tmp_sol;
                    //    t_list.resize(collisions.size());
                    //    sim_stat.ccd_count++;
                    //}
                    /*
                    if (!collisions.empty()) {
                        const Eigen::MatrixXi& E = collision_mesh.edges();
                        const Eigen::MatrixXi& F = collision_mesh.faces();
                        if (is_linesearch) {
                            tbb::parallel_for(
                                tbb::blocked_range<size_t>(0, collisions.size()),
                                [&](tbb::blocked_range<size_t> r) {
                                    for (size_t i = r.begin(); i < r.end(); i++) {
                                        if (t > t_list[i]) {
                                            is_step_valid = false;
                                            return;
                                        }
                                    }
                                });
                        }
                        else {
                            tbb::parallel_for(
                                tbb::blocked_range<size_t>(0, collisions.size()),
                                [&](tbb::blocked_range<size_t> r) {
                                    for (size_t i = r.begin(); i < r.end(); i++) {
                                        //double toi = 0;
                                        bool is_colliding = false;
                                        ipc::VectorMax12d dof = collisions[i].dof(V, E, F);
                                        ipc::VectorMax12d dof0 = collisions[i].dof(V_prev, E, F);
                                        is_colliding = collisions[i].ccd(dof0, dof, t_list[i]);
                                        
                                        //if(is_colliding)
                                        //{
                                        //    if (i < collisions.vv_candidates.size() + collisions.ev_candidates.size())
                                        //        is_colliding = is_colliding;
                                        //    else if (i < collisions.vv_candidates.size() + collisions.ev_candidates.size() + collisions.ee_candidates.size())
                                        //        is_colliding = doubleccd_edge_edge(dof0, dof);
                                        //    else if (i < collisions.size())
                                        //        is_colliding = doubleccd_vertex_face(dof0, dof);
                                        //    else
                                        //        polyfem::log_and_throw_error("Index out of candidates range.");
                                        //}
                                        
                                        if (is_colliding) {
                                            is_step_valid = false;
                                        }
                                    }
                                });
                        }
                    }
                    */
                }
                
				collision_timer.stop();
				sim_stat.collision_time += collision_timer.getElapsedTime();
                sim_stat.func_eval++;
                
                if (outputs.size() > 0) {
                    f = nl_problem(sol);
                    if (!is_step_valid) {
                        //f = std::numeric_limits<double>::infinity();
                        f = std::numeric_limits<double>::quiet_NaN();
                    }
                    outputs[0] = factory.createScalar<double>(f);
                }
                if (outputs.size() > 1) {
                    nl_problem.gradient(sol, grad);
                    outputs[1] = eigenToMatlab(grad);
                }
                if (outputs.size() > 2) {
                    //Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
                    Eigen::SparseMatrix<double> hessian;

                    if (is_step_valid) {
                        //logger().debug("Grad Norm: {:g}", grad.norm());
                        if (grad.lpNorm<Eigen::Infinity>() > 1e-4) 
                        {
                            try {
                                nl_problem.set_project_to_psd(true);
                                nl_problem.hessian(sol, hessian);
                            }
                            catch (const std::runtime_error& e) {
                                // Code to handle the specific exception type
                                logger().info("Invalid hessian, return 0 hessian");
                                hessian.setZero();
                            }
                        }
                        else 
                        {
                            //logger().info("Switch to unprojected hessian");
                            sim_stat.hessian_eval++;
                            nl_problem.set_project_to_psd(false);
                            nl_problem.hessian(sol, hessian);

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
                                    nl_problem.hessian(sol, hessian);
                                }
                                catch (const std::runtime_error& e) {
                                    // Code to handle the specific exception type
                                    //logger().info("Invalid hessian, return 0 hessian");
                                    hessian.setZero();
                                }
							}
                            factorization_timer.stop();
                            sim_stat.extra_factorizing_time += factorization_timer.getElapsedTime();
                        }
                        sol_ls_0 = sol;
                        V_prev = V;
                    }
                    outputs[2] = eigenSparseToMatlab(hessian);
                    json solver_info;
                    nl_problem.post_step(polysolve::nonlinear::PostStepData(1, solver_info, sol, grad));
                }

                // Stop the timer
                timer.stop();

                // Get the elapsed time in seconds (as a double)
                double elapsed_time = timer.getElapsedTime();
                sim_stat.objective_time += elapsed_time;
                //printToMatlab("Time for eval_f : " + std::to_string(elapsed_time) + " seconds\n");
            }
            // --------------------------------------------------------------
            // COMMAND: HESSIAN_PATTERN
            // Usage: h = polyfem_problem_mex('hessian_pattern', handle, x ,x_r)
            // --------------------------------------------------------------
            else if (cmd == "hessian_pattern") {
                Eigen::MatrixXd sol;
                Eigen::VectorXd tmp_sol;
                // Accept Inputs 
                if (inputs.size() == 4 && inputs[2].getType() == ArrayType::DOUBLE && inputs[3].getType() == ArrayType::DOUBLE) {
                    matlabToEigen(inputs[2], sol);
                    matlabToEigen(inputs[3], tmp_sol);
                }
                else {
                    throw std::runtime_error("Invalid input arguments. Usage: h = polyfem_problem_mex('hessian_pattern', handle, x, x_r)");
                }
                igl::Timer timer;
                // Start the timer
                timer.start();

                assert(state.solve_data.nl_problem != nullptr);
                polyfem::solver::NLProblem& nl_problem = *(state->solve_data.nl_problem);
                nl_problem.solution_changed(tmp_sol);
                //if(is_step_valid)
                //    is_step_valid = nl_problem.is_step_collision_free(nl_problem.full_to_reduced(sol), tmp_sol);

                //logger().debug("1: Is step valid? {}", is_step_valid);
                /*
                for (auto form : nl_problem.forms()) {
                    logger().info("form name: {}", form->name());
                    form->second_derivative()
                }
                */
                if (outputs.size() > 0) {
                    //Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
                    Eigen::SparseMatrix<double> hessian;
                    Eigen::SparseMatrix<double> broad_phase_hessian;
                    const ipc::CollisionMesh& collision_mesh = state->collision_mesh;
                    const int ndof = collision_mesh.num_vertices() * collision_mesh.dim();

                    nl_problem.set_project_to_psd(true);
                    nl_problem.hessian(tmp_sol, hessian);
                    for (int k = 0; k < hessian.outerSize(); ++k) {
                        for (Eigen::SparseMatrix<double>::InnerIterator it(hessian, k); it; ++it) {
                            if (it.value() != 0.0) {
                                it.valueRef() = 1.0;
                            }
                        }
                    }
                    //logger().info("Hessian size:  {}", hessian.cols());
                    std::vector<Eigen::Triplet<double>> broad_phase_hessian_triplets;
                    broad_phase_hessian_triplets.clear();

                    ipc::Candidates& collisions = *candidates;
                    double dhat = polyfem::Units::convert(state->args["contact"]["dhat"], state->units.length());

                    Eigen::MatrixXd V = collision_mesh.displace_vertices(utils::unflatten(nl_problem.reduced_to_full(tmp_sol), collision_mesh.dim()));
                    collisions.build(collision_mesh, V, dhat * 20, std::make_shared<ipc::SweepAndPrune>());
                    const Eigen::MatrixXi& E = collision_mesh.edges();
                    const Eigen::MatrixXi& F = collision_mesh.faces();
                    logger().info("Broad phase size:  {}", collisions.size());

                    for (size_t i = 0; i < collisions.size(); i++) {
                        int dof = collisions[i].num_vertices() * 3;
                        auto v_ids = collisions[i].vertex_ids(E, F);
                        
                        ipc::local_hessian_to_global_triplets(ipc::MatrixMax12d::Ones(dof,dof),
                            v_ids, collision_mesh.dim(), broad_phase_hessian_triplets);
                            
                    }

                    //logger().info("Triplets size:  {}", broad_phase_hessian_triplets.size());
                    broad_phase_hessian.resize(ndof, ndof);
                    broad_phase_hessian.setFromTriplets(broad_phase_hessian_triplets.begin(), broad_phase_hessian_triplets.end());
                    //logger().info("ndof:  {}", ndof);
                    broad_phase_hessian = collision_mesh.to_full_dof(broad_phase_hessian);
                    //logger().info("full ndof:  {}", broad_phase_hessian.cols());
                    nl_problem.full_hessian_to_reduced_hessian(broad_phase_hessian);
                    //logger().info("reduced dof:  {}", broad_phase_hessian.cols());

                    hessian += broad_phase_hessian;
                    for (int k = 0; k < hessian.outerSize(); ++k) {
                        for (Eigen::SparseMatrix<double>::InnerIterator it(hessian, k); it; ++it) {
                            if (it.value() != 0.0) {
                                it.valueRef() = 1.0;
                            }
                        }
                    }
                    
                    outputs[0] = eigenSparseToMatlab(hessian);
                }
                // Stop the timer
                timer.stop();

                // Get the elapsed time in seconds (as a double)
                double elapsed_time = timer.getElapsedTime();
                //sim_stat.objective_time += elapsed_time;
                printToMatlab("Time for computing hessian pattern : " + std::to_string(elapsed_time) + " seconds\n");
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
            // --------------------------------------------------------------
            // COMMAND: clear statistics
            // [sol] = polyfem_problem_mex('clear', state)
            // --------------------------------------------------------------
            else if (cmd == "clear") {
                //logger().info("Clear sim statistics");
                sim_stat.total_time = 0;
                sim_stat.init_time = 0;
                sim_stat.objective_time = 0;
                sim_stat.collision_time = 0;
                sim_stat.extra_factorizing_time = 0;
                sim_stat.func_eval = 0;
                sim_stat.hessian_eval = 0;
                sim_stat.ccd_count = 0;
                //logger().info("Current total time: {}s, initial time: {}s, objective funcioin time: {}s, extra factorization time: {}s, collision time: {}s.", sim_stat.total_time, sim_stat.init_time, sim_stat.objective_time, sim_stat.extra_factorizing_time, sim_stat.collision_time);
                //logger().info("Current total function evaluations: {}, total hessian evaluations: {}, ccd count: {}", sim_stat.func_eval, sim_stat.hessian_eval, sim_stat.ccd_count);
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
    /*
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
    */
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
double MexFunction::bc_al_weight = 0.0;
double MexFunction::bc_initial_error = 0.0;
