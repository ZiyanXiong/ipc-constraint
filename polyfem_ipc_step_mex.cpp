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
#include <polyfem/solver/ALSolver.hpp>
#include <polyfem/solver/NLProblem.hpp>
#include <polysolve/nonlinear/Solver.hpp>

// Third-party
#include <igl/Timer.h>

// SPDLOG HEADERS
#include <spdlog/spdlog.h>
#include <spdlog/sinks/base_sink.h>
#include <spdlog/details/null_mutex.h>

using namespace polyfem;
using namespace matlab::data;
using namespace matlab::mex;

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
        matlabPtr->feval(u"drawnow", 0, std::vector<Array>({ factory.createScalar("limitrate") }));
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
// CUSTOM IPC SOLVER FUNCTION
// Copied and adapted from user snippet
// ==============================================================================
void solve(polyfem::State& state, Eigen::MatrixXd& sol, int t) {
    using namespace polyfem::solver; // For ALSolver, NLProblem

    const double t0 = state.args["time"]["t0"];
    const int time_steps = state.args["time"]["time_steps"];
    const double dt = state.args["time"]["dt"];

    {
        double forward_solve_time = 0; // variables for timers

        POLYFEM_SCOPED_TIMER(forward_solve_time);

        // Manual nonlinear solve block
        {
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

            std::shared_ptr<polysolve::nonlinear::Solver> nl_solver = state.make_nl_solver(true);

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
                        {"t", t},
                        {"info", nl_solver->info()} });
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

            try
            {
                const auto scale = nl_problem.normalize_forms();

                // NOTE: User code calls 'minimizer::sqp'. 
                // Assuming 'minimizer' is a valid namespace available in the context or included headers.
                // If 'minimizer' is not found, please ensure you have the correct include.
                //minimizer::sqp(nl_problem, tmp_sol, state.collision_mesh, Units::convert(state.args["contact"]["dhat"], state.units.length()));
                nl_solver->minimize(nl_problem, tmp_sol);
                nl_problem.finish();
            }
            catch (const std::runtime_error& e)
            {
                sol = nl_problem.reduced_to_full(tmp_sol);
                throw e;
            }
            sol = nl_problem.reduced_to_full(tmp_sol);

            al_solver.post_subsolve(0);

            if (state.args["space"]["advanced"]["count_flipped_els_continuous"])
            {
                const auto invalidList = utils::count_invalid(state.mesh->dimension(), state.bases, state.geom_bases(), sol);
                polyfem::logger().debug("Flipped elements (cnt {}) : {}", invalidList.size(), invalidList);
            }
        }

        state.save_timestep(t0 + dt * t, t, t0, dt, sol, Eigen::MatrixXd());

        {
            POLYFEM_SCOPED_TIMER("Update quantities");
            state.solve_data.time_integrator->update_quantities(sol);
            state.solve_data.nl_problem->update_quantities(t0 + (t + 1) * dt, sol);
            state.solve_data.update_dt();
            state.solve_data.update_barrier_stiffness(sol);
        }

        polyfem::logger().info("{}/{}  t={}", t, time_steps, t0 + dt * t);
        if (state.time_callback)
            state.time_callback(t, time_steps, t0 + dt * t, t0 + dt * time_steps);

        const std::string& state_path = state.resolve_output_path(fmt::format(state.args["output"]["data"]["state"], t));
        if (!state_path.empty())
            state.solve_data.time_integrator->save_state(state_path);

        state.save_restart_json(t0, dt, t);
    }

}


// ==============================================================================
// THE MEX FUNCTION
// ==============================================================================
class MexFunction : public matlab::mex::Function {
    ArrayFactory factory;
    std::shared_ptr<matlab::engine::MATLABEngine> matlabPtr = getEngine();

    struct SimulationResult {
        Eigen::MatrixXd sol;
        Eigen::MatrixXd pressure;
        bool has_run = false;
    };

    static std::map<uint64_t, SimulationResult> results_cache;
    // PERSISTENT STATE
    static std::unique_ptr<igl::Timer> timer;

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
        p_logger.flush_on(spdlog::level::info);

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
            // --------------------------------------------------------------
            if (cmd == "start") {
                if (!state->mesh)
                {
                    polyfem::logger().error("Load the mesh first!");
                    return;
                }
                if (state->n_bases <= 0)
                {
                    polyfem::logger().error("Build the bases first!");
                    return;
                }

                state->stats.spectrum.setZero();
                Eigen::MatrixXd sol;
                Eigen::MatrixXd pressure;

                timer->start();
                polyfem::logger().info("Solving {}", state->assembler->name());
                state->init_solve(sol, pressure);
                const double t0 = state->args["time"]["t0"];
                const int time_steps = state->args["time"]["time_steps"];
                const double dt = state->args["time"]["dt"];
                state->init_nonlinear_tensor_solve(sol, t0 + dt);
                state->save_timestep(t0, 0, t0, dt, sol, Eigen::MatrixXd()); // no pressure

                // Return Outputs to MATLAB
                if (outputs.size() > 0) {
                    outputs[0] = factory.createScalar<int>(time_steps);
                }
                if (outputs.size() > 1) {
                    outputs[1] = eigenToMatlab(sol);
                }
                if (outputs.size() > 2) {
                    outputs[2] = eigenToMatlab(pressure);
                }
            }
            // --------------------------------------------------------------
            // COMMAND: SOLVE
            // Usage: [sol] = polyfem_solve_mex('solve', handle, sol, t)
            // --------------------------------------------------------------
            else if (cmd == "solve") {
                printToMatlab("Starting Custom IPC Solver for state " + std::to_string(ptr_val) + "...\n");

                Eigen::MatrixXd sol;

                int t;
                // 1. Accept Inputs 
                if (inputs.size() == 4 && inputs[2].getType() == ArrayType::DOUBLE) {
                    matlabToEigen(inputs[2], sol);
                    t = getInt(inputs[3]);
                }
                else {
                    throw std::runtime_error("Invalid input arguments. Usage: [sol] = polyfem_solve_mex('solve', handle, sol, t)");
                }

                // 2. Run Custom Solver
                // Replacing state->solve_problem with custom solve_ipc
                solve(*state, sol, t);

                // 3. Return Outputs to MATLAB
                if (outputs.size() > 0) {
                    outputs[0] = eigenToMatlab(sol);
                }

                printToMatlab("Solver finished.\n");
            }
            // --------------------------------------------------------------
            // COMMAND: SAVE
            // --------------------------------------------------------------
            else if (cmd == "end") {
                if (results_cache.find(ptr_val) == results_cache.end() || !results_cache[ptr_val].has_run) {
                    throw std::runtime_error("No solution found in cache for this state. Run 'solve' first.");
                }
                printToMatlab("Saving results for state " + std::to_string(ptr_val) + "...\n");

                SimulationResult& res = results_cache[ptr_val];
                timer->stop();
                state->timings.solving_time = timer->getElapsedTime();
                polyfem::logger().info(" took {}s", state->timings.solving_time);
                state->save_json(res.sol);
                state->export_data(res.sol, res.pressure);

                printToMatlab("Results saved to output directory.\n");
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

    // Helper: Eigen::MatrixXd -> MATLAB TypedArray
    TypedArray<double> eigenToMatlab(const Eigen::MatrixXd& in) {
        TypedArray<double> out = factory.createArray<double>({ (size_t)in.rows(), (size_t)in.cols() });
        std::copy(in.data(), in.data() + in.size(), out.begin());
        return out;
    }
};

std::map<uint64_t, MexFunction::SimulationResult> MexFunction::results_cache;