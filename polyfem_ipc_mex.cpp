#include "mex.hpp"
#include "mexAdapter.hpp"

// ==============================================================================
// C-API DECLARATIONS (Manual)
// ==============================================================================

#include <iostream>
#include <vector>
#include <string>
#include <set>
#include <filesystem>
#include <fstream>
#include <memory>
#include <streambuf>
#include <mutex>

// ==============================================================================
// POLYFEM & SPDLOG INCLUDES
// ==============================================================================
#include <polyfem/State.hpp>
#include <polyfem/utils/JSONUtils.hpp>
#include <polyfem/utils/Logger.hpp>
#include <polyfem/io/YamlToJson.hpp>
#include <h5pp/h5pp.h>

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
// HELPER: Custom SPDLOG Sink (C++ API Version)
// Intercepts Polyfem logs and prints them to MATLAB Console
// ==============================================================================
template<typename Mutex>
class MatlabSink : public spdlog::sinks::base_sink<Mutex>
{
    std::shared_ptr<matlab::engine::MATLABEngine> matlabPtr;
    matlab::data::ArrayFactory factory;

public:
    explicit MatlabSink(std::shared_ptr<matlab::engine::MATLABEngine> ptr) : matlabPtr(ptr) {}

protected:
    void sink_it_(const spdlog::details::log_msg& msg) override
    {
        // 1. Format message
        spdlog::memory_buf_t formatted;
        spdlog::sinks::base_sink<Mutex>::formatter_->format(msg, formatted);

        // 2. Convert to string
        std::string log_str(formatted.data(), formatted.size());

        // 3. Print to MATLAB Console
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
// HELPER: Standard cout/cerr Redirector
// ==============================================================================
class MatlabStreamBuf : public std::streambuf {
    std::shared_ptr<matlab::engine::MATLABEngine> matlabPtr;
    ArrayFactory factory;
public:
    MatlabStreamBuf(std::shared_ptr<matlab::engine::MATLABEngine> ptr) : matlabPtr(ptr) {}
protected:
    virtual std::streamsize xsputn(const char* s, std::streamsize n) override {
        matlabPtr->feval(u"fprintf", 0, std::vector<Array>({ factory.createScalar(std::string(s, n)) }));
        return n;
    }
    virtual int overflow(int c) override {
        if (c != EOF) {
            char ch = static_cast<char>(c);
            matlabPtr->feval(u"fprintf", 0, std::vector<Array>({ factory.createScalar(std::string(1, ch)) }));
        }
        return c;
    }
};

// ==============================================================================
// HELPER: Argument Holder
// ==============================================================================
struct SimulationOptions {
    std::string json_file = "";
    std::string yaml_file = "";
    std::string hdf5_file = "";
    std::string output_dir = "";

    unsigned max_threads = std::numeric_limits<unsigned>::max();
    bool is_strict = true;
    bool fallback_solver = false;
	int restart_step = 0;
    int log_level = static_cast<int>(spdlog::level::debug); // Default to Info

    std::set<std::string> active_args;
    void mark_arg(const std::string& name) { active_args.insert(name); }
    bool has_arg(const std::string& name) const { return active_args.find(name) != active_args.end(); }
};

// ==============================================================================
// HELPER: File Loaders
// ==============================================================================
bool load_json(const std::string& json_file, json& out) {
    std::ifstream file(json_file);
    if (!file.is_open()) return false;
    file >> out;
    if (!out.contains("root_path")) out["root_path"] = json_file;
    return true;
}

bool load_yaml(const std::string& yaml_file, json& out) {
    try {
        out = polyfem::io::yaml_file_to_json(yaml_file);
        if (!out.contains("root_path")) out["root_path"] = yaml_file;
    }
    catch (...) {
        return false;
    }
    return true;
}

// ==============================================================================
// THE MEX FUNCTION CLASS
// ==============================================================================
class MexFunction : public matlab::mex::Function {
    ArrayFactory factory;
    std::shared_ptr<matlab::engine::MATLABEngine> matlabPtr = getEngine();

    // PERSISTENT STATE
    static std::unique_ptr<polyfem::State> state;
    static std::unique_ptr<igl::Timer> timer;

public:
    void operator()(ArgumentList outputs, ArgumentList inputs) {
        // 1. Redirect std::cout / std::cerr
        MatlabStreamBuf buffer(matlabPtr);
        std::streambuf* oldOut = std::cout.rdbuf(&buffer);
        std::streambuf* oldErr = std::cerr.rdbuf(&buffer);

        // 2. Redirect Polyfem Logger (SPDLOG) to MATLAB
        auto matlab_sink = std::make_shared<MatlabSink_mt>(matlabPtr);
        matlab_sink->set_pattern("[%^%l%$] %v");

        // Clear existing sinks and add ours
        spdlog::logger& p_logger = polyfem::logger();
        p_logger.sinks().clear();
        p_logger.sinks().push_back(matlab_sink);
        p_logger.set_level(spdlog::level::debug);
        p_logger.flush_on(spdlog::level::debug);

        try {
            checkArguments(inputs);

            CharArray cmdArr = inputs[0];
            std::string cmd = cmdArr.toAscii();

            // --------------------------------------------------------------
            // INIT
            // --------------------------------------------------------------
            if (cmd == "create") {
                if (state) {
                    state.reset();
                    timer.reset();
                    mexUnlock();
                }

                state = std::make_unique<polyfem::State>();
				timer = std::make_unique<igl::Timer>();

                try {
                    initializeState(outputs, inputs, 1);
                }
                catch (const std::exception& e) {
                    state.reset();
                    std::cout.rdbuf(oldOut); std::cerr.rdbuf(oldErr);
                    matlabPtr->feval(u"error", 0, std::vector<Array>({ factory.createScalar("Init failed: " + std::string(e.what())) }));
                    return;
                }

                mexLock();
                matlabPtr->feval(u"fprintf", 0, std::vector<Array>({ factory.createScalar("Polyfem initialized successfully.\n") }));
            }
            // --------------------------------------------------------------
            // SOLVE
            // --------------------------------------------------------------
            else if (cmd == "solve") {
                if (!state) throw std::runtime_error("State not initialized. Call 'init' first.");
                runSolve();
            }
            // --------------------------------------------------------------
            // EXPORT
            // --------------------------------------------------------------
            else if (cmd == "export") {
                if (inputs.size() == 2 && inputs[1].getType() == ArrayType::DOUBLE) {
                    timer->stop();
                    state->timings.solving_time = timer->getElapsedTime();
                    polyfem::logger().info(" took {}s", state->timings.solving_time);
                    Eigen::MatrixXd sol;
                    matlabToEigen(inputs[1], sol);
                    state->save_json(sol);
                    state->export_data(sol, Eigen::MatrixXd());

                    matlabPtr->feval(u"fprintf", 0, std::vector<Array>({ factory.createScalar("Results saved to output directory.\n") }));
                }
                else {
                    matlabPtr->feval(u"error", 0, std::vector<Array>({ factory.createScalar("Not enough input arguments. Usage: polyfem_ipc_mex(\"export\", sol)") }));
                }
            }
            // --------------------------------------------------------------
            // CLEAR
            // --------------------------------------------------------------
            else if (cmd == "delete") {
                if (state) {
                    state.reset();
					timer.reset();
                    mexUnlock();
                    matlabPtr->feval(u"fprintf", 0, std::vector<Array>({ factory.createScalar("Polyfem State cleared.\n") }));
                }
                else {
                    matlabPtr->feval(u"fprintf", 0, std::vector<Array>({ factory.createScalar("No active state to clear.\n") }));
                }
            }
            else {
                matlabPtr->feval(u"error", 0, std::vector<Array>({ factory.createScalar("Unknown command.") }));
            }

        }
        catch (const std::exception& e) {
            matlabPtr->feval(u"error", 0, std::vector<Array>({ factory.createScalar(std::string("Runtime Error: ") + e.what()) }));
        }

        // Restore streams
        std::cout.rdbuf(oldOut);
        std::cerr.rdbuf(oldErr);
    }

private:
    // --------------------------------------------------------------------------
    // INITIALIZATION LOGIC
    // --------------------------------------------------------------------------
    void initializeState(ArgumentList outputs, ArgumentList inputs, size_t start_index) {
        SimulationOptions opts;

        // 1. Parse Arguments
        for (size_t i = start_index; i < inputs.size(); i++) {
            std::string arg = getString(inputs[i]);
            if (arg.empty()) continue;

            if (arg == "-j" || arg == "--json") {
                if (i + 1 < inputs.size()) {
                    opts.json_file = getString(inputs[++i]);
                    opts.mark_arg("json");
                }
            }
            else if (arg == "-y" || arg == "--yaml") {
                if (i + 1 < inputs.size()) {
                    opts.yaml_file = getString(inputs[++i]);
                    opts.mark_arg("yaml");
                }
            }
            else if (arg == "--hdf5") {
                if (i + 1 < inputs.size()) {
                    opts.hdf5_file = getString(inputs[++i]);
                    opts.mark_arg("hdf5");
                }
            }
            else if (arg == "-o" || arg == "--output_dir") {
                if (i + 1 < inputs.size()) {
                    opts.output_dir = getString(inputs[++i]);
                    opts.mark_arg("output_dir");
                }
            }
            else if (arg == "--max_threads") {
                if (i + 1 < inputs.size()) {
                    TypedArray<double> val = inputs[++i];
                    opts.max_threads = static_cast<unsigned>(val[0]);
                    opts.mark_arg("max_threads");
                }
            }
            else if (arg == "--log_level") {
                if (i + 1 < inputs.size()) {
                    TypedArray<double> val = inputs[++i];
                    opts.log_level = static_cast<int>(val[0]);
                    opts.mark_arg("log_level");
                }
            }
            else if (arg == "--restart_step") {
                if (i + 1 < inputs.size()) {
                    TypedArray<double> val = inputs[++i];
                    opts.restart_step = static_cast<int>(val[0]);
                }
            }
            else if (arg == "-s" || arg == "--strict_validation") {
                opts.is_strict = true;
            }
            else if (arg == "--ns" || arg == "--no_strict_validation") {
                opts.is_strict = false;
            }
            else if (arg == "--enable_overwrite_solver") {
                opts.fallback_solver = true;
                opts.mark_arg("enable_overwrite_solver");
            }
        }

        json in_args = json({});
        std::vector<std::string> names;
        std::vector<Eigen::MatrixXi> cells;
        std::vector<Eigen::MatrixXd> vertices;

        // 3. Load File
        if (!opts.json_file.empty() || !opts.yaml_file.empty()) {
            bool ok = !opts.json_file.empty() ? load_json(opts.json_file, in_args) : load_yaml(opts.yaml_file, in_args);
            if (!ok) throw std::runtime_error("Unable to open configuration file.");
        }
        else if (!opts.hdf5_file.empty()) {
            using MatrixXl = Eigen::Matrix<int64_t, Eigen::Dynamic, Eigen::Dynamic>;
            h5pp::File file(opts.hdf5_file, h5pp::FileAccess::READONLY);
            std::string json_string = file.readDataset<std::string>("json");
            in_args = json::parse(json_string);
            in_args["root_path"] = opts.hdf5_file;
            names = file.findGroups("", "/meshes");
            cells.resize(names.size());
            vertices.resize(names.size());
            for (size_t k = 0; k < names.size(); ++k) {
                const std::string& name = names[k];
                cells[k] = file.readDataset<MatrixXl>("/meshes/" + name + "/c").cast<int>();
                vertices[k] = file.readDataset<Eigen::MatrixXd>("/meshes/" + name + "/v");
            }
        }
        else {
            throw std::runtime_error("No input file specified (--json, --yaml, or --hdf5).");
        }

        // 4. Merge CLI Arguments
        json tmp = json::object();
        if (opts.has_arg("log_level")) tmp["/output/log/level"_json_pointer] = opts.log_level;
        if (opts.has_arg("max_threads")) tmp["/solver/max_threads"_json_pointer] = opts.max_threads;
        if (opts.has_arg("output_dir")) tmp["/output/directory"_json_pointer] = std::filesystem::absolute(opts.output_dir);
        if (opts.has_arg("enable_overwrite_solver")) tmp["/solver/linear/enable_overwrite_solver"_json_pointer] = opts.fallback_solver;
        if (opts.restart_step > 0) {
            tmp["/input/data/state"_json_pointer] = (std::filesystem::absolute(opts.output_dir) / fmt::format("states/restart_{:d}.hdf5", opts.restart_step)).string(); 
            tmp["/time/t0"_json_pointer] = in_args["/time/dt"_json_pointer].get<double>() * opts.restart_step;
        }

        in_args.merge_patch(tmp);

        if (in_args.contains("states")) {
            throw std::runtime_error("This MEX function supports Forward Simulation (IPC) only.");
        }

        state->init(in_args, opts.is_strict);
        state->load_mesh(/*non_conforming=*/false, names, cells, vertices);

        if (state->mesh == nullptr) throw std::runtime_error("Mesh was not loaded successfully.");

        state->stats.compute_mesh_stats(*state->mesh);
        state->build_basis();
        state->assemble_rhs();
        state->assemble_mass_mat();

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
            uint64_t ptr_addr = reinterpret_cast<uint64_t>(state.get());
            outputs[0] = factory.createScalar(ptr_addr);
        }
        if (outputs.size() > 1) {
            outputs[1] = factory.createScalar<int>(time_steps);
        }
        if (outputs.size() > 2) {
            outputs[2] = eigenToMatlab(sol);
        }
    }

    // --------------------------------------------------------------------------
    // SOLVE LOGIC
    // --------------------------------------------------------------------------
    void runSolve() {
        Eigen::MatrixXd sol;
        Eigen::MatrixXd pressure;

        matlabPtr->feval(u"fprintf", 0, std::vector<Array>({ factory.createScalar("Starting Polyfem solver...\n") }));
        matlabPtr->feval(u"drawnow", 0, std::vector<Array>({}));

        state->solve_problem(sol, pressure);
        state->compute_errors(sol);

        // This log will now appear in MATLAB
        polyfem::logger().info("total time: {}s", state->timings.total_time());

        state->save_json(sol);
        state->export_data(sol, pressure);
    }

    // --------------------------------------------------------------------------
    // UTILITIES
    // --------------------------------------------------------------------------
    void checkArguments(ArgumentList inputs) {
        if (inputs.size() < 1) throw std::runtime_error("First input must be a command string.");
        if (inputs[0].getType() != ArrayType::CHAR) throw std::runtime_error("First input must be a char array.");
    }

    std::string getString(Array input) {
        if (input.getType() == ArrayType::CHAR) {
            CharArray arr = input;
            return arr.toAscii();
        }
        else if (input.getType() == ArrayType::MATLAB_STRING) {
            StringArray arr = input;
            if (!arr.isEmpty()) return std::string(arr[0]);
        }
        return "";
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

// STATIC MEMBER DEFINITION
std::unique_ptr<polyfem::State> MexFunction::state = nullptr;
std::unique_ptr<igl::Timer> MexFunction::timer = nullptr;