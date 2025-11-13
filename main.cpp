#include <filesystem>

#include <CLI/CLI.hpp>

#include <h5pp/h5pp.h>

#include "minimizer.h"

#include <polyfem/State.hpp>
#include <polyfem/OptState.hpp>

#include <polyfem/utils/JSONUtils.hpp>
#include <polyfem/utils/Logger.hpp>
#include <polyfem/io/YamlToJson.hpp>

#include <polyfem/solver/forms/FrictionForm.hpp>
#include <polyfem/solver/ALSolver.hpp>
#include <polyfem/solver/NLProblem.hpp>
#include <polyfem/utils/Timer.hpp>

using namespace polyfem;
using namespace solver;

bool has_arg(const CLI::App& command_line, const std::string& value)
{
	const auto* opt = command_line.get_option_no_throw(value.size() == 1 ? ("-" + value) : ("--" + value));
	if (!opt)
		return false;

	return opt->count() > 0;
}

bool load_json(const std::string& json_file, json& out)
{
	std::ifstream file(json_file);

	if (!file.is_open())
		return false;

	file >> out;

	if (!out.contains("root_path"))
		out["root_path"] = json_file;

	return true;
}

bool load_yaml(const std::string& yaml_file, json& out)
{
	try
	{
		out = io::yaml_file_to_json(yaml_file);
		if (!out.contains("root_path"))
			out["root_path"] = yaml_file;
	}
	catch (...)
	{
		return false;
	}
	return true;
}

int forward_simulation(const CLI::App& command_line,
	const std::string& hdf5_file,
	const std::string output_dir,
	const unsigned max_threads,
	const bool is_strict,
	const bool fallback_solver,
	const spdlog::level::level_enum& log_level,
	json& in_args);

int optimization_simulation(const CLI::App& command_line,
	const unsigned max_threads,
	const bool is_strict,
	const spdlog::level::level_enum& log_level,
	json& opt_args);

void solve_ipc(State& state, Eigen::MatrixXd& sol, Eigen::MatrixXd& pressure);

int minimize(NLProblem& objFunc, Eigen::VectorXd& x);

int main(int argc, char** argv)
{
	using namespace polyfem;

	CLI::App command_line{ "polyfem" };

	command_line.ignore_case();
	command_line.ignore_underscore();

	// Eigen::setNbThreads(1);
	unsigned max_threads = std::numeric_limits<unsigned>::max();
	command_line.add_option("--max_threads", max_threads, "Maximum number of threads");

	auto input = command_line.add_option_group("input");

	std::string json_file = "";
	input->add_option("-j,--json", json_file, "Simulation JSON file")->check(CLI::ExistingFile);

	std::string yaml_file = "";
	input->add_option("-y,--yaml", yaml_file, "Simulation YAML file")->check(CLI::ExistingFile);

	std::string hdf5_file = "";
	input->add_option("--hdf5", hdf5_file, "Simulation HDF5 file")->check(CLI::ExistingFile);

	input->require_option(1);

	std::string output_dir = "";
	command_line.add_option("-o,--output_dir", output_dir, "Directory for output files")->check(CLI::ExistingDirectory | CLI::NonexistentPath);

	bool is_strict = true;
	command_line.add_flag("-s,--strict_validation,!--ns,!--no_strict_validation", is_strict, "Disables strict validation of input JSON");

	bool fallback_solver = false;
	command_line.add_flag("--enable_overwrite_solver", fallback_solver, "If solver in input is not present, falls back to default.");

	const std::vector<std::pair<std::string, spdlog::level::level_enum>>
		SPDLOG_LEVEL_NAMES_TO_LEVELS = {
			{"trace", spdlog::level::trace},
			{"debug", spdlog::level::debug},
			{"info", spdlog::level::info},
			{"warning", spdlog::level::warn},
			{"error", spdlog::level::err},
			{"critical", spdlog::level::critical},
			{"off", spdlog::level::off} };
	spdlog::level::level_enum log_level = spdlog::level::debug;
	command_line.add_option("--log_level", log_level, "Log level")
		->transform(CLI::CheckedTransformer(SPDLOG_LEVEL_NAMES_TO_LEVELS, CLI::ignore_case));

	CLI11_PARSE(command_line, argc, argv);

	json in_args = json({});

	if (!json_file.empty() || !yaml_file.empty())
	{
		const bool ok = !json_file.empty() ? load_json(json_file, in_args) : load_yaml(yaml_file, in_args);

		if (!ok)
			log_and_throw_error(fmt::format("unable to open {} file", json_file));

		if (in_args.contains("states"))
			return optimization_simulation(command_line, max_threads, is_strict, log_level, in_args);
		else
			return forward_simulation(command_line, "", output_dir, max_threads,
				is_strict, fallback_solver, log_level, in_args);
	}
	else
		return forward_simulation(command_line, hdf5_file, output_dir, max_threads,
			is_strict, fallback_solver, log_level, in_args);
}

int forward_simulation(const CLI::App& command_line,
	const std::string& hdf5_file,
	const std::string output_dir,
	const unsigned max_threads,
	const bool is_strict,
	const bool fallback_solver,
	const spdlog::level::level_enum& log_level,
	json& in_args)
{
	std::vector<std::string> names;
	std::vector<Eigen::MatrixXi> cells;
	std::vector<Eigen::MatrixXd> vertices;

	if (in_args.empty() && hdf5_file.empty())
	{
		logger().error("No input file specified!");
		return command_line.exit(CLI::RequiredError("--json or --hdf5"));
	}

	if (in_args.empty() && !hdf5_file.empty())
	{
		using MatrixXl = Eigen::Matrix<int64_t, Eigen::Dynamic, Eigen::Dynamic>;

		h5pp::File file(hdf5_file, h5pp::FileAccess::READONLY);
		std::string json_string = file.readDataset<std::string>("json");

		in_args = json::parse(json_string);
		in_args["root_path"] = hdf5_file;

		names = file.findGroups("", "/meshes");
		cells.resize(names.size());
		vertices.resize(names.size());

		for (int i = 0; i < names.size(); ++i)
		{
			const std::string& name = names[i];
			cells[i] = file.readDataset<MatrixXl>("/meshes/" + name + "/c").cast<int>();
			vertices[i] = file.readDataset<Eigen::MatrixXd>("/meshes/" + name + "/v");
		}
	}

	json tmp = json::object();
	if (has_arg(command_line, "log_level"))
		tmp["/output/log/level"_json_pointer] = int(log_level);
	if (has_arg(command_line, "max_threads"))
		tmp["/solver/max_threads"_json_pointer] = max_threads;
	if (has_arg(command_line, "output_dir"))
		tmp["/output/directory"_json_pointer] = std::filesystem::absolute(output_dir);
	if (has_arg(command_line, "enable_overwrite_solver"))
		tmp["/solver/linear/enable_overwrite_solver"_json_pointer] = fallback_solver;
	assert(tmp.is_object());
	in_args.merge_patch(tmp);

	State state;
	state.init(in_args, is_strict);
	state.load_mesh(/*non_conforming=*/false, names, cells, vertices);

	// Mesh was not loaded successfully; load_mesh() logged the error.
	if (state.mesh == nullptr)
	{
		// Cannot proceed without a mesh.
		return EXIT_FAILURE;
	}

	state.stats.compute_mesh_stats(*state.mesh);

	state.build_basis();

	state.assemble_rhs();
	state.assemble_mass_mat();

	Eigen::MatrixXd sol;
	Eigen::MatrixXd pressure;

	// state.solve_problem(sol, pressure);
	solve_ipc(state, sol, pressure);

	state.compute_errors(sol);

	logger().info("total time: {}s", state.timings.total_time());

	state.save_json(sol);
	state.export_data(sol, pressure);

	return EXIT_SUCCESS;
}

int optimization_simulation(const CLI::App& command_line,
	const unsigned max_threads,
	const bool is_strict,
	const spdlog::level::level_enum& log_level,
	json& opt_args)
{
	json tmp = json::object();
	if (has_arg(command_line, "log_level"))
		tmp["/output/log/level"_json_pointer] = int(log_level);
	if (has_arg(command_line, "max_threads"))
		tmp["/solver/max_threads"_json_pointer] = max_threads;
	opt_args.merge_patch(tmp);

	OptState opt_state;
	opt_state.init(opt_args, is_strict);

	opt_state.create_states(opt_state.args["compute_objective"].get<bool>() ? polyfem::solver::CacheLevel::Solution : polyfem::solver::CacheLevel::Derivatives, opt_state.args["solver"]["max_threads"].get<int>());
	opt_state.init_variables();
	opt_state.create_problem();

	Eigen::VectorXd x;
	opt_state.initial_guess(x);

	if (opt_state.args["compute_objective"].get<bool>())
	{
		logger().info("Objective is {}", opt_state.eval(x));
		return EXIT_SUCCESS;
	}

	opt_state.solve(x);
	return EXIT_SUCCESS;
}

void solve_ipc(State& state, Eigen::MatrixXd& sol, Eigen::MatrixXd& pressure) {
	if (!state.mesh)
	{
		logger().error("Load the mesh first!");
		return;
	}
	if (state.n_bases <= 0)
	{
		logger().error("Build the bases first!");
		return;
	}

	state.stats.spectrum.setZero();

	igl::Timer timer;
	timer.start();
	logger().info("Solving {}", state.assembler->name());
	state.init_solve(sol, pressure);

	const double t0 = state.args["time"]["t0"];
	const int time_steps = state.args["time"]["time_steps"];
	const double dt = state.args["time"]["dt"];

	// Pre log the output path for easier watching
	if (state.args["output"]["advanced"]["save_time_sequence"])
	{
		logger().info("Time sequence of simulation will be written to: \"{}\"",
			state.resolve_output_path(state.args["output"]["paraview"]["file_name"]));
	}

	// state.solve_transient_tensor_nonlinear(time_steps, t0, dt, sol);
	{
		state.init_nonlinear_tensor_solve(sol, t0 + dt);

		int save_i = 0;

		state.save_timestep(t0, save_i, t0, dt, sol, Eigen::MatrixXd()); // no pressure
		save_i++;

		for (int t = 1; t <= time_steps; ++t)
		{
			double forward_solve_time = 0, remeshing_time = 0, global_relaxation_time = 0;

			POLYFEM_SCOPED_TIMER(forward_solve_time);
			// state.solve_tensor_nonlinear(sol, t);
			{
				assert(state.solve_data.nl_problem != nullptr);
				NLProblem& nl_problem = *(state.solve_data.nl_problem);

				assert(sol.size() == state.rhs.size());

				if (nl_problem.uses_lagging())
				{
					POLYFEM_SCOPED_TIMER("Initializing lagging");
					nl_problem.init_lagging(sol); // TODO: this should be u_prev projected
					logger().info("Lagging iteration 1:");
				}

				// ---------------------------------------------------------------------

				// Save the subsolve sequence for debugging
				int subsolve_count = 0;
				state.save_subsolve(subsolve_count, t, sol, Eigen::MatrixXd()); // no pressure

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
						 {"t", t}, // TODO: null if static?
						 {"info", nl_solver->info()} });
					if (al_weight > 0)
						state.stats.solver_info.back()["weight"] = al_weight;
					state.save_subsolve(++subsolve_count, t, sol, Eigen::MatrixXd()); // no pressure
					};

				Eigen::MatrixXd prev_sol = sol;
				al_solver.solve_al(nl_problem, sol,
					state.args["solver"]["augmented_lagrangian"]["nonlinear"], state.args["solver"]["linear"], state.units.characteristic_length());

				// --------------------------------------------------------------------
				//al_solver.solve_reduced(nl_problem, sol,
				//	state.args["solver"]["nonlinear"], state.args["solver"]["linear"], state.units.characteristic_length());
				assert(sol.size() == nl_problem.full_size());

				Eigen::VectorXd tmp_sol = nl_problem.full_to_reduced(sol);
				nl_problem.use_reduced_size();
				nl_problem.line_search_begin(sol, tmp_sol);

				if (!std::isfinite(nl_problem.value(tmp_sol))
					|| !nl_problem.is_step_valid(sol, tmp_sol)
					|| !nl_problem.is_step_collision_free(sol, tmp_sol))
					log_and_throw_error("Failed to apply constraints conditions; solve with augmented lagrangian first!");
				nl_problem.line_search_end();

				// --------------------------------------------------------------------
				// Perform one final solve with the DBC projected out
				logger().debug("Successfully applied constraints conditions; solving in reduced space");
				nl_problem.init(sol);
				state.solve_data.update_barrier_stiffness(sol);

				try
				{
					const auto scale = nl_problem.normalize_forms();
					auto nl_solver = polysolve::nonlinear::Solver::create(
						state.args["solver"]["nonlinear"],
						state.args["solver"]["linear"],
						state.units.characteristic_length() * scale, logger());

					//nl_solver->minimize(nl_problem, tmp_sol);
					//minimize(nl_problem, tmp_sol);
					//minimizer::finite_difference_constraint(nl_problem, tmp_sol, state.collision_mesh, Units::convert(state.args["contact"]["dhat"], state.units.length()));
					minimizer::projected_newton(nl_problem, tmp_sol, state.collision_mesh, Units::convert(state.args["contact"]["dhat"], state.units.length()));
					//minimizer::sqp(nl_problem, tmp_sol, state.collision_mesh, Units::convert(state.args["contact"]["dhat"], state.units.length()));
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
					logger().debug("Flipped elements (cnt {}) : {}", invalidList.size(), invalidList);
				}
			}

			state.save_timestep(t0 + dt * t, t, t0, dt, sol, Eigen::MatrixXd()); // no pressure
			save_i++;

			{
				POLYFEM_SCOPED_TIMER("Update quantities");
				state.solve_data.time_integrator->update_quantities(sol);
				state.solve_data.nl_problem->update_quantities(t0 + (t + 1) * dt, sol);
				state.solve_data.update_dt();
				state.solve_data.update_barrier_stiffness(sol);
			}

			logger().info("{}/{}  t={}", t, time_steps, t0 + dt * t);
			if (state.time_callback)
				state.time_callback(t, time_steps, t0 + dt * t, t0 + dt * time_steps);

			const std::string& state_path = state.resolve_output_path(fmt::format(state.args["output"]["data"]["state"], t));
			if (!state_path.empty())
				state.solve_data.time_integrator->save_state(state_path);

			// save restart file
			state.save_restart_json(t0, dt, t);
		}
	}

	timer.stop();
	state.timings.solving_time = timer.getElapsedTime();
	logger().info(" took {}s", state.timings.solving_time);

}

int minimize(NLProblem& objFunc, Eigen::VectorXd& x) {
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

		linear_solver->solve(-grad,delta_x); // H Δx = -g
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

			if(ls_iter == 25)
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