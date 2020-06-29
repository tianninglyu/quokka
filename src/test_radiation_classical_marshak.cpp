//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file test_radiation_marshak.cpp
/// \brief Defines a test problem for radiation in the diffusion regime.
///

#include "test_radiation_classical_marshak.hpp"

auto main() -> int
{
	// Initialization

	Kokkos::initialize();

	int result = 0;

	{ // objects must be destroyed before Kokkos::finalize, so enter new
	  // scope here to do that automatically

		result = testproblem_radiation_classical_marshak();

	} // destructors must be called before Kokkos::finalize()
	Kokkos::finalize();

	return result;
}

struct SuOlsonProblem {
}; // dummy type to allow compile-type polymorphism via template specialization

// Su & Olson (1997) parameters
const double eps_SuOlson = 1.0;
const double kappa = 1.0;
const double rho = 1.0;	       // g cm^-3 (matter density)
const double T_hohlraum = 1.0; // dimensionless
// const double T_hohlraum_scaled = 3.481334e6; // K [= 300 eV]
// const double kelvin_to_eV = 8.617385e-5;
const double a_rad = 1.0;
const double c = 1.0;
const double alpha_SuOlson = 4.0 * a_rad / eps_SuOlson;

template <> void RadSystem<SuOlsonProblem>::FillGhostZones(array_t &cons)
{
	// Su & Olson (1996) boundary conditions
	const double T_H = T_hohlraum;
	const double E_inc = radiation_constant_ * std::pow(T_H, 4);
	const double F_inc = c_light_ * E_inc / 4.0;

	// x1 left side boundary (Marshak)
	for (int i = 0; i < nghost_; ++i) {
		cons(radEnergy_index, i) = E_inc;
		cons(x1RadFlux_index, i) = F_inc;
	}

	// x1 right side boundary (reflecting)
	for (int i = nghost_ + nx_; i < nghost_ + nx_ + nghost_; ++i) {
		cons(radEnergy_index, i) = cons(
		    radEnergy_index, (nghost_ + nx_) - (i - nx_ - nghost_ + 1));
		cons(x1RadFlux_index, i) =
		    -1.0 * cons(x1RadFlux_index,
				(nghost_ + nx_) - (i - nx_ - nghost_ + 1));
	}
}

template <>
auto RadSystem<SuOlsonProblem>::ComputeOpacity(const double rho,
					       const double Tgas) -> double
{
	return kappa;
}

template <>
auto RadSystem<SuOlsonProblem>::ComputeTgasFromEgas(const double rho,
						    const double Egas) -> double
{
	return std::pow(4.0 * Egas / alpha_SuOlson, 1. / 4.);
}

template <>
auto RadSystem<SuOlsonProblem>::ComputeEgasFromTgas(const double rho,
						    const double Tgas) -> double
{
	return (alpha_SuOlson / 4.0) * std::pow(Tgas, 4);
}

template <>
auto RadSystem<SuOlsonProblem>::ComputeEgasTempDerivative(const double rho,
							  const double Tgas)
    -> double
{
	// This is also known as the heat capacity, i.e.
	// 		\del E_g / \del T = \rho c_v,
	// for normal materials.

	// However, for this problem, this must be of the form \alpha T^3
	// in order to obtain an exact solution to the problem.
	// The input parameters are the density and *temperature*, not Egas
	// itself.

	return alpha_SuOlson * std::pow(Tgas, 3);
}

auto testproblem_radiation_classical_marshak() -> int
{
	// For this problem, you must do reconstruction in the reduced
	// flux, *not* the flux. Otherwise, F exceeds cE at sharp temperature
	// gradients.

	// Problem parameters

	const int max_timesteps = 2e5;
	const double CFL_number = 0.4;
	const int nx = 1500;

	const double initial_dtau = 1e-9; // dimensionless time
	const double max_dtau = 1e-2;	  // dimensionless time
	const double max_tau = 10.0;	  // dimensionless time
	const double Lz = 100.0;	  // dimensionless length

	// Su & Olson (1997) parameters
	const double chi = rho * kappa; // cm^-1 (total matter opacity)
	const double Lx = Lz / chi;	// cm
	const double max_time = max_tau / (eps_SuOlson * c * chi);	  // s
	const double max_dt = max_dtau / (eps_SuOlson * c * chi);	  // s
	const double initial_dt = initial_dtau / (eps_SuOlson * c * chi); // s

	// Problem initialization

	RadSystem<SuOlsonProblem> rad_system(
	    {.nx = nx, .lx = Lz, .cflNumber = CFL_number});

	rad_system.set_radiation_constant(a_rad);
	rad_system.set_c_light(c);
	rad_system.set_lx(Lx);

	const auto initial_Egas =
	    1e-10 * rad_system.ComputeEgasFromTgas(rho, T_hohlraum);
	const auto initial_Erad = 1e-10 * (a_rad * std::pow(T_hohlraum, 4));
	rad_system.Erad_floor_ = initial_Erad;

	auto nghost = rad_system.nghost();
	for (int i = nghost; i < nx + nghost; ++i) {
		rad_system.set_radEnergy(i) = initial_Erad;
		rad_system.set_x1RadFlux(i) = 0.0;
		rad_system.set_gasEnergy(i) = initial_Egas;
		rad_system.set_staticGasDensity(i) = rho;
		rad_system.set_x1GasMomentum(i) = 0.0;
		rad_system.set_radEnergySource(i) = 0.0;
	}

	const auto Erad0 = rad_system.ComputeRadEnergy();
	const auto Egas0 = rad_system.ComputeGasEnergy();
	const auto Etot0 = Erad0 + Egas0;

	std::cout << "radiation constant (code units) = " << a_rad << "\n";
	std::cout << "c_light (code units) = " << c << "\n";
	std::cout << "Lx = " << Lx << "\n";
	std::cout << "initial_dt = " << initial_dt << "\n";
	std::cout << "max_dt = " << max_dt << "\n";

	// Main time loop

	for (int j = 0; j < max_timesteps; ++j) {

		if (rad_system.time() >= max_time) {
			std::cout << "Timestep " << j
				  << "; t = " << rad_system.time()
				  << "; dt = " << rad_system.dt() << "\n";

			const auto Erad = rad_system.ComputeRadEnergy();
			const auto Egas = rad_system.ComputeGasEnergy();
			const auto Etot = Erad + Egas;
			const auto Ediff = std::fabs(Etot - Etot0);

			std::cout << "radiation energy = " << Erad << "\n";
			std::cout << "gas energy = " << Egas << "\n";
			std::cout << "Total energy = " << Etot << "\n";
			std::cout << "(Energy nonconservation = " << Ediff
				  << ")\n";
			std::cout << "\n";

			break;
		}

		const double this_dtMax = ((j == 0) ? initial_dt : max_dt);
		rad_system.AdvanceTimestepRK2(this_dtMax);
	}

	std::vector<double> xs(nx);
	std::vector<double> Trad(nx);
	std::vector<double> Tgas(nx);
	std::vector<double> Erad(nx);
	std::vector<double> Egas(nx);
	std::vector<double> x1GasMomentum(nx);
	std::vector<double> x1RadFlux(nx);

	for (int i = 0; i < nx; ++i) {
		const double x = Lx * ((i + 0.5) / static_cast<double>(nx));
		xs.at(i) = std::sqrt(3.0) * x;

		const auto Erad_t = rad_system.radEnergy(i + nghost) / std::sqrt(3.);
		Erad.at(i) = Erad_t;
		Trad.at(i) = std::pow(Erad_t / a_rad, 1. / 4.);

		const auto Etot_t = rad_system.gasEnergy(i + nghost);
		const auto rho = rad_system.staticGasDensity(i + nghost);
		const auto x1GasMom = rad_system.x1GasMomentum(i + nghost);
		const auto Ekin = (x1GasMom*x1GasMom) / (2.0*rho);

		const auto Egas_t = (Etot_t - Ekin) / std::sqrt(3.);
		Egas.at(i) = Egas_t;
		Tgas.at(i) = rad_system.ComputeTgasFromEgas(rho, Egas_t);

		x1GasMomentum.at(i) = rad_system.x1GasMomentum(i + nghost);
		x1RadFlux.at(i) = rad_system.x1RadFlux(i + nghost);
	}

	// read in exact solution

	std::vector<double> xs_exact;
	std::vector<double> Trad_exact;
	std::vector<double> Tmat_exact;

	std::string filename = "../../extern/SuOlson/100pt_tau10p0.dat";
	std::ifstream fstream(filename, std::ios::in);
	assert(fstream.is_open());

	std::string header;
	std::getline(fstream, header);

	for (std::string line; std::getline(fstream, line);) {
		std::istringstream iss(line);
		std::vector<double> values;

		for (double value; iss >> value;) {
			values.push_back(value);
		}
		auto x_val = std::sqrt(3.0) * values.at(1);
		auto Trad_val = values.at(4);
		auto Tmat_val = values.at(5);

		xs_exact.push_back(x_val);
		Trad_exact.push_back(Trad_val);
		Tmat_exact.push_back(Tmat_val);
	}

	// compute error norm

	std::vector<double> Trad_interp(xs_exact.size());
	interpolate_arrays(xs_exact.data(), Trad_interp.data(), xs_exact.size(),
			   xs.data(), Trad.data(), xs.size());

	double err_norm = 0.;
	double sol_norm = 0.;
	for (int i = 0; i < xs_exact.size(); ++i) {
		err_norm += std::pow(Trad_interp[i] - Trad_exact[i], 2);
		sol_norm += std::pow(Trad_exact[i], 2);
	}

	const double error_tol = 0.003;
	const double rel_error = err_norm / sol_norm;
	std::cout << "Relative L2 error norm = " << rel_error << std::endl;

	// plot results

	// temperature
	std::map<std::string, std::string> Trad_args;
	Trad_args["label"] = "radiation temperature";
	matplotlibcpp::plot(xs, Trad, Trad_args);

	std::map<std::string, std::string> Trad_exact_args;
	Trad_exact_args["label"] = "radiation temperature (exact)";
	matplotlibcpp::plot(xs_exact, Trad_exact, Trad_exact_args);

	std::map<std::string, std::string> Tgas_args;
	Tgas_args["label"] = "gas temperature";
	matplotlibcpp::plot(xs, Tgas, Tgas_args);

	std::map<std::string, std::string> Tgas_exact_args;
	Tgas_exact_args["label"] = "gas temperature (exact)";
	matplotlibcpp::plot(xs_exact, Tmat_exact, Tgas_exact_args);

	matplotlibcpp::xlabel("length x (dimensionless)");
	matplotlibcpp::ylabel("temperature (dimensionless)");
	matplotlibcpp::xlim(0.4, 100.); // dimensionless
	matplotlibcpp::ylim(0.0, 1.0);	// dimensionless
	matplotlibcpp::xscale("log");
	matplotlibcpp::legend();
	matplotlibcpp::title(fmt::format("time t = {:.4g}", rad_system.time()));
	matplotlibcpp::save("./classical_marshak_wave_temperature.pdf");

	// momentum
	std::map<std::string, std::string> gasmom_args, radmom_args;
	gasmom_args["label"] = "gas momentum density";
	radmom_args["label"] = "radiation momentum density";

	matplotlibcpp::clf();
	matplotlibcpp::plot(xs, x1GasMomentum, gasmom_args);
	matplotlibcpp::plot(xs, x1RadFlux, radmom_args);
	matplotlibcpp::xlabel("length x (dimensionless)");
	matplotlibcpp::ylabel("momentum density (dimensionless)");
	matplotlibcpp::xlim(0.4, 100.); // dimensionless
	matplotlibcpp::ylim(0.0, 3.0);	// dimensionless
	matplotlibcpp::xscale("log");
	matplotlibcpp::legend();
	matplotlibcpp::save("./classical_marshak_wave_momentum.pdf");

	// energy density
	matplotlibcpp::clf();

	std::map<std::string, std::string> Erad_args;
	Erad_args["label"] = "Numerical solution";
	Erad_args["color"] = "black";
	matplotlibcpp::plot(xs, Erad, Erad_args);

	std::map<std::string, std::string> Egas_args;
	Egas_args["label"] = "gas energy density";
	Egas_args["color"] = "red";
	matplotlibcpp::plot(xs, Egas, Egas_args);

	matplotlibcpp::xlabel("length x (dimensionless)");
	matplotlibcpp::ylabel("radiation energy density (dimensionless)");
	matplotlibcpp::xlim(0.4, 100.0); // cm
	matplotlibcpp::ylim(0.0, 1.0);
	matplotlibcpp::xscale("log");
	matplotlibcpp::legend();
	matplotlibcpp::title(fmt::format(
	    "time ct = {:.4g}", rad_system.time() * (eps_SuOlson * c * chi)));
	matplotlibcpp::save("./classical_marshak_wave.pdf");

	matplotlibcpp::xscale("log");
	matplotlibcpp::yscale("log");
	matplotlibcpp::xlim(0.4, 100.0); // cm
	matplotlibcpp::ylim(1e-8, 1.3);
	matplotlibcpp::save("./classical_marshak_wave_loglog.pdf");

	// Cleanup and exit
	std::cout << "Finished." << std::endl;

	int status = 0;
	if ((rel_error > error_tol) || std::isnan(rel_error)) {
		status = 1;
	}
	return status;
}