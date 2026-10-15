"""We solve both cases and keep post-search verification outside quantum search."""
from __future__ import annotations

import copy
import itertools
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import queue
from time import perf_counter

# We limit threads before child processes import NumPy or Qiskit.
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "RAYON_NUM_THREADS", "QISKIT_NUM_PROCS"):
    os.environ[_name] = "1"

import numpy as np

from model import (
    bitstring, bits_from_string, encoding, evaluate, pool_summary, qubo,
    score_batch, validate_case, verify_small,
)


def _record(solver, returned, seconds, evaluations, **extra):
    return dict(solver=solver, status="completed", returned=returned, gap=None,
                runtime_seconds=seconds,
                budgets=dict(candidate_evaluations=evaluations,
                             objective_evaluations=None, shots=None,
                             wall_seconds_limit=None), **extra)


def classical_results(case):
    """We enumerate feasible assignments and independently search 100 random ones."""
    case = validate_case(case)
    n = len(case["ids"])
    if n not in (6, 20):
        raise ValueError("We support the fixed 6-person and 20-person cases in this version.")
    started = perf_counter()
    choices = itertools.combinations(range(n), n // 2)
    best_energy, best_bits, ties, checked = math.inf, None, 0, 0
    # We retain one batch and its best value, without a million Python state records.
    while block := list(itertools.islice(choices, 4096)):
        bits = np.zeros((len(block), n), dtype=np.int8)
        bits[np.arange(len(block))[:, None], np.asarray(block)] = 1
        energies = score_batch(case, bits)
        local_min = float(energies.min())
        if local_min < best_energy - 1e-8:
            best_energy = local_min
            best_bits = bits[np.flatnonzero(np.abs(energies - local_min) < 1e-8)[0]].copy()
            ties = 0
        if abs(local_min - best_energy) < 1e-8:
            ties += int(np.count_nonzero(np.abs(energies - best_energy) < 1e-8))
        checked += len(block)
    assert checked == math.comb(n, n // 2)
    exact = _record("exact", evaluate(case, best_bits), perf_counter() - started,
                    checked, optimal_assignment_count=ties)
    exact["status"] = "optimal"

    started = perf_counter()
    rng = np.random.default_rng(case["random_seed"])
    candidates = np.zeros((100, n), dtype=np.int8)
    for bits in candidates:
        bits[rng.choice(n, n // 2, replace=False)] = 1
    energies = score_batch(case, candidates)
    random = _record("random", evaluate(case, candidates[int(np.argmin(energies))]),
                     perf_counter() - started, 100, seed=int(case["random_seed"]))
    return [exact, random]


def _quadratic_program(case):
    from qiskit_optimization import QuadraticProgram
    q = qubo(case)
    program = QuadraticProgram("team_builder")
    for i in range(len(case["ids"])):
        program.binary_var(f"x_{i}")
    program.minimize(
        constant=q["constant"], linear=q["linear"],
        quadratic={(i, j): coefficient
                   for i, row in enumerate(q["quadratic"])
                   for j, coefficient in enumerate(row) if j > i and coefficient != 0})
    return program


def _count_distribution(case, counts):
    shots = sum(counts.values())
    if shots != 2048:
        raise RuntimeError("Final counts do not match the fixed budget of 2048 shots.")
    feasible = sum(count for text, count in counts.items()
                   if text.count("1") == len(case["ids"]) // 2)
    return feasible / shots


def _quantum_worker(case, config, messages):
    """We pass only the model and seed into search; no classical optimum is available here."""
    try:
        from qiskit import transpile
        from qiskit.primitives import StatevectorSampler
        from qiskit_optimization.algorithms import MinimumEigenOptimizer
        from qiskit_optimization.minimum_eigensolvers import QAOA
        from qiskit_optimization.optimizers import COBYLA

        program = _quadratic_program(case)
        _, offset = program.to_ising()
        seed = config["seed"]
        initial = np.random.default_rng(seed).uniform(0, np.pi, 2)
        history = []

        def callback(evaluation, point, value, metadata):
            item = dict(evaluation=int(evaluation), parameters=np.asarray(point).tolist(),
                        energy=float(np.real(value)) + float(offset))
            history.append(item)
            messages.put({"event": "progress", "value": item})

        optimizer = COBYLA(maxiter=config["max_evaluations"], tol=1e-3)
        algorithm = QAOA(
            sampler=StatevectorSampler(seed=seed, default_shots=2048),
            optimizer=optimizer, reps=1, initial_point=initial, callback=callback)
        result = MinimumEigenOptimizer(algorithm).solve(program)
        eigen = result.min_eigen_solver_result
        if result.x is None or eigen is None or eigen.optimal_circuit is None:
            raise RuntimeError("QAOA did not return a decodable measurement result.")
        returned = evaluate(case, result.x)
        if abs(float(result.fval) - returned["E"]) > 1e-8:
            raise RuntimeError("The Qiskit energy disagrees with our independent evaluator.")

        # We retain the library's answer and separately sample with an independent seed.
        library_counts = {
            format(int(index), f"0{len(case['ids'])}b"): int(round(float(prob) * 2048))
            for index, prob in eigen.eigenstate.items()}
        if sum(library_counts.values()) != 2048:
            raise RuntimeError("Library probabilities cannot be reconstructed as shot counts.")
        circuit = eigen.optimal_circuit.assign_parameters(eigen.optimal_parameters)
        diagnostic_seed = int(np.random.SeedSequence([seed, 2048]).generate_state(1)[0])
        messages.put({"event": "stage", "value": "independent_sampling"})
        sampled = StatevectorSampler(seed=diagnostic_seed, default_shots=2048).run(
            [circuit], shots=2048).result()[0]
        if len(circuit.cregs) != 1 or circuit.num_clbits != len(case["ids"]):
            raise RuntimeError("The measurement register does not match the expected bit order.")
        counts = getattr(sampled.data, circuit.cregs[0].name).get_counts()
        feasible_mass = _count_distribution(case, counts)
        native = transpile(circuit.remove_final_measurements(inplace=False),
                           basis_gates=["h", "rx", "rz", "cx"], optimization_level=0,
                           num_processes=1)
        evaluations = int(eigen.cost_function_evals)
        if evaluations != len(history):
            raise RuntimeError("The evaluation count disagrees with the recorded history.")
        quantum = dict(
            p=1, seed=seed, diagnostic_seed=diagnostic_seed,
            parameters=np.asarray(eigen.optimal_point).tolist(),
            parameter_names=[str(p) for p in eigen.optimal_circuit.parameters],
            initial_parameters=initial.tolist(), counts=counts, library_counts=library_counts,
            feasible_sample_mass=feasible_mass, optimal_sample_mass=None,
            probability_source="independent finite-shot sample; empirical frequencies",
            returned_source="MinimumEigenOptimizer result.x; unchanged",
            history=history,
            stop_reason=("evaluation_limit" if evaluations >= config["max_evaluations"]
                         else "optimizer_terminated"),
            resources=dict(n_qubits=native.num_qubits, depth=native.depth(),
                           cx_count=int(native.count_ops().get("cx", 0)),
                           basis=["h", "rx", "rz", "cx"],
                           connectivity="ideal all-to-all; no routing or noise"))
        status = "completed" if returned["feasible"] else "infeasible"
        if not any(count > 0 and key.count("1") == len(case["ids"]) // 2
                   for key, count in library_counts.items()):
            status = "no_feasible_samples"
        messages.put({"event": "done", "returned": returned, "status": status,
                      "qaoa": quantum, "evaluations": evaluations})
    except Exception as error:
        messages.put({"event": "error", "error": f"{type(error).__name__}: {error}"})


def run_quantum(case, *, wall_seconds_limit=300):
    """We run actual QAOA in a process with a hard deadline and no classical fallback."""
    case = validate_case(case)
    n = len(case["ids"])
    if n not in (6, 20):
        raise ValueError("We support the fixed 6-person and 20-person cases in this version.")
    if wall_seconds_limit <= 0:
        raise ValueError("wall_seconds_limit must be positive.")
    config = dict(seed=int(case["qaoa_seed"]), max_evaluations=200 if n == 6 else 60)
    context = mp.get_context("spawn")
    messages = context.Queue()
    process = context.Process(target=_quantum_worker, args=(case, config, messages))
    history, event, stage = [], None, "starting"
    started = perf_counter()
    process.start()
    try:
        while perf_counter() - started < wall_seconds_limit:
            try:
                message = messages.get(timeout=min(0.2, max(
                    0.001, wall_seconds_limit - (perf_counter() - started))))
            except queue.Empty:
                if not process.is_alive():
                    event = dict(event="error", error=f"QAOA worker exited with code {process.exitcode}.")
                    break
                continue
            if message["event"] == "progress":
                history.append(message["value"])
                stage = "training"
            elif message["event"] == "stage":
                stage = message["value"]
            else:
                event = message
                break
        if process.is_alive():
            if event is None:
                process.terminate()
            process.join(timeout=2)
        if process.is_alive():
            process.kill()
        process.join(timeout=2)
    finally:
        if process.is_alive():
            process.kill()
            process.join(timeout=2)
        messages.close()
    elapsed = perf_counter() - started
    result = _record("qaoa", None, elapsed, None)
    result["budgets"].update(objective_evaluations=len(history), shots=2048,
                              wall_seconds_limit=wall_seconds_limit,
                              objective_evaluations_limit=config["max_evaluations"],
                              training_shots_per_evaluation=2048,
                              final_library_shots=0,
                              independent_diagnostic_shots=0)
    result["qaoa"] = dict(
        p=1, seed=config["seed"], parameters=None, counts={}, library_counts={},
        feasible_sample_mass=None, optimal_sample_mass=None, history=history,
        resources=dict(n_qubits=n, depth=None, cx_count=None), stage=stage)
    if event is None:
        result["status"] = "timeout"
        result["qaoa"]["stop_reason"] = "wall_time_limit"
    elif event["event"] == "error":
        result["status"] = "error"
        result["error"] = event["error"]
        result["qaoa"]["stop_reason"] = "solver_error"
    else:
        result.update(status=event["status"], returned=event["returned"], qaoa=event["qaoa"])
        result["budgets"].update(objective_evaluations=event["evaluations"],
                                  final_library_shots=2048,
                                  independent_diagnostic_shots=2048)
    return result


def finalize_case(case, quantum, classic):
    """We attach exact references and sample metrics only after search has finished."""
    case = validate_case(case)
    results = copy.deepcopy(classic + [quantum])
    exact_energy = next(r["returned"]["E"] for r in results if r["solver"] == "exact")
    for result in results:
        returned = result["returned"]
        if returned is not None:
            checked = evaluate(case, returned["bits"])
            if abs(checked["E"] - returned["E"]) > 1e-8:
                raise RuntimeError("The independent export-time score disagrees with the solver result.")
            if checked["feasible"] and checked["E"] < exact_energy - 1e-8:
                raise RuntimeError("A feasible candidate beats the exhaustive reference; we must investigate.")
            result["gap"] = max(0.0, checked["E"] - exact_energy) if checked["feasible"] else None
        if result["solver"] == "qaoa" and result["qaoa"]["counts"]:
            counts = result["qaoa"]["counts"]
            mass = sum(count for text, count in counts.items()
                       if (record := evaluate(case, bits_from_string(text)))["feasible"]
                       and abs(record["E"] - exact_energy) < 1e-8)
            result["qaoa"]["optimal_sample_mass"] = mass / sum(counts.values())
    return dict(id=case["id"], n=len(case["ids"]), team_size=len(case["ids"]) // 2,
                ids=list(case["ids"]), skills=list(case["skills"]),
                scores=case["scores"].tolist(), seeds=case["seeds"],
                **encoding(case),
                verification=verify_small(case) if len(case["ids"]) == 6 else None,
                results=results)


def export_results(cases, quantum_results, classic_results, data_dir, output_path):
    from importlib.metadata import version
    import platform
    if not (len(cases) == len(quantum_results) == len(classic_results) == 2):
        raise ValueError("We must export both the 6-person and 20-person cases together.")
    payload = dict(
        schema_version=1, master_seed=20261014, pool=pool_summary(data_dir),
        environment=dict(python=platform.python_version(), platform=platform.platform(),
                         machine=platform.machine(), processor=platform.processor(),
                         threads=1, versions={name: version(name) for name in
                         ("qiskit", "qiskit-optimization", "numpy", "scipy")}),
        cases=[finalize_case(c, q, r) for c, q, r in
               zip(cases, quantum_results, classic_results)])
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
                    encoding="utf-8")
    return payload
