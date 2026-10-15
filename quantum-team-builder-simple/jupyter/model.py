"""We define synthetic data, a two-team model, and verifiable quantum encodings.

Importing this module never creates data or runs a solver.  Skill scores are
synthetic integers, not measured abilities or evidence about real people.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

MASTER_SEED = 20261014
SKILLS = ("Programming", "Design", "Communication")
STREAM_NAMES = ("data", "selection", "random6", "random20", "qaoa6", "qaoa20")
BIT_ORDER = "Qiskit displays x[n-1]...x[0]; our bits and IDs use x[0]...x[n-1]."


def generate_data(data_dir) -> dict:
    """Explicitly write our seeded 1,000-person pool and nested case manifest."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    children = np.random.SeedSequence(MASTER_SEED).spawn(len(STREAM_NAMES))
    seeds = {name: int(child.generate_state(1, dtype=np.uint32)[0])
             for name, child in zip(STREAM_NAMES, children)}
    # Every consumer initializes from the stored integer, making replay direct.
    scores = np.random.default_rng(seeds["data"]).integers(0, 4, size=(1000, 3))
    ids = [f"P{i + 1:04d}" for i in range(len(scores))]
    csv_path = data_dir / "participants.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["id", *SKILLS])
        writer.writerows([person, *map(int, row)] for person, row in zip(ids, scores))
    rng = np.random.default_rng(seeds["selection"])
    positions20 = rng.choice(1000, size=20, replace=False)
    positions6_in20 = rng.choice(20, size=6, replace=False)
    positions6 = positions20[positions6_in20]
    manifest = {
        "schema_version": 1, "synthetic": True, "master_seed": MASTER_SEED,
        "generator": "numpy.default_rng (PCG64); independent uniform integers 0..3",
        "seed_derivation": "SeedSequence(master).spawn(6), each generate_state(1, uint32)",
        "stream_order": list(STREAM_NAMES), "seeds": seeds,
        "pool_rows": 1000, "skills": list(SKILLS),
        "participants_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
        "selection": "20 without replacement; then 6 from those 20; sampled order preserved",
        "cases": [],
    }
    for n, positions in ((6, positions6), (20, positions20)):
        record = {"id": f"n{n}", "ids": [ids[int(i)] for i in positions],
                  "pool_positions_zero_based": positions.tolist(),
                  "random_seed": seeds[f"random{n}"], "qaoa_seed": seeds[f"qaoa{n}"]}
        if n == 6:
            record["positions_within_n20_zero_based"] = positions6_in20.tolist()
        manifest["cases"].append(record)
    (data_dir / "cases.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def _read_data(data_dir):
    data_dir = Path(data_dir)
    csv_path = data_dir / "participants.csv"
    manifest = json.loads((data_dir / "cases.json").read_text(encoding="utf-8"))
    if hashlib.sha256(csv_path.read_bytes()).hexdigest() != manifest["participants_sha256"]:
        raise ValueError("participants.csv does not match the manifest SHA256.")
    with csv_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["id", *manifest["skills"]]:
            raise ValueError("Participant columns do not match the manifest.")
        rows = [{"id": row["id"], **{key: int(row[key]) for key in manifest["skills"]}}
                for row in reader]
    if len(rows) != manifest["pool_rows"] or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Participant count or unique IDs do not match the manifest.")
    if not manifest.get("synthetic"):
        raise ValueError("This demonstration requires explicitly labeled synthetic data.")
    if any(not 0 <= row[key] <= 3 for row in rows for key in manifest["skills"]):
        raise ValueError("Synthetic skill scores must be integers in 0..3.")
    return manifest, rows


def load_cases(data_dir) -> list[dict]:
    """Load the recorded cases without sampling again or changing their order."""
    manifest, rows = _read_data(data_dir)
    cases = []
    for record in manifest["cases"]:
        positions = record["pool_positions_zero_based"]
        if any(isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < len(rows)
               for i in positions):
            raise ValueError("Invalid case positions in the pool.")
        selected = [rows[i] for i in positions]
        if [row["id"] for row in selected] != record["ids"]:
            raise ValueError("Case IDs do not agree with their recorded pool positions.")
        case = {**record, "skills": manifest["skills"], "synthetic": True,
                "scores": [[row[key] for key in manifest["skills"]] for row in selected],
                "seeds": {"master": manifest["master_seed"],
                          "data": manifest["seeds"]["data"],
                          "selection": manifest["seeds"]["selection"],
                          "random": record["random_seed"], "qaoa": record["qaoa_seed"]}}
        cases.append(validate_case(case))
    return cases


def pool_summary(data_dir) -> dict:
    manifest, rows = _read_data(data_dir)
    return {"rows": len(rows), "skills": manifest["skills"], "synthetic": True,
            "sha256": manifest["participants_sha256"], "head": rows[:6],
            "master_seed": manifest["master_seed"], "seeds": manifest["seeds"],
            "score_range": [0, 3], "distribution": "independent discrete uniform"}


def validate_case(case) -> dict:
    """Return a normalized copy; reject missing scores and silently rounded bits."""
    ids, skills = tuple(case["ids"]), tuple(case["skills"])
    if len(ids) < 2 or len(ids) % 2:
        raise ValueError("Two equal teams require an even number of people, at least 2.")
    for name, values in (("IDs", ids), ("skills", skills)):
        if not values or any(not isinstance(v, str) or not v.strip() for v in values):
            raise ValueError(f"{name} must be nonempty strings.")
        if len(set(values)) != len(values):
            raise ValueError(f"Duplicate {name} are not allowed.")
    scores = np.array(case["scores"], dtype=float, copy=True)
    if scores.shape != (len(ids), len(skills)):
        raise ValueError("Score shape must agree with the IDs and skills.")
    if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 3)):
        raise ValueError("Skill scores must be finite values in 0..3.")
    scores.setflags(write=False)
    return {**case, "ids": ids, "skills": skills, "scores": scores, "n": len(ids)}


def _parts(case):
    case = validate_case(case)
    scores = case["scores"]
    c = np.abs(scores[:, None, :] - scores[None, :, :]).mean(axis=2)
    totals = scores.sum(axis=0)
    cmax = float(np.triu(c, 1).sum())
    # E0 >= -Cmax, whereas every feasible E0 <= Bmax.  P > Cmax+Bmax
    # therefore makes every infeasible state strictly worse than any feasible one.
    p = cmax + float(0.5 * np.dot(totals, totals)) + 1.0
    return case, scores, c, totals, cmax, p


def penalty(case) -> float:
    return _parts(case)[-1]


def _bits(bits, n, *, batch=False):
    x = np.asarray(bits)
    wrong_shape = (x.ndim != 2 or x.shape[1] != n) if batch else x.shape != (n,)
    if wrong_shape:
        raise ValueError("One binary entry is required per person.")
    if x.dtype.kind not in "biuf" or not np.all(np.isfinite(x)) or not np.all((x == 0) | (x == 1)):
        raise ValueError("Assignments must contain exactly binary values.")
    return x.astype(float, copy=False)


def evaluate(case, bits) -> dict:
    """Direct pair-by-pair objective, independent of the QUBO evaluator."""
    case, scores, c, totals, _, p = _parts(case)
    x = _bits(bits, case["n"]).astype(int)
    team1 = x == 1
    t1, t2 = scores[team1].sum(axis=0), scores[~team1].sum(axis=0)
    C = float(sum(c[i, j] for i in range(case["n"]) for j in range(i + 1, case["n"])
                  if x[i] == x[j]))
    B = float(np.sum((t1 - totals / 2) ** 2 + (t2 - totals / 2) ** 2))
    violation = int(x.sum()) - case["n"] // 2
    size_penalty = float(p * violation ** 2)
    return {"bits": x.tolist(),
            "team_1": [person for person, bit in zip(case["ids"], x) if bit == 1],
            "team_2": [person for person, bit in zip(case["ids"], x) if bit == 0],
            "team_sizes": {"team_1": int(x.sum()), "team_2": int(len(x) - x.sum())},
            "skill_sums": {"team_1": dict(zip(case["skills"], map(float, t1))),
                           "team_2": dict(zip(case["skills"], map(float, t2)))},
            "C": C, "B": B, "size_penalty": size_penalty, "E": -C + B + size_penalty,
            "feasible": violation == 0, "size_violation": violation}


def score_batch(case, bitsmatrix) -> np.ndarray:
    """Vectorized energies; callers can stream combinations in bounded batches."""
    case, scores, c, totals, cmax, p = _parts(case)
    x = _bits(bitsmatrix, case["n"], batch=True)
    C = cmax - x @ c.sum(axis=1) + np.einsum("bi,bi->b", x @ c, x)
    B = 2 * np.sum((x @ scores - totals / 2) ** 2, axis=1)
    return -C + B + p * (x.sum(axis=1) - case["n"] // 2) ** 2


def qubo(case) -> dict:
    """E(x)=constant+sum linear[i]x_i+sum_{i<j}quadratic[i][j]x_ix_j."""
    case, scores, c, totals, cmax, p = _parts(case)
    n, m = case["n"], case["n"] // 2
    constant = -cmax + 0.5 * np.dot(totals, totals) + p * m ** 2
    linear = c.sum(axis=1) + 2 * np.sum(scores ** 2 - scores * totals, axis=1) + p * (1 - 2 * m)
    quadratic = np.triu(-2 * c + 4 * (scores @ scores.T) + 2 * p, 1)
    return {"constant": float(constant), "linear": linear.tolist(),
            "quadratic": quadratic.tolist()}


def to_ising(case):
    """Sparse I/Z/ZZ encoding only; never allocate a 2**n by 2**n matrix."""
    from qiskit.quantum_info import SparsePauliOp

    case = validate_case(case)
    q = qubo(case)
    h, J = np.asarray(q["linear"]), np.asarray(q["quadratic"])
    offset = float(q["constant"] + h.sum() / 2 + J.sum() / 4)
    z = -h / 2 - (J.sum(axis=0) + J.sum(axis=1)) / 4
    # Exchanging team labels leaves this objective unchanged, so its Z fields
    # vanish analytically. Remove cancellation noise, not physical interactions.
    tolerance = 8 * np.finfo(float).eps * max(1.0, float(np.max(np.abs(h))))
    if np.max(np.abs(z)) > tolerance:
        raise AssertionError("Complement-invariant team objective unexpectedly has nonzero Z fields.")
    z[:] = 0
    terms = []
    for i, value in enumerate(z):
        if value != 0:
            label = ["I"] * case["n"]
            label[case["n"] - 1 - i] = "Z"
            terms.append(("".join(label), float(value)))
    for i in range(case["n"]):
        for j in range(i + 1, case["n"]):
            if J[i, j] != 0:
                label = ["I"] * case["n"]
                label[case["n"] - 1 - i] = label[case["n"] - 1 - j] = "Z"
                terms.append(("".join(label), float(J[i, j] / 4)))
    return SparsePauliOp.from_list(terms or [("I" * case["n"], 0.0)]), offset


def encoding(case) -> dict:
    operator, offset = to_ising(case)
    return {"penalty": penalty(case), "qubo": qubo(case),
            "ising": {"offset": offset,
                      "terms": [{"label": label, "coefficient": float(np.real(coef))}
                                for label, coef in operator.to_list()]},
            "bit_order": BIT_ORDER}


def bits_from_string(value: str) -> np.ndarray:
    if not isinstance(value, str) or not value or any(char not in "01" for char in value):
        raise ValueError("A nonempty Qiskit bitstring containing only 0 and 1 is required.")
    return np.array([int(char) for char in reversed(value)], dtype=int)


def bitstring(bits) -> str:
    x = np.asarray(bits)
    if x.ndim != 1 or x.size == 0:
        raise ValueError("A nonempty binary vector is required.")
    return "".join(str(int(bit)) for bit in reversed(_bits(x, len(x))))


def verify_small(case) -> dict:
    """Actually verify every state through four independent evaluation paths."""
    from qiskit import QuantumCircuit
    from qiskit.primitives import StatevectorSampler
    from qiskit.quantum_info import Statevector
    from qiskit_optimization import QuadraticProgram

    case = validate_case(case)
    n = case["n"]
    if n > 10:
        raise ValueError("Full-state encoding verification is restricted to n <= 10.")
    states = ((np.arange(1 << n, dtype=np.uint64)[:, None]
               >> np.arange(n, dtype=np.uint64)) & 1).astype(int)
    q = qubo(case)
    qp = QuadraticProgram("encoding_verification")
    for i in range(n):
        qp.binary_var(f"x_{i}")
    qp.minimize(constant=q["constant"], linear=q["linear"],
                quadratic={(i, j): q["quadratic"][i][j]
                           for i in range(n) for j in range(i + 1, n)})
    operator, offset = to_ising(case)
    direct_records = [evaluate(case, x) for x in states]
    direct = np.array([r["E"] for r in direct_records])
    qvalues = np.array([q["constant"] + np.dot(q["linear"], x)
                        + sum(q["quadratic"][i][j] * x[i] * x[j]
                              for i in range(n) for j in range(i + 1, n)) for x in states])
    qpvalues = np.array([qp.objective.evaluate(x) for x in states])
    isingvalues = np.full(len(states), offset)
    for label, coefficient in operator.to_list():
        indices = [i for i, letter in enumerate(reversed(label)) if letter == "Z"]
        if any(letter not in "IZ" for letter in label):
            raise AssertionError("A classical binary energy must have only diagonal I/Z terms.")
        isingvalues += float(np.real(coefficient)) * np.prod(1 - 2 * states[:, indices], axis=1)
    sdkvalues = np.array([float(np.real(
        Statevector.from_int(index, 1 << n).expectation_value(operator))) + offset
        for index in range(1 << n)])
    errors = {"direct_vs_qubo": float(np.max(np.abs(direct - qvalues))),
              "direct_vs_qiskit": float(np.max(np.abs(direct - qpvalues))),
              "direct_vs_ising": float(np.max(np.abs(direct - isingvalues))),
              "direct_vs_ising_sdk": float(np.max(np.abs(direct - sdkvalues))),
              "direct_vs_batch": float(np.max(np.abs(direct - score_batch(case, states))))}
    max_error = max(errors.values())
    if max_error >= 1e-8:
        raise AssertionError(f"Energy representations disagree: {errors}")
    for index, x in enumerate(states):
        if not np.array_equal(bits_from_string(format(index, f"0{n}b")), x):
            raise AssertionError("Qiskit basis order does not decode into participant order.")
        if int(bitstring(x), 2) != index:
            raise AssertionError("Bitstring encoding does not round-trip.")
    basis = QuantumCircuit(n)
    basis.x(0)
    basis.measure_all()
    counts = StatevectorSampler(seed=0, default_shots=8).run([basis]).result()[0].data.meas.get_counts()
    expected = np.zeros(n, dtype=int)
    expected[0] = 1
    if counts != {bitstring(expected): 8} or not np.array_equal(
            bits_from_string(next(iter(counts))), expected):
        raise AssertionError("Measured basis state did not decode into participant order.")
    feasible = states.sum(axis=1) == n // 2
    ground = np.isclose(direct, direct.min(), rtol=0, atol=1e-8)
    ground_feasible = bool(np.all(feasible[ground]))
    feasible_min = float(direct[feasible].min())
    if not ground_feasible or not np.isclose(direct.min(), feasible_min, rtol=0, atol=1e-8):
        raise AssertionError("The supposedly safe penalty did not enforce equal team sizes.")
    return {"passed": True, "n_states": len(states), "states_checked": len(states),
            "tolerance": 1e-8, "n_feasible": int(feasible.sum()),
            "max_error": max_error, "errors": errors, "bit_order_roundtrip": True,
            "measurement_bit_order_passed": True,
            "ground_states_feasible": ground_feasible, "ground_energy": float(direct.min()),
            "feasible_min_energy": feasible_min, "ground_state_count": int(ground.sum()),
            "ground_states": states[ground].tolist(), "penalty": penalty(case),
            "states": [{"bits": r["bits"], "bitstring": bitstring(r["bits"]),
                        "E": r["E"], "feasible": r["feasible"],
                        "qubo_energy": float(qvalues[i]), "qiskit_energy": float(qpvalues[i]),
                        "ising_energy": float(isingvalues[i])}
                       for i, r in enumerate(direct_records)]}
