# Quantum Team Builder · Simple Edition

We use synthetic skill data to split groups of 6 and 20 participants into two equally sized teams. Our notebook follows six steps: background, objective and constraints, QUBO, Ising, QAOA, and classical verification. The static website displays the results exported by that same run.

For a hackathon, we want teams with programming, design, and communication skills. Manual allocation depends on organizer judgment; random allocation can produce uneven skill coverage; optimizing one score can concentrate a particular skill. We make the tradeoffs explicit through a reproducible objective, using differences between skill profiles as a simplified complementarity measure.

```text
quantum-team-builder-simple/
├── frontend/
│   ├── index.html
│   ├── styles.css
│   ├── app.js
│   └── data/results.json     # Actual results exported by the notebook
├── jupyter/
│   ├── team_builder.ipynb
│   ├── model.py
│   ├── solvers.py
│   ├── requirements.txt
│   └── data/
│       ├── participants.csv  # Our 1,000-person synthetic pool
│       └── cases.json        # Fixed subsets, seeds, and source hash
└── README.md
```

## Run the notebook

We verified the computational environment with Python 3.12.14. We pin the core dependencies in `jupyter/requirements.txt`. From the project root, we run:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r jupyter/requirements.txt
cd jupyter
```

For a graphical interface, we can install JupyterLab separately and open the notebook. JupyterLab is optional and is not part of the pinned computational environment:

```bash
python -m pip install "jupyterlab>=4,<5"
jupyter lab team_builder.ipynb
```

We can also execute the notebook from the beginning and save its outputs with the pinned NBClient package, without JupyterLab or nbconvert. From `jupyter/`, we run:

```bash
python - <<'PYEXEC'
from pathlib import Path
import nbformat
from nbclient import NotebookClient
path = Path("team_builder.ipynb")
notebook = nbformat.read(path, as_version=4)
NotebookClient(notebook, timeout=900, kernel_name="python3",
               resources={"metadata": {"path": str(path.parent.resolve())}}).execute()
nbformat.write(notebook, path)
PYEXEC
```

We run all cells in order. Each execution regenerates the same data and reruns both quantum cases without loading previous solver results. We run the two QAOA cases sequentially, with a 300-second limit for each case. A complete run updates `frontend/data/results.json`.

## Model and computation

We generate 1,000 synthetic participants with master seed `20261014` and three integer skill scores from 0 to 3. The 6-person case is a fixed subset of the 20-person case. We minimize `E = -C + B + P(sum(x) - n/2)^2`: C rewards complementary profiles within a team, B penalizes unequal skill totals, and P is automatically set using a safe bound. We show the formulas and coefficients in the notebook.

We actually run Qiskit QAOA with MinimumEigenOptimizer, p=1, COBYLA, and StatevectorSampler with 2,048 shots per objective evaluation. We allow at most 200 objective evaluations for 6 participants and 60 for 20 participants. We preserve the solver's returned assignment and use a separate seed for 2,048 independent final diagnostic shots. We retain timeouts, errors, and infeasible returned assignments without substituting classical answers.

We verify the encoding for all 64 states of the 6-person case. We obtain exact references by enumerating all feasible labeled assignments: 20 for 6 participants and 184,756 for 20 participants. We also compare a search over 100 uniformly random feasible candidates. We use the exact reference only after quantum search to calculate the gap and optimal-assignment sample frequency. Our counts include both assignments obtained by swapping team labels.

We use synthetic cases and ideal simulation. Finite-shot frequencies are empirical estimates, not exact ideal probabilities. The algorithms have different budgets, and we make no claim of quantum speedup or quantum advantage.

## Preview and deploy the website

From the project root, we start a local static preview:

```bash
python -m http.server 8000 --directory frontend
```

We open `http://localhost:8000` in the browser. This command is only for local development preview. The deployed site has no Python backend and reads the exported JSON.

In Vercel, we set **Root Directory** to `frontend` and **Framework Preset** to `Other`. We leave **Build Command** empty and keep the default **Output Directory** setting without an override. Before deployment, we run the notebook completely and include the generated `frontend/data/results.json`. The website does not launch QAOA online.
