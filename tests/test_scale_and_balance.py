import subprocess, sys, pathlib
import numpy as np
import pandas as pd

SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "src" / "scale_and_balance.py"
FLAGS = ["CODE_GENDER_M", "FLAG_OWN_CAR_Y"]


def _setup(root):
    proc = root / "data" / "processed"
    proc.mkdir(parents=True)  # models/ deliberately NOT created (clean checkout)
    rng = np.random.default_rng(0)

    def X(n):
        return pd.DataFrame({
            "DAYS_BIRTH": -rng.integers(7000, 23000, n),
            "AMT_CREDIT": rng.normal(5e5, 1e5, n),
            "AMT_INCOME_TOTAL": rng.normal(1.8e5, 5e4, n),
            "CNT_FAM_MEMBERS": rng.integers(1, 6, n),
            "CODE_GENDER_M": rng.integers(0, 2, n),
            "FLAG_OWN_CAR_Y": rng.integers(0, 2, n),
        })

    def y(n, pos):
        v = np.r_[np.ones(pos), np.zeros(n - pos)]
        rng.shuffle(v)
        return pd.DataFrame({"TARGET": v.astype(int)})

    X(600).to_csv(proc / "X_train.csv", index=False); y(600, 60).to_csv(proc / "y_train.csv", index=False)
    X(150).to_csv(proc / "X_test.csv", index=False); y(150, 12).to_csv(proc / "y_test.csv", index=False)


def _run(root, *args):
    return subprocess.run([sys.executable, str(SCRIPT), *args], cwd=root, capture_output=True, text=True)


def test_runs_on_clean_checkout(tmp_path):
    _setup(tmp_path)
    r = _run(tmp_path)
    assert r.returncode == 0, r.stderr[-800:]


def test_default_keeps_real_distribution_and_flags(tmp_path):
    _setup(tmp_path)
    assert _run(tmp_path).returncode == 0
    proc = tmp_path / "data" / "processed"
    Xtr = pd.read_csv(proc / "X_train_modified.csv"); ytr = pd.read_csv(proc / "y_train_modified.csv")
    Xte = pd.read_csv(proc / "X_test_modified.csv"); yte = pd.read_csv(proc / "y_test_modified.csv")
    assert len(Xtr) == 600 and ytr["TARGET"].sum() == 60          # no resampling by default
    assert "DAYS_BIRTH" not in Xtr.columns and list(Xtr.columns) == list(Xte.columns)
    assert len(Xte) == 150 and yte["TARGET"].sum() == 12
    assert set(np.unique(Xtr[FLAGS].values)) <= {0, 1}             # flags never scaled
    assert abs(Xtr["AMT_CREDIT"].mean()) < 1e-6                    # continuous cols scaled
    assert (tmp_path / "models" / "credit_scaler.pkl").exists()


def test_smote_flag_balances_and_keeps_flag_distribution(tmp_path):
    _setup(tmp_path)
    assert _run(tmp_path, "--smote").returncode == 0
    proc = tmp_path / "data" / "processed"
    Xtr = pd.read_csv(proc / "X_train_modified.csv"); ytr = pd.read_csv(proc / "y_train_modified.csv")
    orig_X = pd.read_csv(proc / "X_train.csv"); orig_y = pd.read_csv(proc / "y_train.csv")["TARGET"]
    assert ytr["TARGET"].mean() == 0.5
    real_minority = orig_X[orig_y == 1]
    synthetic = Xtr.iloc[len(orig_X):]
    for c in FLAGS:
        assert set(synthetic[c].unique()) <= {0, 1}
        # plain SMOTE silently drags flag frequencies toward 0; SMOTENC must not
        assert abs(synthetic[c].mean() - real_minority[c].mean()) < 0.12, c