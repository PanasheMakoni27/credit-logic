import importlib.util
from pathlib import Path
import pandas as pd


def load_module_from_src(name: str, filename: str):
    path = Path(__file__).resolve().parents[1] / 'src' / filename
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_add_super_clues_basic():
    mod = load_module_from_src('preprocess', 'preprocess.py')
    df = pd.DataFrame([
        {
            'Age': 40,
            'TotalDebtToIncomeRatio': 0.15,
            'Savings': 1000,
            'Checking': 500,
            'CurrentBalance': 200,
            'TotalAvailableCredit': 1000,
        }
    ])

    out = mod.add_super_clues(df)
    assert 'DTI_Risk' in out.columns
    assert out.loc[0, 'DTI_Risk'] == 'Safe' or str(out.loc[0, 'DTI_Risk']) == 'Safe'
    assert 'Total_Liquid' in out.columns
    assert out.loc[0, 'Total_Liquid'] == 1500
    assert 'CreditUtilization' in out.columns
    assert out.loc[0, 'CreditUtilization'] == 0.2


def test_add_super_clues_handles_existing_utilization():
    mod = load_module_from_src('preprocess', 'preprocess.py')
    df = pd.DataFrame([
        {
            'CreditUtilization': 0.33,
            'Savings': 0,
            'Checking': 0,
            'TotalDebtToIncomeRatio': 0.4,
        }
    ])
    out = mod.add_super_clues(df)
    assert out.loc[0, 'CreditUtilization'] == 0.33
    assert out.loc[0, 'DTI_Risk'] == 'High Risk' or str(out.loc[0, 'DTI_Risk']) == 'High Risk'
