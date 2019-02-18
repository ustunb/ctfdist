from pathlib import Path

# parent directory
repo_dir = Path(__file__).absolute().parent.parent

# directory with dataset files
data_dir = repo_dir / 'data/'

# directory with results files
results_dir = repo_dir / 'results/'


data_dir.mkdir(exist_ok = True)
results_dir.mkdir(exist_ok = True)