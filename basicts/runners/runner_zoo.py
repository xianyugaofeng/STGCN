from .base_runner import BaseRunner

RUNNER_ZOO = {
    'BaseRunner': BaseRunner,
    'STGCNRunner': BaseRunner,
}

def get_runner(runner_name):
    if runner_name not in RUNNER_ZOO:
        raise ValueError(f"Runner {runner_name} not found in RUNNER_ZOO")
    return RUNNER_ZOO[runner_name]