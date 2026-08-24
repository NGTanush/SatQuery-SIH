import importlib.util
from pathlib import Path

module_path = Path(__file__).resolve().parents[1] / 'experiments' / 'adaptation' / 'train_lora.py'
spec = importlib.util.spec_from_file_location('train_lora_module', module_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_normalize_answer() -> None:
    assert module.normalize_answer('  YES  ') == 'yes'
    assert module.normalize_answer('Rural Area') == 'rural area'
    assert module.normalize_answer('  7  ') == '7'
