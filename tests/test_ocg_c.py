"""Exercise the compiled kernel, including OpenMP paths and merge constraints."""
import os
from pathlib import Path
import random
import shutil
import subprocess

import pytest


@pytest.fixture(scope='module')
def ocg_binaries(tmp_path_factory):
    compiler = shutil.which(os.environ.get('CC', 'gcc'))
    if compiler is None:
        pytest.skip('C compiler unavailable (set CC to run native OCG tests)')
    work = tmp_path_factory.mktemp('ocg_build')
    source = Path(__file__).resolve().parents[1] / 'ocg' / 'OCG_fast_openai.c'
    binaries = []
    for name, flags in [('serial', []), ('parallel', ['-fopenmp'])]:
        binary = work / (name + ('.exe' if os.name == 'nt' else ''))
        subprocess.run([compiler, '-O2', *flags, str(source), '-o', str(binary)],
                       check=True, capture_output=True, text=True)
        binaries.append(binary)
    return binaries


@pytest.mark.parametrize('options', [[], ['-c', '0', '-s', '12', '-n', '4'],
                                   ['-e', '-n', '4', '-s', '12'], ['-m', '-n', '4']])
def test_parallel_matches_serial(ocg_binaries, tmp_path, options):
    rng = random.Random(729)
    # More than 256 centered initial classes: enter the actual OpenMP regions.
    edges = {(i, (i + 1) % 360) for i in range(360)}
    edges.update(tuple(sorted(rng.sample(range(360), 2))) for _ in range(650))
    graph = tmp_path / 'edges.txt'
    graph.write_text(''.join(f'{u:04d} {v:04d}\n' for u, v in sorted(edges)))
    outputs = []
    for binary, threads in [(ocg_binaries[0], 1), (ocg_binaries[1], 1),
                            (ocg_binaries[1], 4), (ocg_binaries[1], 4)]:
        subprocess.run([str(binary), *options, '-p', 'partition.txt', 'edges.txt'],
                       cwd=tmp_path, env=dict(os.environ, OMP_NUM_THREADS=str(threads)),
                       capture_output=True, check=True, timeout=60)
        outputs.append((tmp_path / 'partition.txt').read_text())
    assert all(output == outputs[0] for output in outputs)
    assigned = {line for line in outputs[0].splitlines()
                if line and not line.startswith(('#', '>'))}
    assert assigned == {f'{i:04d}' for i in range(360)}


def test_weighted_parallel_matches_serial(ocg_binaries, tmp_path):
    (tmp_path / 'edges.txt').write_text(
        'a b 0.5\nb c 1.5\na c 2\nc d 0.75\nd e 2\ne f 1\nd f 1.25\n')
    outputs = []
    for binary in ocg_binaries:
        subprocess.run([str(binary), '-w', '-p', 'partition.txt', 'edges.txt'],
                       cwd=tmp_path, capture_output=True, check=True, timeout=10)
        outputs.append((tmp_path / 'partition.txt').read_text())
    assert outputs[0] == outputs[1]
