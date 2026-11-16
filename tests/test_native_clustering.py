"""Native adapter contracts without requiring third-party executables."""
from pathlib import Path
from types import SimpleNamespace

import pytest

import native_clustering as native


@pytest.mark.parametrize('name', ['clusterone', 'oslom'])
def test_native_overlap_and_noise(name, tmp_path, monkeypatch):
    artifact = tmp_path / 'algorithm.bin'
    artifact.write_bytes(b'test executable')
    monkeypatch.setenv('CLUSTERONE_JAR', str(artifact))
    monkeypatch.setattr(native, 'executable', lambda _: str(artifact))

    def run(args, **kwargs):
        edges = Path(kwargs['cwd']) / 'edges.txt'
        assert edges.read_text() == '0 1\n1 2\n2 3\n'
        if name == 'oslom':
            assert '-uw' in args and args[args.index('-seed') + 1] == '42'
            directory = Path(str(edges) + '_oslo_files')
            directory.mkdir()
            (directory / 'tp').write_text('#module 0 size: 2 bs: 0.01\n0 1\n#module 1\n1 2\n')
            (directory / 'tp1').write_text('0 1 2 3\n')
        return SimpleNamespace(stdout='0\t1\n1\t2\n')

    monkeypatch.setattr(native.subprocess, 'run', run)
    groups, details = native.execute_native(name, {'u': [0, 1, 2], 'v': [1, 2, 3]}, 42)
    assert groups == [{0, 1}, {1, 2}]
    assert 3 not in set.union(*groups)
    assert details['artifact_sha256']


def test_missing_clusterone_is_unavailable(tmp_path, monkeypatch):
    monkeypatch.setenv('CLUSTERONE_JAR', str(tmp_path / 'missing.jar'))
    with pytest.raises(ImportError, match='CLUSTERONE_JAR'):
        native.execute_native('clusterone', {}, 42)


def test_missing_oslom_is_unavailable(monkeypatch):
    monkeypatch.setattr(native.shutil, 'which', lambda _: None)
    with pytest.raises(ImportError, match='OSLOM_EXEC'):
        native.execute_native('oslom', {}, 42)
