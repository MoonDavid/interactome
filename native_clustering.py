"""Original ClusterONE and OSLOM executables, isolated from optional Python backends."""
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory

CLUSTERONE_JAR = Path(__file__).resolve().parent / 'tools' / 'clusterone' / 'cluster_one-1.0.jar'


def executable(value):
    resolved = shutil.which(value)
    if resolved is None:
        raise ImportError(f'Executable unavailable: {value}. Configure JAVA or OSLOM_EXEC.')
    return str(Path(resolved).resolve())


def tool_fingerprints():
    """Include tool changes in the runner's environment and resume identity."""
    paths = {
        'clusterone': Path(os.environ.get('CLUSTERONE_JAR', str(CLUSTERONE_JAR))).resolve(),
        'oslom': Path(shutil.which(os.environ.get('OSLOM_EXEC', 'oslom_undir')) or '__missing_oslom__'),
    }
    return {name: {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
            if path.is_file() else None for name, path in paths.items()}


def execute_native(name, graph, seed):
    """Return native overlapping covers; never fill noise nodes with synthetic groups."""
    if name == 'clusterone':
        jar = Path(os.environ.get('CLUSTERONE_JAR', str(CLUSTERONE_JAR))).resolve()
        if not jar.is_file():
            raise ImportError('ClusterONE requires its original JAR; set CLUSTERONE_JAR to its path.')
        command = [executable(os.environ.get('JAVA', 'java')), '-jar', str(jar)]
        artifact = jar
        options = ['--input-format', 'edge_list', '--output-format', 'plain']
    elif name == 'oslom':
        artifact = Path(executable(os.environ.get('OSLOM_EXEC', 'oslom_undir')))
        command = [str(artifact)]
        options = ['-uw', '-seed', str(max(1, int(seed))), '-r', '10', '-hr', '10']
    else:
        raise ValueError(f'Unknown native clustering method: {name}')
    with TemporaryDirectory(prefix=f'atlas_{name}_') as temporary:
        edges = Path(temporary) / 'edges.txt'
        with edges.open('w', encoding='utf-8') as stream:
            for u, v in zip(graph['u'], graph['v']):
                stream.write(f'{int(u)} {int(v)}\n')
        args = command + options + ([str(edges)] if name == 'clusterone' else ['-f', str(edges)])
        result = subprocess.run(args, cwd=temporary, capture_output=True, text=True,
                                check=True, timeout=300)
        if name == 'clusterone':
            output = result.stdout
        else:
            # tp is the finest native level; tp1, tp2, ... are higher hierarchy levels.
            output = Path(str(edges) + '_oslo_files', 'tp').read_text(encoding='utf-8')
        groups = []
        for line in output.splitlines():
            line = line.strip()
            if line and not line.startswith(('#', '%')):
                groups.append({int(node) for node in line.split()})
    return groups, {
        'implementation': f'original {name} command-line executable',
        'artifact': str(artifact),
        'artifact_sha256': hashlib.sha256(artifact.read_bytes()).hexdigest(),
        'options': options,
        'semantics': 'undirected binary topology; native overlapping cover; noise remains unassigned',
        'hierarchy_level': 'finest (tp)' if name == 'oslom' else None,
    }
