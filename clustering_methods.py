"""Clustering method inventory without optional CLI or backend imports."""
from dataclasses import dataclass

@dataclass(frozen=True)
class Method:
    id: str
    backend: str
    name: str
    family: str = 'hard'
    costly: bool = False
    dense: bool = False

def catalog() -> dict[str, Method]:
    from moonlighting import CLUSTERING_CATALOG
    methods = [Method(f'atlas.{name}', 'atlas', name, spec['paradigm'],
                      name not in {'louvain', 'label_propagation', 'k_core_shells'})
               for name, spec in CLUSTERING_CATALOG.items()]

    def add(backend, names, family='hard', costly=False, dense=False):
        methods.extend(Method(f'{backend}.{name}', backend, name, family, costly, dense)
                       for name in names.split())

    add('networkx', 'louvain_communities fast_label_propagation_communities '
        'asyn_lpa_communities label_propagation_communities')
    add('networkx', 'greedy_modularity_communities girvan_newman', 'hierarchical', True)
    add('networkx', 'k_clique_communities', 'overlapping', True)
    add('igraph', 'multilevel leiden infomap label_propagation')
    add('igraph', 'fastgreedy walktrap leading_eigenvector edge_betweenness', 'hierarchical', True)
    add('leidenalg', 'RBConfigurationVertexPartition CPMVertexPartition ModularityVertexPartition')
    add('sknetwork', 'Louvain Leiden PropagationClustering')
    add('sknetwork_hierarchy', 'Paris LouvainHierarchy', 'hierarchical', True)
    add('graspologic_partition', 'leiden hierarchical_leiden')
    add('graspologic_cluster', 'AutoGMMCluster KMeansCluster', 'embedding', True)
    add('sklearn', 'KMeans MiniBatchKMeans Birch DBSCAN OPTICS MeanShift', 'embedding', True)
    add('sklearn', 'AffinityPropagation AgglomerativeClustering', 'embedding', True, True)
    add('sklearn', 'SpectralClustering', 'spectral', True)
    add('cdlib', 'louvain leiden infomap label_propagation')
    add('cdlib', 'slpa angel kclique', 'overlapping', True)
    add('cdlib', 'walktrap surprise_communities significance_communities', 'hard', True)
    add('karateclub', 'BigClam DANMF MNMF NNSED SymmNMF', 'soft', True)
    add('karateclub', 'EdMot SCD GEMSEC', 'hard', True)
    add('networkit', 'PLM PLP ParallelLeiden')
    add('cugraph', 'louvain leiden ecg')
    add('infomap', 'Infomap')
    add('markov_clustering', 'mcl', 'flow', True)
    add('hdbscan', 'HDBSCAN', 'embedding', True)
    add('native', 'clusterone oslom', 'overlapping', True)
    return {m.id: m for m in methods}
