# Local clustering tools

ClusterONE 1.0 is stored in `clusterone/cluster_one-1.0.jar`.
Both `native_clustering.py` and `pipelines/run_clusterone.py` use this JAR by default.
The native adapter accepts a `CLUSTERONE_JAR` override; the standalone pipeline accepts `--jar`.
Java must be installed (`JAVA` can override the native adapter executable).

`native/run-native-clustering.sh` resolves the repository relative to itself and uses
`tools/native/` for the WSL Java wrapper, OSLOM and temporary files.
OSLOM requires its separately installed executable in `tools/native/OSLOM2/oslom_undir`
or an `OSLOM_EXEC` override. Python environments and Java remain external prerequisites.
