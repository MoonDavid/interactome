"""Compatibility entry point; annotation implementation lives in clusterannotation."""
import sys
import clusterannotation as _implementation

if __name__ == '__main__':
    _implementation.main()
else:
    sys.modules[__name__] = _implementation
