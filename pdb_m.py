import pdb
import runpy
import sys


def main():
    module = sys.argv[1]
    sys.argv[1:] = sys.argv[2:]
    pdb.runcall(runpy.run_module, module, run_name='__main__')


__name__ == '__main__' and main()
