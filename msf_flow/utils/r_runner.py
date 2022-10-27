import argparse
from subprocess import run, PIPE
import os, sys

def parse_args():
    """Retrieve command line parameters.
    
    Returns:
        ArgumentParse: command line parameters
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-r", "--rscript", required=True,
                        help="path to R script to run")
    args = parser.parse_args()
    return args.rscript

def r_runner(rscript_cmd):
    rscript_exec_name = "Rscript"
    rscript_name = rscript_cmd[0]
    rscript_dir = os.path.dirname(rscript_name)
    rscript_basename = os.path.basename(rscript_name)
    rscript_cmd[0] = rscript_basename
    cur_dir = os.getcwd()
    os.chdir(rscript_dir)
    cmd = [rscript_exec_name] + rscript_cmd
    r_output = run(cmd, stdout=PIPE, stderr=PIPE, universal_newlines=True)    
    os.chdir(cur_dir)
    return r_output

def main():
    rscript_name = parse_args()
    r_output = r_runner(rscript_name.split())
    print("stdout: {}".format(r_output.stdout))
    print("stderr: {}".format(r_output.stderr))
    print("returncode: {}".format(r_output.returncode))

if __name__ == "__main__":
    main()
