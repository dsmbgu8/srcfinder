# msf_flow.utils

General utility functions of use in the other modules.


## dir_watcher

`class` **`DirWatcher`**`(`*`local_dir, force=False, regex='.\*'`*`)`

### Methods:

**`whats_new_local`**`()`

Traverse a local directory and return a list of files that are new since
the last time the program was run.  The current time is used for the first
run so that nothing is new.  The `regex` parameter can be passed to restrict
the files to only those that match the indicated regular expression.  The
default regex matches all file names.  Setting `force=True` will return all
matching files regardless of when they were last modified.

### Example:

```
from msf_flow.utils.dir_watcher import DirWatcher

# Match only new files (since the last run) with *.txt extension.
watcher = DirWatcher("/data/mydata", regex="\.txt$")
new_files = watcher.whats_new_local()
```

### Monitoring a read-only directory

This utility tracks the last run time by writing a small hidden file in to
the directory being watched.  If you would like to watch for new files in 
a read-only directory for which you lack write access, do the following:

1. Create a new directory to be watched in a location where you do have 
write access.

        mkdir my-writeable-dir

2. In the directory you just created in step 1 above, create a symbolic link
pointing to the read-only directory that you would like to watch.

        cd my-writeable-dir
        ln -s /path/to/read-only-dir-to-be-watched

3. Create the 'DirWatcher' for the directory you created in step 1 above.

        watcher = DirWatcher("my-writeable-dir")

##logger

### Functions:

**`init_logger`**`(`*`log_level=logging.WARNING`*`)`

Create a `logging.Logger` and initialize it with the indicated logging level.

### Example:

```
from msf_flow.utils.logger import init_logger

logger = init_logger(logging.info)
logger.info("hello world")
```

## r_runner

### Functions:

**`r_runner`**`(`*`rscript_cmd`*`)`

Given an R script command line (script name and arguments), run the script
and return the `stdout` and `stderr`.

### Example:

```
from msf_flow.utils.r_runner import r_runner

r_cmd = "/proj/scripts/my_script.R arg1 arg2"
r_output = r_runner(r_cmd)
print("stdout: {}".format(r_output.stdout))
print("stderr: {}".format(r_output.stderr))
print("returncode: {}".format(r_output.returncode))
```
