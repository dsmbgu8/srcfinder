import os
import argparse
from datetime import datetime, timezone
import re

class DirWatcher:
    """This class that can check a local directory for files that match
    a regex and have changed since the last time it was checked.
    """
    def __init__(self, local_dir, force=False, regex='.*'):
        """Set up local directory and last run time tracker.

        :param local_dir: local directory path
        :param force: if True, matches a file regardless of whether or not
                      it changed since the last check
        :param regex: only match files that match the indicated regex
       """
        # Confirm input directory exists.
        if not os.path.isdir(local_dir):
            raise ValueError('Directory {} does not exist.'.format(local_dir))
        self._local_dir = local_dir

        # Configure last run time tracker
        self._datetime_fmt = "%Y-%m-%dT%H:%M:%S%z"
        self._lrt_fname = os.path.join(local_dir, '.last_run_time')
        if force:
            # Set last run time to a long time ago so everyting looks new 
            # Here I use Jan 1 in year 1.
            self._last_run_time = datetime(1,1,1, tzinfo=timezone.utc)
        else:
            # Get last run time from file.
            self._last_run_time = self.get_last_run_time()
        self.update_last_run_time()

        # Configure filename match pattern
        self._matcher = re.compile(regex)

    def get_last_run_time(self):
        """Get the last time this program was run with the current local
        directory.  The last run time is stored in a file.  If this is the 
        first time this program was run with the current local directory
        then this file will not exist.  In that case, use the current time.

        :returns: the last run time in UTC as a datetime object
        """
        if os.path.isfile(self._lrt_fname):
            with open(self._lrt_fname, "r") as f:
                date_str = f.readline().strip()
                last_run_time = datetime.strptime(date_str, self._datetime_fmt)
        else:
            last_run_time = datetime.now(timezone.utc)
        return last_run_time
    
    def update_last_run_time(self):
        """Get the current time, convert it to a string, and save it to a file.
        """
        cur_time = datetime.strftime(datetime.now(timezone.utc), 
                                     self._datetime_fmt)
        with open(self._lrt_fname, "w") as f:
            f.write(cur_time)

    def whats_new_local(self):
        """Get a list of all the files in the local directory that have 
        changed since the last time this program was run with that directory.

        :returns: list of new or updated files
        """
        new_files = []
        for root, dirs, files in os.walk(self._local_dir, followlinks=True):
            # Skip hidden directories and files
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            files_full_path = [os.path.join(root, f) for f in files
                               if not f.startswith('.')]

            # Skip files that don't match provided regex pattern and
            # check time stamps on each file to get just what's new.
            new_files += [f for f in files_full_path
                          if (self._matcher.search(f) and
                              datetime.fromtimestamp(os.stat(f).st_mtime).astimezone() > self._last_run_time)]
        return new_files

def parse_args():
    """Retrieve command line parameters.

    Returns:
        ArgumentParse: command line parameters
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-d", "--watchdir", required=True,
                        help="path to directory to watch")
    args = parser.parse_args()
    return args.watchdir

def main():
    watchdir = parse_args()
    watcher = DirWatcher(watchdir)
    new_files = watcher.whats_new_local()
    print(new_files)


if __name__ == "__main__":
    main()
