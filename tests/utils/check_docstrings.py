"""This library contains functions that checks the docstrings


Header info specific to ESP
"""

import os
import re


def extractName(s: str, search_class: bool = False) -> str:
    """Extracts the names of the function or classes in the input string.

    Arguments
    ---------
    s: str
        Input string where to search for function or class names.
    search_class: bool
        If True, searches for class names.

    Returns
    -------
    string: str
        Name of the function or class detected.
    """
    string = ""
    if search_class:
        regexp = re.compile(r"(class)\s(.*)\:")
    else:
        regexp = re.compile(r"(def)\s(.*)\(.*\)\:")
    for m in regexp.finditer(s):
        string += m.group(2)
    return string


def check_docstrings(base_folder: str = ".", check_folders: list = None) -> bool:
    """Checks if all the functions or classes have a docstring.

    Arguments
    ---------
    base_folder: path
        The main folder of speechbrain.
    check_folders: list
        List of subfolders to check.

    Returns
    -------
    check: bool
        True if all the functions/classes have a docstring, False otherwise.
    """
    # Search all python libraries in the folder of interest
    lib_lst = get_all_files(
        base_folder,
        match_and=[".py"],
        match_or=check_folders,
        exclude_or=[".pyc"],
    )
    check = True
    # Loop over the detected libraries
    for libpath in lib_lst:
        if "__" in libpath:
            continue
        print("Checking %s..." % (libpath))

        # Support variable initialization
        fun_name = libpath
        class_name = libpath
        check_line = True
        is_class = False
        first_line = True
        with open(libpath, encoding="utf-8") as f:
            for line in f:
                # Remove spaces or tabs
                line = line.strip()

                # Avoid processing lines with the following patterns
                if ">>>" in line:
                    continue
                if "..." in line:
                    continue
                if len(line) == 0:
                    continue
                if line[0] == "#":
                    continue

                # Check if the docstring is written after the class/funct declaration
                if check_line:
                    if line[0] != '"' and not is_class:
                        if line[0:2] == 'r"':
                            check_line = False
                            continue
                        check = False
                        if first_line:
                            print(
                                "\tERROR: The library %s must start with a docstring. "
                                % (libpath)
                                + "Please write it."
                            )
                        else:
                            print(
                                "\tERROR: The function %s in %s has no docstring. "
                                % (fun_name, libpath)
                                + "Please write it."
                            )
                    if line[0] != '"' and is_class:
                        if line[0:2] == 'r"':
                            check_line = False
                            continue

                        check = False
                        print(
                            "\tERROR: The class %s in %s has no docstring. "
                            % (class_name, libpath)
                            + "Please write it."
                        )

                    # Update support variables
                    check_line = False
                    is_class = False
                    first_line = False
                    continue

                # Extract function name (if any)
                fun_name = extractName(line)
                if len(fun_name) > 0 and line[0:3] == "def":
                    if fun_name[0] == "_":
                        continue
                    check_line = True

                # Extract class name (if any)
                class_name = extractName(line, search_class=True)
                if len(class_name) > 0 and line[0:5] == "class":
                    check_line = True
                    is_class = True
    return check


def get_all_files(
    dirName: str,
    match_and: list = None,
    match_or: list = None,
    exclude_and: list = None,
    exclude_or: list = None,
) -> list:
    """Returns a list of files found within a folder.

    Different options can be used to restrict the search to some specific
    patterns.

    Arguments
    ---------
    dirName : str
        The directory to search.
    match_and : list
        A list that contains patterns to match. The file is
        returned if it matches all the entries in `match_and`.
    match_or : list
        A list that contains patterns to match. The file is
        returned if it matches one or more of the entries in `match_or`.
    exclude_and : list
        A list that contains patterns to match. The file is
        returned if it matches none of the entries in `exclude_and`.
    exclude_or : list
        A list that contains pattern to match. The file is
        returned if it fails to match one of the entries in `exclude_or`.

    Returns
    -------
    allFiles : list
        The list of files matching the patterns.

    Example
    -------
    >>> get_all_files('tests/', match_and=['test_docstrings.py'])
    ['tests/consistency/test_docstrings.py']
    """
    # Match/exclude variable initialization
    match_and_entry = True
    match_or_entry = True
    exclude_or_entry = False
    exclude_and_entry = False

    # Create a list of file and sub directories
    listOfFile = os.listdir(dirName)
    allFiles = list()

    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)

        # If entry is a directory then get the list of files in this directory
        if os.path.isdir(fullPath):
            allFiles = allFiles + get_all_files(
                fullPath,
                match_and=match_and,
                match_or=match_or,
                exclude_and=exclude_and,
                exclude_or=exclude_or,
            )
        else:
            # Check match_and case
            if match_and is not None:
                match_and_entry = False
                match_found = 0

                for ele in match_and:
                    if ele in fullPath:
                        match_found = match_found + 1
                if match_found == len(match_and):
                    match_and_entry = True

            # Check match_or case
            if match_or is not None:
                match_or_entry = False
                for ele in match_or:
                    if ele in fullPath:
                        match_or_entry = True
                        break

            # Check exclude_and case
            if exclude_and is not None:
                match_found = 0

                for ele in exclude_and:
                    if ele in fullPath:
                        match_found = match_found + 1
                if match_found == len(exclude_and):
                    exclude_and_entry = True

            # Check exclude_or case
            if exclude_or is not None:
                exclude_or_entry = False
                for ele in exclude_or:
                    if ele in fullPath:
                        exclude_or_entry = True
                        break

            # If needed, append the current file to the output list
            if (
                match_and_entry
                and match_or_entry
                and not (exclude_and_entry)
                and not (exclude_or_entry)
            ):
                allFiles.append(fullPath)

    return allFiles
