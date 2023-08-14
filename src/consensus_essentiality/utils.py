"""
Module with utility functions for consensus_essentiality
"""
import os
import re
import sys


class HiddenPrints:
    """
    Context manager to stop methods within the context from printing
    """

    def __enter__(self):
        """
        The __enter__ function is called when the context manager is entered.
        It returns whatever object should be assigned to the variable in the as clause of a with statement.

        :param self: Access the class attributes and methods
        :return: Nothing
        :doc-author: Trelent
        """
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        The __exit__ function is called when the context manager exits.
        It can be used to clean up resources, such as closing a file or database connection.

        :param self: Represent the instance of the class
        :param exc_type: Store the exception type that was raised
        :param exc_val: Store the exception value
        :param exc_tb: Get the traceback object
        :return: Nothing
        :doc-author: Trelent
        """
        sys.stdout.close()
        sys.stdout = self._original_stdout


def parse_enum_method(method: str):
    """
    The parse_enum_method function takes a string and returns the corresponding enumeration method.

    :param method: Method string to parse
    :type method: str
    :return: A string representing the enumeration method
    :rtype: str
    :doc-author: Trelent
    """
    if method[0:3].lower() == "div":
        return "diversity"
    elif method[0:4].lower() == "icut":
        return "icut"
    elif method[0:3].lower() == "max":
        return "max-dist"
    else:
        raise ValueError("Couldn't Parse Enumeration Method: %s" % method)


def parse_model_method(method: str):

    """
    The parse_model_method function takes a string and parses it into one of the following strings:
        - enforce_active
        - enforce_inactive
        - enforce_inactive_off
        - enforce_off

    :param method: Specify the method used to train the model
    :type method: str
    :return: A string describing the model method
    :rtype: str
    :doc-author: Trelent
    """
    inactive_pattern = re.compile("inact[ive]*", re.IGNORECASE)
    active_pattern = re.compile("(?<!in)act[ive]*", re.IGNORECASE)
    off_pattern = re.compile("off", re.IGNORECASE)
    enforce_pattern = re.compile("enf[orce]*", re.IGNORECASE)
    both_pattern = re.compile("both", re.IGNORECASE)
    active_flag = bool(active_pattern.search(method))
    inactive_flag = bool(inactive_pattern.search(method))
    off_flag = bool(off_pattern.search(method))
    enforce_flag = bool(enforce_pattern.search(method))
    both_flag = bool(both_pattern.search(method))
    if enforce_flag:
        if both_flag:
            if not off_flag:
                return "enforce_both"
            else:
                raise NotImplementedError("Enforcing both and off constraints not currently implemented")
        elif active_flag:
            if not (inactive_flag or off_flag):
                return "enforce_active"
            else:
                if inactive_flag:
                    raise ValueError("Can't enforce active and inactive simultaneously")
                if off_flag:
                    raise NotImplementedError("Enforcing active and off simultaneously not currently implemented")
        elif inactive_flag:
            if off_flag:
                return "enforce_inactive_off"
            else:
                return "enforce_inactive"
        elif off_flag:
            return "enforce_off"
        else:
            raise ValueError("Couldn't parse context specific model method: %s" % method)
    else:
        raise ValueError("Couldn't parse context specific model method: %s" % method)
