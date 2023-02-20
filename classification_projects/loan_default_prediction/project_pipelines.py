"""
This module provides functions for performing various mathematical operations.
"""

import re

class Node:
    """
    This is a docstring that describes what the class does.
    """
    def __init__(self, func, inputs, outputs, name):
        self.func = func
        self.inputs = inputs if inputs is not None else []
        self.outputs = outputs if outputs is not None else []
        self.name = name

class Pipeline:
    """
    This is a docstring that describes what the class does.
    """
    def __init__(self):
        self.nodes = []

    def add_node(self, node):
        """
        This is a docstring that describes what the function does.
        Parameters:
        arg1 (int): The first argument.
        arg2 (str): The second argument.
        Returns:
        str: The result of the function.
        """
        #print(f'{node.name} - {node.inputs}')
        self.nodes.append(node)

    def node_dependencies(self):
        """
        This is a docstring that describes what the function does.
        Parameters:
        arg1 (int): The first argument.
        arg2 (str): The second argument.
        Returns:
        str: The result of the function.
        """
        nodes_deps = []
        for node in self.nodes:
            deps = []
            name = ""

            for other_node in self.nodes:
                if node.name == other_node.name:
                    break
                name = other_node.name

            deps.append(name)
            nodes_deps.append({
                'node': node.func.__name__, 
                'name': node.name, 
                'deps': deps, 
                'inputs': node.inputs,
                'outputs': node.outputs
            })
        return nodes_deps

def create_pipeline(nodes):
    """
    This is a docstring that describes what the function does.
    Parameters:
    arg1 (int): The first argument.
    arg2 (str): The second argument.
    Returns:
    str: The result of the function.
    """
    pipeline = Pipeline()
    for node in nodes:
        pipeline.add_node(node)
    return pipeline

# def clean_name(name):
#     """
#     This is a docstring that describes what the function does.
#     Parameters:
#     arg1 (int): The first argument.
#     arg2 (str): The second argument.
#     Returns:
#     str: The result of the function.
#     """
#     return re.sub(r"[\W_]+", "-", name).strip("-")

# def convert_to_kebab_case(string):
#     string = string.lower()
#     kebab_case = ""
#     for char in string:
#         if char.isupper():
#             kebab_case += "-" + char.lower()
#         else:
#             kebab_case += char
#     return "--" + kebab_case

# def flatten_list_of_lists(lst):
#     return [item for sublist in lst for item in sublist]