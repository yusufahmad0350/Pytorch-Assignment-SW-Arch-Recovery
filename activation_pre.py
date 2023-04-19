import ast
import torch
import inspect
from torch.nn.modules import activation
import networkx as nx

# Get the source code of ReLU class
source_code = inspect.getsource(activation)

# Parse the source code into an AST
tree = ast.parse(source_code)

class InformationExtractor(ast.NodeVisitor):
    def __init__(self):
        self.classes = []
        self.functions = []
        self.imports = []
        self.inheritance = {}
        self.compositions = {}
        self.calls = []

    def visit_Import(self, node):
        for alias in node.names:
            self.imports.append(alias.name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        for alias in node.names:
            self.imports.append(f"{node.module}.{alias.name}")
        self.generic_visit(node)

    def visit_ClassDef(self, node):
        self.classes.append(node.name)
        if node.bases:
            self.inheritance[node.name] = [base.id for base in node.bases]
        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        self.functions.append(node.name)
        self.generic_visit(node)

    def visit_Call(self, node):
        func = node.func
        if isinstance(func, ast.Attribute):
            self.calls.append(func.attr)
        elif isinstance(func, ast.Name):
            self.calls.append(func.id)
        self.generic_visit(node)

    def visit_Assign(self, node):
        if isinstance(node.value, ast.Call):
            call = node.value
            func = call.func
            for target in node.targets:
                if isinstance(target, ast.Name):
                    if isinstance(func, ast.Attribute):
                        self.compositions[target.id] = func.attr
                    elif isinstance(func, ast.Name):
                        self.compositions[target.id] = func.id
        self.generic_visit(node)

# Traverse the AST using the custom visitor
extractor = InformationExtractor()
extractor.visit(tree)

# Print the extracted information
print("Imports:", extractor.imports)
print("Classes:", extractor.classes)
print("Inheritance:", extractor.inheritance)
print("Functions:", extractor.functions)
print("Compositions:", extractor.compositions)
print("Function Calls:", extractor.calls)


class ArchitectureGraph:
    def __init__(self):
        self.graph = nx.DiGraph()

    def add_class(self, class_name, inheritance=None):
        self.graph.add_node(class_name, type="class")
        if inheritance:
            for base_class in inheritance:
                self.graph.add_edge(base_class, class_name, type="inheritance")

    def add_function(self, function_name):
        self.graph.add_node(function_name, type="function")

    def add_dependency(self, module_from, module_to):
        self.graph.add_edge(module_from, module_to, type="dependency")

    def add_data_flow(self, function_from, function_to):
        self.graph.add_edge(function_from, function_to, type="data_flow")

    def add_composition(self, class_from, class_to):
        self.graph.add_edge(class_from, class_to, type="composition")

# Create an ArchitectureGraph instance
arch_graph = ArchitectureGraph()

# Add classes and inheritance relationships
for class_name, bases in extractor.inheritance.items():
    arch_graph.add_class(class_name, bases)

# Add functions
for function_name in extractor.functions:
    arch_graph.add_function(function_name)

# Add dependencies
for module in extractor.imports:
    arch_graph.add_dependency('activation', module)

# Add data flow (function calls)
for idx, call in enumerate(extractor.calls[:-1]):
    arch_graph.add_data_flow(extractor.calls[idx], extractor.calls[idx+1])

# Add composition relationships
for obj, class_name in extractor.compositions.items():
    arch_graph.add_composition('activation', class_name)

# Print the graph edges
print("Edges in the architecture graph:")
for edge in arch_graph.graph.edges(data=True):
    print(edge)