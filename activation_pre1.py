import ast
import torch
import inspect
from torch.nn.modules import activation

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