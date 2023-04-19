import ast
import torch
import inspect
from torch import Tensor
import networkx as nx
import matplotlib.pyplot as plt

# Get the source code of Tensor class
source_code = inspect.getsource(Tensor)

# Add the inheritance relationship with torch._C._TensorBase
source_code = source_code.replace("class Tensor(", "class Tensor(torch._C._TensorBase, ")

# Parse the source code into an AST
tree = ast.parse(source_code)

# Traverse the AST using the custom visitor
extractor = InformationExtractor()
extractor.visit(tree)

# Create a directed graph for the IR
ir_graph = nx.DiGraph()

# Add nodes for classes and functions
for class_name in extractor.classes:
    ir_graph.add_node(class_name, type="class")

for function_name in extractor.functions:
    ir_graph.add_node(function_name, type="function")

# Add the torch._C._TensorBase node
ir_graph.add_node("torch._C._TensorBase", type="class")

# Add edges for inheritance relationships
for child, parents in extractor.inheritance.items():
    for parent in parents:
        ir_graph.add_edge(child, parent, type="inheritance")

# Add edges for compositions
for source, target in extractor.compositions.items():
    if target in ir_graph.nodes:
        ir_graph.add_edge(source, target, type="composition")

# Add edges for function calls
for source, target in zip(extractor.functions, extractor.calls):
    if target in ir_graph.nodes:
        ir_graph.add_edge(source, target, type="call")

# Print the extracted information and IR graph
print("Imports:", extractor.imports)
print("Classes:", extractor.classes)
print("Inheritance:", extractor.inheritance)
print("Functions:", extractor.functions)
print("Compositions:", extractor.compositions)
print("Function Calls:", extractor.calls)
print("Intermediate Representation (IR) Graph:")
print(ir_graph.nodes.data())
print(ir_graph.edges.data())

# Visualize the IR graph
pos = nx.spring_layout(ir_graph)
nx.draw(ir_graph, pos, with_labels=True, font_weight="bold", node_color="skyblue", node_size=3000)
edge_labels = {(u, v): d["type"] for u, v, d in ir_graph.edges(data=True)}
nx.draw_networkx_edge_labels(ir_graph, pos, edge_labels=edge_labels)

plt.show()
