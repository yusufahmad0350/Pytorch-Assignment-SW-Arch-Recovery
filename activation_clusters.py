import ast
import torch
import inspect
from torch.nn.modules import activation
import networkx as nx
import community as community_louvain
import leidenalg
import igraph as ig

# Get the source code of activation
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




# Create a directed graph for the IR
ir_graph = nx.DiGraph()

# Add nodes for classes and functions
for class_name in extractor.classes:
    ir_graph.add_node(class_name, type="class")

for function_name in extractor.functions:
    ir_graph.add_node(function_name, type="function")

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

# Print the IR graph
print("Intermediate Representation (IR) Graph:")
print(ir_graph.nodes.data())
print(ir_graph.edges.data())

# Visualize the IR graph
import matplotlib.pyplot as plt

pos = nx.spring_layout(ir_graph)
nx.draw(ir_graph, pos, with_labels=True, font_weight="bold", node_color="skyblue", node_size=3000)
edge_labels = {(u, v): d["type"] for u, v, d in ir_graph.edges(data=True)}
nx.draw_networkx_edge_labels(ir_graph, pos, edge_labels=edge_labels)

plt.show()
###############################################################
# Convert NetworkX graph to iGraph graph
ig_graph = ig.Graph.from_networkx(ir_graph)
# Ensure that all vertices in the IR graph are present in the iGraph graph
for node, data in ir_graph.nodes(data=True):
    try:
        ig_node = ig_graph.vs.find(name=node)
    except ValueError:
        ig_graph.add_vertex(name=node, **data)
        ig_node = ig_graph.vs.find(name=node)

    



# Apply the Leiden algorithm for community detection
partition = leidenalg.find_partition(ig_graph, leidenalg.ModularityVertexPartition)

# Calculate modularity
modularity = partition.modularity
print("Modularity:", modularity)

# Convert the partition dictionary into a list of sets
clusters = {}
for node, cluster_id in enumerate(partition.membership):
    if cluster_id not in clusters:
        clusters[cluster_id] = set()
    clusters[cluster_id].add(ig_graph.vs[node]["name"])


cluster_list = list(clusters.values())
print("Cluster List:", cluster_list)

# Define the Jaccard index function
def jaccard_index(set1, set2):
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union

# The actual structure from PyTorch in phase 1
actual_structure = [
    {"ReLU"}, {"ReLU6"}, {"ELU"}, {"CELU"},{"GLU"}, {"Hardtanh"}, {"Hardshrink"},{"Hardsigmoid"}, {"Hardswish"}, {"LeakyReLU"},{"LogSigmoid"}, {"LogSoftmax"}, {"MultiheadAttention"},{"PReLU"}, {"RReLU"}, {"Sigmoid"}, {"SiLU"}]
  
# Compare the actual structure with the clustering results
best_matches = []
print("Best Matches:", best_matches)

for module in actual_structure:
    best_match = None
    best_similarity = 0
    
    for cluster in cluster_list:
        similarity = jaccard_index(module, cluster)
        if similarity > best_similarity:
            best_similarity = similarity
            best_match = cluster
            
    best_matches.append((module, best_match, best_similarity))

for module, best_match, similarity in best_matches:
    print(f"Module {module} best matches cluster {best_match} with Jaccard similarity {similarity}")
