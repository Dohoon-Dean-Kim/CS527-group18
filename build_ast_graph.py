import os
import networkx as nx
import javalang

def build_ast_call_graph(repo_path):
    G = nx.DiGraph()
    
    for root, dirs, files in os.walk(repo_path):
        for file in files:
            # Target only the actual production code 
            if file.endswith(".java") and "src/main/java" in root:
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        source = f.read()
                    
                    tree = javalang.parse.parse(source)
                    
                    for _, class_node in tree.filter(javalang.tree.ClassDeclaration):
                        class_name = class_node.name
                        for _, method_node in class_node.filter(javalang.tree.MethodDeclaration):
                            caller = f"{class_name}.{method_node.name}"
                            
                            for _, invocation_node in method_node.filter(javalang.tree.MethodInvocation):
                                callee = invocation_node.member
                                G.add_edge(caller, callee)
                except Exception:
                    continue
                    
    return G

if __name__ == "__main__":
    history_dir = "history"
    output_dir = "graphs"
    
    for project_name in os.listdir(history_dir):
        trunk_path = os.path.join(history_dir, project_name, f"{project_name}_trunk")
        
        # Extract graph
        graph = build_ast_call_graph(trunk_path)

        output_file = os.path.join(output_dir, f"{project_name}_callgraph.csv")
        nx.write_edgelist(graph, output_file, delimiter=',', data=False)