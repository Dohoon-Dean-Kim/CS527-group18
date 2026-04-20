import pandas as pd
import os

def create_heuristic_edges(nodes_root="output/nodes", output_root="output/edges_heuristic"):
    """
    Use heuristic (name-based matching) to generate edges between tests and source files.
    """
    for root, dirs, files in os.walk(nodes_root):
        for f in files:
            if f.endswith("_nodes.csv"):
                nodes_df = pd.read_csv(os.path.join(root, f))
                
                # Node Separation
                tests = nodes_df[nodes_df['Type'] == 'Test']
                sources = nodes_df[nodes_df['Type'] == 'Source_File']
                
                edges = []
                
                for _, test_row in tests.iterrows():
                    test_full_name = test_row['Name'] # ex: org.apache.activemq.BrokerTest
                    # Extract only the class name
                    test_class_name = test_full_name.split('.')[-1].replace('Test', '')
                    
                    for _, src_row in sources.iterrows():
                        src_path = src_row['Name'] # ex: .../org/apache/activemq/Broker.java
                        
                        # Heuristic matching if the class name is included in the path
                        if test_class_name in src_path:
                            edges.append({
                                'Source_ID': test_row['Node_ID'],
                                'Target_ID': src_row['Node_ID'],
                                'Type': 'TEST_COVERS_CODE'
                            })
                
                if edges:
                    edges_df = pd.DataFrame(edges)
                    project_name = os.path.basename(root)
                    project_out_dir = os.path.join(output_root, project_name)
                    os.makedirs(project_out_dir, exist_ok=True)
                    
                    out_filename = f.replace("_nodes.csv", "_edges.csv")
                    edges_df.to_csv(os.path.join(project_out_dir, out_filename), index=False)
                    print(f"{project_name} - {out_filename}: {len(edges_df)} edges")

if __name__ == "__main__":
    create_heuristic_edges()