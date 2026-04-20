import pandas as pd
import json
import os
import zipfile
import io
import re

def get_files_from_diff(diff_path):
    """
    Extract the list of changed files from the diff file
    """
    changed_files = []
    if os.path.exists(diff_path):
        with open(diff_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                if line.startswith('+++ b/'): # Extract modified file paths
                    changed_files.append(line[6:].strip())
                    
    return changed_files

def find_shadata_files(project_name, pr_name, build_id, shadata_root="shadata"):
    """
    Find the JSON folder and Diff files
    """
    changed_files = []
    
    # Search the JSON folder
    base_search_path = os.path.join(shadata_root, project_name, "compare_commits")
    if os.path.exists(base_search_path):
        # Browse folders containing PR numbers
        for folder in os.listdir(base_search_path):
            if pr_name in folder: # Search if the PR number matches
                json_folder = os.path.join(base_search_path, folder)
                if os.path.isdir(json_folder):
                    for f in os.listdir(json_folder):
                        if f.endswith(".json"):
                            try:
                                with open(os.path.join(json_folder, f), 'r', encoding='utf-8') as f:
                                    meta = json.load(f)
                                    if 'files' in meta:
                                        changed_files.extend([fi['filename'] for fi in meta['files']])
                            except:
                                pass

    # If not found in JSON, search the Diff folder (shadata/[project]/compare_commits/diff/)
    if not changed_files:
        diff_dir = os.path.join(shadata_root, project_name, "compare_commits", "diff")
        if os.path.exists(diff_dir):
            # Search for diff files containing the PR number and build number
            diff_pattern = f"{pr_name}_{build_id}.diff"
            diff_path = os.path.join(diff_dir, diff_pattern)
            changed_files = get_files_from_diff(diff_path)
            
    return list(set(changed_files))

def extract_nodes(test_result_dir="processed_test_result", shadata_dir="shadata", output_root="output/nodes"):
    for root, dirs, files in os.walk(test_result_dir):
        for f in files:
            if f.endswith(".zip"):
                zip_path = os.path.join(root, f)
                
                rel_path = os.path.relpath(root, test_result_dir)
                path_parts = rel_path.split(os.sep)
                project_name, pr_name, build_id = path_parts[0], path_parts[1], path_parts[2]
                changed_files = find_shadata_files(project_name, pr_name, build_id, shadata_dir)

                try:
                    with zipfile.ZipFile(zip_path, 'r') as z:
                        if 'test_class.csv' in z.namelist():
                            with z.open('test_class.csv') as f:
                                df = pd.read_csv(io.BytesIO(f.read()))
                                test_nodes = pd.DataFrame({'Name': df['testclass'].unique(), 'Type': 'Test', 'Is_Changed': 0})
                                source_nodes = pd.DataFrame({'Name': changed_files, 'Type': 'Source_File', 'Is_Changed': 1})
                                
                                nodes_df = pd.concat([test_nodes, source_nodes], ignore_index=True)
                                nodes_df['Node_ID'] = range(len(nodes_df))
                                
                                project_out_dir = os.path.join(output_root, project_name)
                                os.makedirs(project_out_dir, exist_ok=True)
                                nodes_df.to_csv(os.path.join(project_out_dir, f"{pr_name}_{build_id}_nodes.csv"), index=False)
                                
                                print(f"{project_name} - {pr_name}_{build_id}: {len(changed_files)} changed files")

                except Exception as e:
                    print(f"Error: {e}")

if __name__ == "__main__":
    extract_nodes()