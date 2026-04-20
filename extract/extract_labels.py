import pandas as pd
import os
import zipfile
import io

def extract_labels(base_dir, output_root="output/labels"):
    """
    Iterate through the ZIP files in the project folder and extract the labels
    """
    projects = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    
    for project in projects:
        project_path = os.path.join(base_dir, project)
        print(f"Processing Project: {project}")
        
        # Create a folder to save to
        project_output_dir = os.path.join(output_root, project)
        if not os.path.exists(project_output_dir):
            os.makedirs(project_output_dir)

        # Browse zip files
        for root, dirs, files in os.walk(project_path):
            for file in files:
                if file.endswith(".zip"):
                    zip_path = os.path.join(root, file)
                    build_id = file.replace(".csv.zip", "") # Extract build_id
                    
                    try:
                        with zipfile.ZipFile(zip_path, 'r') as z:
                            with z.open('test_class.csv') as f:
                                df = pd.read_csv(io.BytesIO(f.read()))
                                
                                # Create Label for Binary Classification
                                # If the outcome is 0, it is Success (0); otherwise, it is Failure (1)
                                labels_df = df[['testclass', 'outcome']].copy()
                                labels_df['Label'] = labels_df['outcome'].apply(lambda x: 1 if x != 0 else 0)
                                labels_df.rename(columns={'testclass': 'Node_Name'}, inplace=True)
                                labels_df['Node_ID'] = range(len(labels_df))
                                
                                # Save result (output/labels/project_name/PR-xxxx_buildId_labels.csv)
                                output_filename = f"{build_id}_labels.csv"
                                labels_df[['Node_ID', 'Node_Name', 'Label']].to_csv(os.path.join(project_output_dir, output_filename), index=False)
                                
                                failed_count = labels_df['Label'].sum()
                                print(f"{build_id}: Total {len(labels_df)}, Failures {failed_count}")
                    except Exception as e:
                        print(f"Error {zip_path}: {e}")

if __name__ == "__main__":
    base_data_dir = "processed_test_result"
    extract_labels(base_data_dir)