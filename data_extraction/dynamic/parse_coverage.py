# """
# Parses JaCoCo XML coverage reports and exports instruction-level coverage to a CSV.
# """

# import os
# import csv
# import xml.etree.ElementTree as ET

# def extract_coverage_to_csv(history_dir, output_file):
#     print("Start")
    
#     xml_files = []
#     for root, dirs, files in os.walk(history_dir):
#         if "jacoco" in file.lower() and file.endswith(".xml"):
#             xml_files.append(os.path.join(root, file))
            
#     print(f"Found {len(xml_files)} xml files")

#     with open(output_file, mode='w', newline='', encoding='utf-8') as csv_file:
#         writer = csv.writer(csv_file)
#         writer.writerow(['Project_Module', 'Package', 'Class', 'Method', 'Instruction_Covered', 'Instruction_Missed', 'Is_Executed'])

#         for xml_path in xml_files:
#             path_parts = xml_path.split(os.sep)
            
#             try:
#                 history_idx = path_parts.index("history")
#                 project_name = path_parts[history_idx + 1] # ex. hadoop
#                 module_name = path_parts[history_idx + 3]  # ex. hadoop-common
#                 full_project_module = f"{project_name}/{module_name}"
#             except ValueError:
#                 full_project_module = "Unknown"

#             try:
#                 tree = ET.parse(xml_path)
#                 root = tree.getroot()

#                 for package in root.findall('package'):
#                     pkg_name = package.get('name', 'default_pkg').replace('/', '.')
                    
#                     for cls in package.findall('class'):
#                         cls_name = cls.get('name', 'unknown_class').replace('/', '.')
                        
#                         for m in cls.findall('method'):
#                             method_name = m.get('name', 'unknown_method')
#                             method_desc = m.get('desc', '')
#                             full_method = f"{method_name}{method_desc}"
                            
#                             covered = 0
#                             missed = 0
#                             for counter in m.findall('counter'):
#                                 if counter.get('type') == 'INSTRUCTION':
#                                     covered = int(counter.get('covered', 0))
#                                     missed = int(counter.get('missed', 0))
#                                     break
                            
#                             is_executed = 1 if covered > 0 else 0
                            
#                             writer.writerow([full_project_module, pkg_name, cls_name, full_method, covered, missed, is_executed])
                            
#             except Exception as e:
#                 print(f"Saved coverage data ({output_csv})")

#     print(f"Parsing Done. Data saved ({output_file})")

# if __name__ == "__main__":
#     extract_coverage_to_csv("history", "method_coverage_nodes.csv")

"""
Parses JaCoCo XML coverage reports and exports instruction-level coverage to a CSV.
"""
import os
import csv
import xml.etree.ElementTree as ET

def extract_coverage_to_csv(history_dir, output_file):
    print("Start")
    
    xml_files = []
    for root, dirs, files in os.walk(history_dir):
        for file in files:
            if "jacoco" in file.lower() and file.endswith(".xml"):
                xml_files.append(os.path.join(root, file))
            
    print(f"Found {len(xml_files)} xml files")

    with open(output_file, mode='w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['Project_Module', 'Package', 'Class', 'Method', 'Instruction_Covered', 'Instruction_Missed', 'Is_Executed'])

        for xml_path in xml_files:
            path_parts = xml_path.split(os.sep)
            
            try:
                history_idx = path_parts.index("history")
                project_name = path_parts[history_idx + 1] # ex. hadoop
                module_name = path_parts[history_idx + 3]  # ex. hadoop-common
                full_project_module = f"{project_name}/{module_name}"
            except ValueError:
                full_project_module = "Unknown"

            try:
                tree = ET.parse(xml_path)
                root = tree.getroot()

                for package in root.findall('package'):
                    pkg_name = package.get('name', 'default_pkg').replace('/', '.')
                    
                    for cls in package.findall('class'):
                        cls_name = cls.get('name', 'unknown_class').replace('/', '.')
                        
                        for m in cls.findall('method'):
                            method_name = m.get('name', 'unknown_method')
                            method_desc = m.get('desc', '')
                            full_method = f"{method_name}{method_desc}"
                            
                            covered = 0
                            missed = 0
                            for counter in m.findall('counter'):
                                if counter.get('type') == 'INSTRUCTION':
                                    covered = int(counter.get('covered', 0))
                                    missed = int(counter.get('missed', 0))
                                    break
                            
                            is_executed = 1 if covered > 0 else 0
                            
                            writer.writerow([full_project_module, pkg_name, cls_name, full_method, covered, missed, is_executed])
                            
            except Exception as e:
                print(f"Parse error in {xml_path}: {e}")

    print(f"Parsing Done. Data saved ({output_file})")

if __name__ == "__main__":
    extract_coverage_to_csv("history", "method_coverage_nodes.csv")