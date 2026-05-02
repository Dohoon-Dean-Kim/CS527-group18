"""
Executes Maven test pipelines to inject JaCoCo agents and generate coverage reports.
"""

import os
import subprocess

def run_jacoco_perfect_final(history_dir):
    print("Starting Final Robust Pipeline for 7 Projects")

    # Try to avoid network timeout
    custom_env = os.environ.copy()
    custom_env["MAVEN_OPTS"] = "-Dsun.net.client.defaultConnectTimeout=5000 -Dsun.net.client.defaultReadTimeout=5000"

    target_projects = [
        "log4j", 
        "jackrabbit-oak", 
        "hadoop", 
        "hbase", 
        "karaf", 
        "james", 
        "kafka"
    ]
    
    for project in target_projects:
        trunk_path = os.path.join(history_dir, project, f"{project}_trunk")
        
        if os.path.isdir(trunk_path):
            print("Start")
            # Clean Install & Resolve Dependencies
            install_command = [
                "mvn", "-B", "-q", "clean", "install", "-DskipTests",
                "-Dmaven.javadoc.skip=true", "-Dmaven.site.skip=true",
                "-Dcheckstyle.skip=true", "-Drat.skip=true", "-Dpmd.skip=true",
                "-Denforcer.skip=true", "-Dassembly.skipAssembly=true", "-fn",
                "-Dmaven.site.skip=true", "-Dmaven.javadoc.skip=true"
            ]
            
            try:
                subprocess.run(install_command, cwd=trunk_path, env=custom_env, check=False)
            except Exception as e:
                print(f"[{project}] Install failed: {e}")

            print(f"Running tests with JaCoCo agent")
            test_command = [
                "mvn", "-B", "-q",
                "org.jacoco:jacoco-maven-plugin:0.8.10:prepare-agent", "test", 
                "org.jacoco:jacoco-maven-plugin:0.8.10:report", "-fn", 
                "-Dmaven.test.failure.ignore=true",
                "-Dsurefire.forkedProcessTimeoutInSeconds=120",
                "-Dmaven.javadoc.skip=true", "-Dmaven.site.skip=true",
                "-Dcheckstyle.skip=true", "-Drat.skip=true",
                "-Dmaven.site.skip=true", "-Dmaven.javadoc.skip=true"
            ]
            
            try:
                process = subprocess.Popen(
                    test_command, cwd=trunk_path, env=custom_env, shell=False
                )
                process.wait() 
                print(f"[{project}] Pipeline done")
            except Exception as e:
                print(f"[{project}] Error: {e}")

if __name__ == "__main__":
    run_jacoco_perfect_final("history")