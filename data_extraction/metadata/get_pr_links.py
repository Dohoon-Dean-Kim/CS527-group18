"""
Scrapes and stores Pull Request build links and metadata from the CI server.
"""

import requests
import json
import os
from datetime import datetime, date
import glob

import const

# Headers to mimic the browser
headers = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 \
    (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'
}


def get_pr_links(project):
    pr_dir = os.path.join(const.metadir, project, const.PRLINKS)
    os.makedirs(pr_dir, exist_ok=True)
    
    # store links into json
    res = requests.get(url=const.PROJECT_URLS[project], headers=headers)
    today = datetime.today().strftime('%Y-%m-%d')
    out_file = f"{pr_dir}/{today}.json"
    
    if not os.path.exists(out_file):
        if res.status_code != 200:
            print(f"[{project}] Connection failed. Status: {res.status_code}")
            return
        
        # JSON parsing error
        try:
            data = json.loads(res.text)
            with open(out_file, "w") as f:
                json.dump(data, f, indent=2)
            print(f"Saved: {out_file}")
            
        except json.decoder.JSONDecodeError:
            print(f"[{project}] JSON decode error")
            print(f"Response: {res.text[:100]}")


def get_all_prs(project):
    # get all pr files by date
    pr_dir = os.path.join(const.metadir, project, const.PRLINKS)
    files = glob.glob(os.path.join(pr_dir, "*.json"))
    print("#files", project, len(files))
    
    # collect all pr links from all files
    links = {}
    for f in files:
        with open(f, "r") as file_obj:
            jobs = json.load(file_obj).get("jobs", [])
            for job in jobs:
                if job["name"] not in links:
                    links[job["name"]] = job["url"]
                    
    print(f"[{project}] total pr links: {len(links)}")
    return links

if __name__ == "__main__":
    for project in const.PROJECTS:
        get_pr_links(project)
    pass