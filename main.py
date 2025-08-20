import json
import requests
from fastapi import FastAPI, Query, HTTPException, UploadFile, File
from langchain_core.documents import Document
from globalfunc import *
from globalfunc import llm, git_token
import tiktoken
import os
import re


app = FastAPI()

@app.post("/getpushid")
def get_push_id(repo_url: str = Query(...)):
    GITHUB_API = "https://api.github.com"
    headers = {
        "Authorization": f"token {git_token}"
    }
    owner, repo, events = get_repo_info(repo_url)

    push_events = [e for e in events if e['type'] == 'PushEvent']
    ids = []

    for event in push_events:
        for commit in event.get("payload", {}).get("commits", []):
            sha = commit.get("sha")
            if sha:
                # Fetch commit details to get changed files
                commit_url = f"{GITHUB_API}/repos/{owner}/{repo}/commits/{sha}"
                commit_resp = requests.get(commit_url, headers=headers)
                if commit_resp.status_code == 200:
                    commit_data = commit_resp.json()
                    files = commit_data.get("files", [])
                    for f in files:
                        if f.get("filename") == "loan-approval-prediction.ipynb":
                            ids.append(event["id"])
                            break  # no need to check more files for this commit
    
    filtered_ids = list(set(ids))
    return filtered_ids

@app.post("/fetchpushhistory")
def get_push_history(repo_url: str = Query(...),push_id1: str = Query(...),push_id2: str = Query(...)):    # Parse GitHub repo URL
    repo_info = get_repo_info(repo_url)
    owner = repo_info[0]
    repo_name = repo_info[1]
    events = repo_info[2]
    push_events = [e for e in events if e['type'] == 'PushEvent']
    ids = [e['id'] for e in push_events]
    try:
        idx1 = ids.index(push_id1)
        idx2 = ids.index(push_id2)
    except ValueError:
        return {"error": "One or both push IDs not found in recent events."}

    start = min(idx1, idx2)
    end = max(idx1, idx2)
    headers = {
        "Authorization": f"token {git_token}"
    }
    history_between = push_events[start:end+1]
    commits_list = []
    push_ids = []
    for i, event in enumerate(history_between):
        push_ids.append(event['id'])
        event_commits = [c["sha"] for c in event["payload"]["commits"]]
        if i == 0:
            if event_commits:
                commits_list.append(event_commits[-1])
        else:
            commits_list.extend(event_commits)
    diffs = []
    for i in range(1, len(commits_list)):
        c1 = commits_list[i-1]
        c2 = commits_list[i]

        url = f"https://api.github.com/repos/{owner}/{repo_name}/compare/{c2}...{c1}"
        resp = requests.get(url, headers=headers)

        if resp.status_code != 200:
            return {"error": f"Failed to fetch diff {c2}..{c1}: {resp.text}"}

        diff_data = resp.json()
        
        processed_files = []
        for file in diff_data.get("files", []):
            if "patch" in file and file["filename"] == "loan-approval-prediction.ipynb":
                code_changes = []
                for line in file["patch"].splitlines():
                    # --- NEW FILTERING LOGIC ---
                    # Ignore diff header lines
                    if line.startswith('---') or line.startswith('+++') or line.startswith('@@'):
                        continue
                        
                    # Ignore any lines containing "execution_count" regardless of prefix
                    if "execution_count" in line:
                        continue
                    # --- END NEW FILTERING LOGIC ---

                    # Only process lines that are additions or removals
                    if line.startswith('-') or line.startswith('+'):
                        # Strip leading '+', '-', and whitespace
                        cleaned_line = line[1:].strip()
                        # Remove JSON string formatting and escaped characters
                        cleaned_line = cleaned_line.strip('",')
                        cleaned_line = cleaned_line.strip('"')
                        cleaned_line = cleaned_line.replace('\\n', '')
                        # Remove HTML tags using a regular expression
                        cleaned_line = re.sub('<[^>]+>', '', cleaned_line)
                        
                        # Re-add the diff indicator (+ or -) to the cleaned line
                        code_changes.append(line[0] + ' ' + cleaned_line)
                            
                processed_files.append({
                    "filename": file["filename"],
                    "extracted_code_changes": code_changes
                })

        if processed_files:
            diffs.append({
                "target": c2,
                "base": c1,
                "files": processed_files
            })

    
    with open("push_events.json", "w") as f:
        json.dump([push_id1,push_id2, owner, repo_name, repo_url], f, indent=2)


    
    return {
        "total_push_count": len(push_ids),
        "total_commits": len(commits_list),
        "commits": commits_list,
        "diffs": diffs
    }


@app.get('/createsummary')
def create_summary_of_events():
    json_res = json_data()
    summary = create_summary(json_res)
    return summary


    
@app.get('/test_cases')
def generate_tescases():
    commits = get_push_commits()
    sha = get_latest_commit_sha(commits)

    with open("push_events.json", "r") as file:
        data = json.load(file)
    push_id = data[1]

    state_file = "last_state.json"
    last_push_id, last_sha = None, None
    if os.path.exists(state_file):
        with open(state_file, "r") as f:
            state = json.load(f)
            last_push_id = state.get("push_id")
            last_sha = state.get("sha")

    # if push_id == last_push_id and sha == last_sha:
    #     return f'Test cases for push id {push_id} are already created.'

    notebook_content = get_file_content("loan-approval-prediction.ipynb", sha)

    try:
        nb = nbformat.reads(notebook_content, as_version=4)
        code_cells = [cell['source'] for cell in nb.cells if cell['cell_type'] == 'code']
        notebook_content = "\n".join(code_cells)
    except Exception as e:
        print(f"⚠️ Could not parse notebook as nbformat, falling back to raw content. Error: {e}")

    test_cases = get_file_content("test_cases.txt", sha) or ""

    if not is_test_cases_relevant(notebook_content, test_cases):
        new_test_cases = generate_test_cases(notebook_content)
        update_file("test_cases.txt", new_test_cases, sha)
        print("✅ test_cases.txt updated with new test cases")
        result = new_test_cases
    else:
        result = "ℹ️ Test cases are already relevant. No update needed."

    with open(state_file, "w") as f:
        json.dump({"push_id": push_id, "sha": sha}, f)

    return result
