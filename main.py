import json
import requests
from fastapi import FastAPI, Query, HTTPException, UploadFile, File
from langchain_core.documents import Document
from globalfunc import *
from globalfunc import llm, git_token


app = FastAPI()

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
        patches = []
        for file in diff_data.get("files", []):
            if "patch" in file:  
                patches.append({
                    "filename": file["filename"],
                    "patch": file["patch"]
                })
        if patches:
            diffs.append({
                "target": c2,
                "base": c1,
                "patches": patches
            })
    with open("push_events.json", "w") as f:
        json.dump([push_id1,push_id2, owner, repo_name], f, indent=2)

    
    return {
        "intermediate_push_count": len(push_ids),
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
    path = file_url()
    check = is_test_cases_relevant(path[0], path[1])
    if not check:
        cases = test_cases_generation()
        return "Test cases are not relevant — generating new ones...", cases
    else:
        return "Test cases are already relevant to the notebook."