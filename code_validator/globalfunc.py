import requests
import json
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.schema import Document
import re 
from langchain_openai import ChatOpenAI
import os 
from os import environ as env
from dotenv import load_dotenv
from pprint import pprint
import os
import nbformat
from difflib import SequenceMatcher
import base64
from github import Github
import nbformat
from git import Repo


load_dotenv() 

OPENAI_API_KEY = env['OPENAI_API_KEY']


llm = ChatOpenAI(api_key=OPENAI_API_KEY, model="gpt-4o-mini", temperature=0.7)

git_token = env['GIT_TOKEN']

def get_repo_info(repo_url):
    match = re.match(r"https://github.com/([^/]+)/([^/]+)(?:\.git)?", repo_url)
    if not match:
        print(status_code=400, detail="❌ Invalid GitHub repository URL format.")

    owner, repo_name_candidate = match.groups()
    repo_name = repo_name_candidate[:-4] if repo_name_candidate.endswith('.git') else repo_name_candidate

    api_url = f"https://api.github.com/repos/{owner}/{repo_name}/events"
    headers = {}
    if git_token:
        headers["Authorization"] = f"token {git_token}"

    response = requests.get(api_url, headers=headers)
    if response.status_code != 200:
        print(status_code=response.status_code, detail=f"GitHub API error: {response.text}")
    events = response.json()
    return owner, repo_name, events

def get_commit_list(repo_info, push_id1, push_id2):
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
    with open("push_events.json", "w") as f:
        json.dump([push_id1,push_id2, commits_list[0], commits_list[-1], owner, repo_name], f, indent=2) ###[0] is target, [-1] is base
    return commits_list

def get_commit_diff():
    with open("push_events.json", "r") as file:
        data = json.load(file)
    owner = data[4]
    repo = data[5]
    base_commit = data[3]
    target_commit = data[2]
    url = f"https://api.github.com/repos/{owner}/{repo}/compare/{base_commit}...{target_commit}"
    headers = {}
    if git_token:
        headers["Authorization"] = f"token {git_token}"  # for private repos
    
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    data = response.json()
    
    # GitHub returns patches per file in 'files'
    diffs = [f["patch"] for f in data.get("files", []) if "patch" in f]
    return "\n\n".join(diffs)


def get_clean_code(diff_text):
    code_changes = []

    for line in diff_text.splitlines():
        if line.startswith('---') or line.startswith('+++') or line.startswith('@@'):
            continue
        if "execution_count" in line:
            continue

        if line.startswith('-') or line.startswith('+'):
            cleaned_line = line[1:].strip()
            cleaned_line = cleaned_line.strip('",')
            cleaned_line = cleaned_line.strip('"')
            cleaned_line = cleaned_line.replace('\\n', '')

            cleaned_line = re.sub('<[^>]+>', '', cleaned_line)

            code_changes.append(line[0] + ' ' + cleaned_line)

    return code_changes

def get_commit_diffs():
    with open("push_events.json", "r") as file:
        data = json.load(file)
    owner = data[4]
    repo = data[5]
    push_id1 = data[0]
    push_id2 = data[1]
    headers = {}
    if git_token:
        headers["Authorization"] = f"token {git_token}" 
        api_url = f"https://api.github.com/repos/{owner}/{repo}/events"
    

    response = requests.get(api_url, headers=headers)
    if response.status_code != 200:
        print(status_code=response.status_code, detail=f"GitHub API error: {response.text}")
    events = response.json()
    push_events = [e for e in events if e['type'] == 'PushEvent']
    ids = [e['id'] for e in push_events]
    try:
        idx1 = ids.index(push_id1)
        idx2 = ids.index(push_id2)
    except ValueError:
        return {"error": "One or both push IDs not found in recent events."}

    start = min(idx1, idx2)
    end = max(idx1, idx2)
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
    
    commits_list = commits_list[::-1]
    results = []
    for commit_sha in commits_list:
    # Fetch diff for this commit
        commit_url = f"https://api.github.com/repos/{owner}/{repo}/commits/{commit_sha}"
        commit_resp = requests.get(commit_url, headers=headers)
        if commit_resp.status_code != 200:
            continue

        commit_data = commit_resp.json()
        commit_message = commit_data["commit"]["message"]

        code_changes = []
        for file in commit_data.get("files", []):
            patch = file.get("patch")
            if patch:
                code_changes.append({
                    "filename": file["filename"],
                    "diff": patch.splitlines()
                })

        results.append({
            "hash": commit_sha,
            "message": commit_message,
            "diff": code_changes
        })
    
    return results

# def json_data():
#     with open("push_events.json", "r") as file:
#         data = json.load(file)

#     owner = data[2]
#     repo_name = data[3]
#     api_url = f"https://api.github.com/repos/{owner}/{repo_name}/events"
#     headers = {}
#     if git_token:
#         headers["Authorization"] = f"token {git_token}"

#     response = requests.get(api_url, headers=headers)
#     events = response.json()

#     push_events = [e for e in events if e['type'] == 'PushEvent']
#     ids = [e['id'] for e in push_events]

#     try:
#         idx1 = ids.index(data[0])
#         idx2 = ids.index(data[1])
#     except ValueError:
#         return {"error": "One or both push IDs not found in recent events."}

#     start = min(idx1, idx2)
#     end = max(idx1, idx2)
#     history_between = push_events[start:end+1]  

#     commits_list = []
#     for i, event in enumerate(history_between):
#         event_commits = event["payload"]["commits"]
#         if i == 0 and event_commits:
#             commits_list.append({
#                 "sha": event_commits[-1]["sha"],
#                 "message": event_commits[-1]["message"]
#             })
#         else:
#             for commit in event_commits:
#                 commits_list.append({
#                     "sha": commit["sha"],
#                     "message": commit["message"]
#                 })

#     diffs = []
#     for i in range(1, len(commits_list)):
#         c1 = commits_list[i-1]["sha"]
#         c2 = commits_list[i]["sha"]

#         url = f"https://api.github.com/repos/{owner}/{repo_name}/compare/{c2}...{c1}"
#         resp = requests.get(url, headers=headers)
#         if resp.status_code != 200:
#             return {"error": f"Failed to fetch diff {c2}..{c1}: {resp.text}"}

#         diff_data = resp.json()
#         processed_files = []

#         for file in diff_data.get("files", []):
#             if "patch" in file and file["filename"] == "loan-approval-prediction.ipynb":
#                 code_changes = []
#                 for line in file["patch"].splitlines():
#                     # Ignore diff header lines
#                     if line.startswith(('---', '+++', '@@')):
#                         continue
#                     # Ignore execution_count
#                     if "execution_count" in line:
#                         continue
#                     # Only process added/removed lines
#                     if line.startswith(('+', '-')):
#                         cleaned_line = line[1:].strip()
#                         cleaned_line = cleaned_line.strip('",')
#                         cleaned_line = cleaned_line.strip('"')
#                         cleaned_line = cleaned_line.replace('\\n', '')
#                         cleaned_line = re.sub('<[^>]+>', '', cleaned_line)

#                         if cleaned_line:  # only keep meaningful changes
#                             code_changes.append(line[0] + ' ' + cleaned_line)

#                 if code_changes:  # add only if non-empty patch
#                     processed_files.append({
#                         "filename": file["filename"],
#                         "patch": code_changes
#                     })

#         # ✅ Append diff only if files_changed is not empty
#         if processed_files:
#             diffs.append({
#                 "from": c2,
#                 "to": c1,
#                 "from_message": commits_list[i-1]["message"],
#                 "to_message": commits_list[i]["message"],
#                 "files_changed": processed_files
#             })

#     return {"diffs": diffs}


def create_summary(json_data):
    # if isinstance(json_data, dict):
    #     json_data = [json_data]

    # req_data = json.dumps(json_data, indent=2)
    summary_prompt_template = """
    You are an expert software engineer and technical writer.  
    I will provide you with one or more commit diffs that include:
    - "hash"
    - "message"
    - a list of extracted code changes

    Your task:  
    1. Generate an **Overall Summary** that explains the main themes and impact of all commits combined.  
    - Highlight how the project evolved across these changes.  
    - Summarize the broader direction (e.g., schema improvements, feature engineering, bug fixes, performance enhancements).  

    2. For each commit, generate a **Commit Diff Summary** with the following structure:  
    - **commit hashe** hash
    - **commit messages** message
    - **Textual Summary**: A detailed natural-language explanation of what was added, modified, or removed, and why it matters.  
    - **Code Difference Summary**: A clear textual description of the code-level changes (e.g., column renames, new imports, added functions, removed logic, or feature engineering additions).  

    3. Be **descriptive and deep**. Do not write one-liners.  
    - Assume the reader is a developer who wants to understand the commit without reading raw diffs.  
    - Mention both functional impact (how the code will behave differently) and structural impact (readability, maintainability, consistency).  

    4. After all commit summaries, provide a **✅ Final Takeaway** section that synthesizes the importance of these commits together.  
   - Highlight the overall project trajectory, maturity, and likely goals.  
   - Provide insight into how these changes collectively improve functionality, maintainability, or predictive performance.  

    5. The output must always follow this structure:  
    🔹 Overall Summary  
    🔹 Commit Diff 1  
    🔹 Commit Diff 2  
    … (and so on for each commit)  
    ✅ Final Takeaway  

    Context:  
    {context}  

    Now provide the **Overall Summary** followed by per-commit summaries as described above.  
    """

    summary_prompt = PromptTemplate(
        template=summary_prompt_template,
        input_variables=["context"]
    )

    formatted_prompt = summary_prompt.format(context=json_data)

    response = llm.invoke(formatted_prompt)

    with open("event_summary.txt", "w", encoding="utf-8") as f:
        f.write(response.content)

    return response.content


def get_push_commits():
    with open("push_events.json", "r", encoding="utf-8") as file:
        data = json.load(file)
    owner = data[4]
    repo_name = data[5]
    push_id = data[1]
    
    headers = {"Authorization": f"token {git_token}"} if git_token else {}
    url = f"https://api.github.com/repos/{owner}/{repo_name}/events"
    resp = requests.get(url, headers=headers)
    resp.raise_for_status()

    events = resp.json()
    push_event = [e for e in events if e['type'] == 'PushEvent' and e['id'] == push_id]
    if not push_event:
        raise ValueError(f"Push event {push_id} not found")
    
    return push_event[0]["payload"]["commits"]


def get_latest_commit_sha(commits, file_path="src/model_v1.py"):
    with open("push_events.json", "r", encoding="utf-8") as file:
        data = json.load(file)
    owner = data[4]
    repo_name = data[5]
    push_id = data[1]

    headers = {"Authorization": f"token {git_token}"} if git_token else {}
    latest_sha = None

    for commit in reversed(commits):
        sha = commit["sha"]
        url = f"https://api.github.com/repos/{owner}/{repo_name}/commits/{sha}"
        resp = requests.get(url, headers=headers)
        resp.raise_for_status()
        files = resp.json().get("files", [])
        for f in files:
            if f["filename"] == file_path:
                latest_sha = sha
                break
        if latest_sha:
            break

    # Save last state locally
    with open("last_state.json", "w", encoding="utf-8") as f:
        json.dump({"push_id": push_id, "latest_sha": latest_sha}, f, indent=2)

    return latest_sha


def get_file_content(file_path, sha):
    with open("push_events.json", "r", encoding="utf-8") as file:
        data = json.load(file)
    owner = data[4]
    repo_name = data[5]

    headers = {"Authorization": f"token {git_token}"} if git_token else {}
    url = f"https://api.github.com/repos/{owner}/{repo_name}/contents/{file_path}?ref={sha}"
    resp = requests.get(url, headers=headers)

    if resp.status_code == 404:
        return None

    resp.raise_for_status()
    content = resp.json().get("content", "")
    return base64.b64decode(content).decode("utf-8") if content else None


# ----------------- Test Case Generation ----------------- #

def chunk_text(text, max_chunk_size=20000):
    """Split text into smaller chunks (so we never exceed token limit)."""
    lines = text.splitlines()
    chunks, current_chunk, length = [], [], 0
    for line in lines:
        length += len(line.split())
        if length > max_chunk_size:
            chunks.append("\n".join(current_chunk))
            current_chunk, length = [], 0
        current_chunk.append(line)
    if current_chunk:
        chunks.append("\n".join(current_chunk))
    return chunks


def generate_test_cases(py_raw: str) -> str:
    chunks = chunk_text(py_raw)
    all_cases = []

    for i, chunk in enumerate(chunks, start=1):
        prompt = f"""
        You are an assistant that writes software test cases.

        The following is part {i} of a Python script (only code):
        ---
        {chunk}
        ---
        Based on this code, generate meaningful test cases that validate preprocessing, model training, predictions, and evaluation. 
        Each test case should include: 
        - A title (e.g., Test Case 1: Data Import Validation). 
        - A description of what it validates. 
        - A 'How to Perform' section with Python code snippets that show how the test can be executed. 

        Output them as a clear, bullet-point list with proper formatting.
        """
        response = llm.invoke(prompt)
        all_cases.append(response.content)
    return "\n".join(all_cases)


def update_file(path: str, content: str, sha: str, message="Update test cases"):
    """Update file on GitHub; fallback to local write if repo not found."""
    with open("push_events.json", "r", encoding="utf-8") as file:
        data = json.load(file)
    owner = data[4]
    repo_name = data[5]

    try:
        g = Github(git_token)
        repo_obj = g.get_repo(f"{owner}/{repo_name}")
        file_obj = repo_obj.get_contents(path, ref="main")
        repo_obj.update_file(path, message, content, file_obj.sha, branch="main")
    except Exception as e:
        print(f"⚠️ GitHub update failed: {e}. Saving locally instead.")
        
