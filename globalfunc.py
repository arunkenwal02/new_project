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
load_dotenv() 

OPENAI_API_KEY = env['OPENAI_API_KEY']


# embeddings = GoogleGenerativeAIEmbeddings(google_api_key=GOOGLE_API_KEY, model="models/embedding-001")

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


def json_data():
    with open("push_events.json", "r") as file:
        data = json.load(file)
    owner = data[2]
    repo_name = data[3]
    api_url = f"https://api.github.com/repos/{owner}/{repo_name}/events"
    headers = {}
    if git_token:
        headers["Authorization"] = f"token {git_token}"

    response = requests.get(api_url, headers=headers)
    # response = requests.get(api_url)
    events = response.json()

    push_events = [e for e in events if e['type'] == 'PushEvent']

    ids = [e['id'] for e in push_events]
    try:
        idx1 = ids.index(data[0])
        idx2 = ids.index(data[1])
    except ValueError:
        return {"error": "One or both push IDs not found in recent events."}

    start = min(idx1, idx2)
    end = max(idx1, idx2)

    history_between = push_events[start:end+1]  

    commits_list = []
    for i, event in enumerate(history_between):
        event_commits = event["payload"]["commits"]
        if i == 0 and event_commits:
            commits_list.append({
                "sha": event_commits[-1]["sha"],
                "message": event_commits[-1]["message"]
            })
        else:
            for commit in event_commits:
                commits_list.append({
                    "sha": commit["sha"],
                    "message": commit["message"]
                })

    diffs = []
    for i in range(1, len(commits_list)):
        c1 = commits_list[i-1]["sha"]
        c2 = commits_list[i]["sha"]

        url = f"https://api.github.com/repos/{owner}/{repo_name}/compare/{c2}...{c1}"
        resp = requests.get(url, headers=headers)

        if resp.status_code != 200:
            return {"error": f"Failed to fetch diff {c1}..{c2}: {resp.text}"}

        diff_data = resp.json()
        diffs.append({
            "from": c2,
            "to": c1,
            "from_message": commits_list[i-1]["message"],
            "to_message": commits_list[i]["message"],
            "files_changed": diff_data.get("files", [])
        })

    return {
        "diffs": diffs
    }

def create_summary(json_data):
    if isinstance(json_data, dict):
        json_data = [json_data]

    req_data = json.dumps(json_data, indent=2)

    summary_prompt_template = """
            You are an expert at analyzing GitHub commit diffs and push events.

            Start your output with an **Overall Summary** as a single, detailed narrative paragraph of at least 700 to 1000 words:
            - Include the repository name (if available)
            - Mention the total number of commits being analyzed
            - Integrate all commit messages ('from_message' and 'to_message') into a natural, flowing story
            - Highlight major changes across all commits, including new features, bug fixes, refactoring, and dependency updates
            - Explain the impact of these changes on the repository’s functionality, maintainability, and usability
            - Use descriptive language to provide context about why each change matters and how the commits relate to each other
            - Avoid bullet points or lists—write it as a single cohesive paragraph


            After the overall summary, for **each commit comparison** in the input JSON:
            1. Write a "Textual Summary" section:
                - Include the short SHA (first 7 characters) of 'from' and 'to'
                - Include the commit messages ('from_message' and 'to_message')
                - Number of files changed
                - Brief description of the purpose of the change in plain English
                - Mention if the change updates dependencies or adds major features

            2. Write a "Code Difference Summary" section:
                - Describe file-level changes (created, modified, deleted)
                - Explain the intent of significant changes, like bug fixes, new features, or dependency upgrades
                - Do NOT include raw code; only describe the effect of changes

            Skip any comparison where 'files_changed' is an empty list.

            Use exactly the headings "Overall Summary:", "Textual Summary:", and "Code Difference Summary:" for the output. Repeat this format for all valid comparisons.

            Here is the input JSON:
            Context: 
            {context}
            """

    summary_prompt = PromptTemplate(
        template=summary_prompt_template,
        input_variables=["context"]
    )

    formatted_prompt = summary_prompt.format(context=req_data)

    response = llm.invoke(formatted_prompt)
    pprint(response.content)
    with open("event_summary.txt", "w", encoding="utf-8") as f:
        f.write(response.content)
    return response.content

def test_cases_generation():
    prompt = """
        You are an expert in Python, PyTest, and Data Science workflows. 
        Generate **comprehensive, well-structured PyTest test cases** (including edge cases and negative scenarios) 
        for validating a full machine learning pipeline that includes the following steps:

        1. **Data Processing** - Validate data cleaning, handling missing values, and type conversions.  
        2. **EDA (Exploratory Data Analysis)** - Ensure data summaries, correlations, and statistical outputs are correct.  
        3. **Data Distribution** - Test histogram generation, skewness/kurtosis calculations, and distribution fitting.  
        4. **Encoding** - Test Label Encoding, One-Hot Encoding, and `pd.get_dummies()` to ensure correctness for both categorical and mixed data types.  
        5. **Train-Test Split** - Validate splitting ratio correctness, stratification for imbalanced datasets, and randomness seed reproducibility.  
        6. **Model Training** - Verify training pipelines for Logistic Regression, Decision Tree, and Random Forest, ensuring model parameters and fitting steps are correct.  
        7. **Metrics Calculation** - Test accuracy, precision, recall, and f1-score calculations for correctness using both balanced and imbalanced datasets.

        Requirements for the output:
        - Use **pytest** framework.
        - Include **mock data** or fixtures where needed.
        - Add **edge cases**, such as empty datasets, all-null columns, and unseen categories during encoding.
        - Cover **invalid input handling**.
        - Ensure **assertions** verify correctness of outputs and expected exceptions.
        - Organize tests into classes or modules per pipeline step.
        - Include **descriptive comments** explaining each test case.
    """
    summary_prompt = PromptTemplate(
        template=prompt,
        input_variables=[]
    )

    formatted_prompt = summary_prompt.format()

    response = llm.invoke(formatted_prompt)
    pprint(response.content)
    with open('test_cases.txt', 'w') as f:
        f.write(response.content)
    return response.content



def is_test_cases_relevant(notebook_path, test_cases_path, threshold=0.3):
    """Check if test_cases.txt is relevant to the notebook content."""
    if not os.path.exists(test_cases_path):
        return False

    nb = nbformat.read(notebook_path, as_version=4)
    nb_content = " ".join([cell['source'] for cell in nb.cells if cell['cell_type'] == 'code' or cell['cell_type'] == 'markdown'])

    with open(test_cases_path, 'r') as f:
        test_cases_content = f.read()

    similarity = SequenceMatcher(None, nb_content.lower(), test_cases_content.lower()).ratio()
    print(f"Similarity Score: {similarity:.2f}")

    return similarity >= threshold

def file_url():
    with open('test_cases.txt', "w") as f:
            f.write("")
    with open("push_events.json", "r") as file:
        data = json.load(file)
    owner = data[2]
    repo_name = data[3]
    notebook_raw_url = f"https://raw.githubusercontent.com/{owner}/{repo_name}/main/loan-approval-prediction_v1.ipynb"
    local_notebook_path = "loan-approval-prediction_v1.ipynb"

    r = requests.get(notebook_raw_url)
    r.raise_for_status()
    with open(local_notebook_path, "wb") as f:
        f.write(r.content)

    testcases_path = "test_cases.txt"
    return local_notebook_path, testcases_path
    


