import requests

# 1. Get the Data
url = "https://tds-llm-analysis.s-anand.net/project2/gh-tree.json"
data = requests.get(url).json()

# 2. Email Offset Calculation
email = "23f2000977@ds.study.iitm.ac.in"
email_offset = len(email) % 2
print(f"Email Length: {len(email)}")
print(f"Offset (Length % 2): {email_offset}")

# 3. Recursive Finder
def count_md_files(node):
    count = 0
    name = node.get('name', '')
    
    # If it is a file and ends with .md
    if node.get('type') == 'blob' and name.endswith('.md'):
        count = 1
        # print(f"Found: {name}") # Uncomment to see filenames
    
    # If it is a folder, recurse
    if 'children' in node:
        for child in node['children']:
            count += count_md_files(child)
            
    return count

# 4. Search top-level folders
print("\n--- ANALYZING FOLDERS ---")

total_md = count_md_files(data)
print(f"TOTAL .md files in entire repo: {total_md}")
print(f"-> Answer if prefix is '/': {total_md + email_offset}")

if 'children' in data:
    for child in data['children']:
        if child.get('type') == 'tree':
            folder_name = child['name']
            count = count_md_files(child)
            print(f"\nPrefix '{folder_name}': Found {count} .md files")
            print(f"-> Answer: {count + email_offset}")