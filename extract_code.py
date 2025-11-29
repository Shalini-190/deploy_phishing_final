import json

notebook_path = r"C:\Users\Admin\.gemini\antigravity\scratch\phishing_project\Untitled42 (1).ipynb"
output_path = r"C:\Users\Admin\.gemini\antigravity\scratch\phishing_project\extracted_code.py"

try:
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    code_cells = []
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            code_cells.append(source)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n\n# %% [code cell]\n'.join(code_cells))

    print(f"Successfully extracted {len(code_cells)} code cells to {output_path}")

except Exception as e:
    print(f"Error: {e}")
