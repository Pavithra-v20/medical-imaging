import ast
import os
import sys

def get_imports(path):
    imports = set()
    for root, _, files in os.walk(path):
        for file in files:
            if file.endswith('.py'):
                try:
                    with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                        tree = ast.parse(f.read())
                    for node in ast.walk(tree):
                        if isinstance(node, ast.Import):
                            for n in node.names:
                                imports.add(n.name.split('.')[0])
                        elif isinstance(node, ast.ImportFrom):
                            if node.module:
                                imports.add(node.module.split('.')[0])
                except Exception as e:
                    pass
    return imports

if __name__ == "__main__":
    path = sys.argv[1]
    with open('imports.txt', 'w') as f:
        for imp in sorted(get_imports(path)):
            f.write(imp + '\n')
