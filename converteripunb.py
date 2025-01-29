import nbformat as nbf
from nbconvert import PythonExporter

# Carregar o notebook
notebook_filename = "desafio.ipynb"
with open(notebook_filename, "r", encoding="utf-8") as f:
    nb = nbf.read(f, as_version=4)

# Exportar o notebook para um script Python
python_exporter = PythonExporter()
python_script, _ = python_exporter.from_notebook_node(nb)

# Salvar o script Python no arquivo 'desafio.py'
script_filename = "desafio.py"
with open(script_filename, "w", encoding="utf-8") as f:
    f.write(python_script)

print(f"Arquivo '{script_filename}' salvo com sucesso!")