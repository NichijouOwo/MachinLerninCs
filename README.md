### Machine Learning Cs GO
## Instrucciones para Iniciar
### Clonar el Repositorio
Abre la terminal o el símbolo del sistema y navega al directorio donde deseas clonar el repositorio. Luego, ejecuta el siguiente comando para clonar el repositorio desde Git:
```python
git clone https://github.com/NichijouOwo/MachinLerninCs.git            
```
El proyecto se estructura de la siguiente manera:
### Importante Habilitar los permisos de ejecucion de scripts.
1. Abrir PowerShell.
2. Poner el siguiente comando:
 ```python
Set-ExecutionPolicy
  ```
3. Darle la opcion si a todo.
   
### Crear Entorno Virtual
1. Navega al directorio del proyecto en la terminal.
2. Crea un entorno virtual ejecutando:
```python
python -m venv env
```
Esto creará un nuevo directorio env en tu proyecto que contendrá el entorno virtual.

### Instalar Dependencias
Activa el entorno virtual:

En Windows:
```bash
.\env\Scripts\Activate.ps1
```
En macOS y Linux:

```bash
source env/bin/activate
```
Instala las dependencias del proyecto utilizando el archivo requirements.txt:
```bash
pip install -r requirements.txt
```
### Ejecutar el proyecto
Ejecuta el script Python principal con:

```bash
python script.py
```
