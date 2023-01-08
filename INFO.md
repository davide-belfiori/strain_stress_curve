# How to start a new project

1. Create a new virtual environment and activate it.

```console
python -m venv .venv
```

```console
.\.venv\Scripts\Activate.ps1
```

2. Edit `requirements.txt` and `setup.cfg` files. 

3. Install this project as a python module.

```console
python -m pip install -e .
```

4. Install project requirements.

```console
python -m pip install -r requirements.txt
```

5. Install development requirements.

```console
python -m pip install -r requirements_dev.txt
```
