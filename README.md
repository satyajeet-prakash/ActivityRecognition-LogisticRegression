create env

```bash
conda create -n activity-rec python=3.9 -y
```

Activate env
```bash
conda activate activity-rec
```

Create requirements file

Install the requirements
```bash
pip install -r requirements.txt
```

```bash
git init

dvc init

dvc add data_given/*/*.csv

git add . && git commit -m "initial commit"

git remote add origin git@github.com:satyajeet-prakash/ActivityRecognition-LogisticRegression.git

git branch -M main
 
git push -u origin main

```

<pre>Add pytest and tox to requirements file</pre>

tox command - 
```bash
tox
```

tox command for rebuilding -
```bash
tox -r
```

pytest command
```bash
pytest -v
```

setup command
```bash
pip install -e .
```

build your own package commands -
```bash
python setup.py sdist bdist_wheel
```

min-max for NOT IN RANGE
```bash
overview = df.describe()
overview.loc[["min", "max"]]
overview.loc[["min", "max"]].to_json("schema_in.json")
```

Heroku Link: https://activity-recognistion-logr.herokuapp.com/
