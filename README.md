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

git add . && git commit -m "first commit"

git remote add origin git@github.com:satyajeet-prakash/ActivityRecognition-LogisticRegression.git

git branch -M main
 
git push -u origin main

```