This is the folder for DVC module of mlops course


```
dvc init --subdir #dvc initiated in a subfolder
dvc remote add -d <name> <path> 
git add .
git commit -m "dvc init" 
git push
dvc add <dataset.csv>
git add <dataset.csv>.dvc .gitignore
git commit -m "start tracking data"
git push
dvc push
```
