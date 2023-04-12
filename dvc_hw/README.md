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

Create DVC pipeline
```
dvc run -n processing -d processing.py -o combined_data.csv --no-exec python processing.py
```
or create dvc.yaml manually

```
dvc repro
``` 
to reproduce pipeline