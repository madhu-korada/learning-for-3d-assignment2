# Instructions to execute the code

## 1 Fit the model
```
python fit_data.py --type 'point|vox|mesh'
```
or
```
python -m main --mode "fit" --type 'point|vox|mesh' 
```

## 2. Train and Eval the model
### Training
```
python -m train_model --type 'point|vox|mesh'  
```
or
```
python -m main --mode "train" --type 'point|vox|mesh'  
```

### Evaluation
```
python -m eval_model --type 'point|vox|mesh' --load_checkpoint 
```
or
```
python -m main --mode "eval" --type 'point|vox|mesh' --load_checkpoint --n_points 1000
```