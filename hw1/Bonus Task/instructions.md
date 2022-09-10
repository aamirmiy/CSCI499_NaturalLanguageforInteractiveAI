# Instructions
1.Run the below command to download GLoVE embeddings
```
!wget http://nlp.stanford.edu/data/wordvecs/glove.6B.zip
!unzip glove.6B.zip
```
2.Set the path of the txt file containing the embeddings on line 309   
3.Depending on the dimension size to be used make changes on line  312, 319, 320, 322, 329

# Running the code
```
Train:
python train.py \
    --in_data_fn=lang_to_sem_data.json \
    --model_output_dir=experiments/lstm \
    --batch_size=1000 \
    --num_epochs=100 \
    --val_every=5 \
    --force_cpu 
```

