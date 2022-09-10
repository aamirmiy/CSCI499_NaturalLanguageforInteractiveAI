# Instructions
1.Run the below command to download GLoVE embeddings
```
!wget http://nlp.stanford.edu/data/wordvecs/glove.6B.zip
!unzip glove.6B.zip
```
2.Set the path of the txt file containing the embeddings on line 309 
 ```
 glove = pd.read_csv('/content/glove.6B.300d.txt', sep=" ", quoting=3, header=None, index_col=0)
 ```
 
3.Depending on the dimension size to be used make changes on line  312, 319, 320, 322, 329
  ```
    312 - embedding_matrix = create_embedding_matrix(v2i, glove_embedding, 300)
    319 - eos_vector = np.random.standard_normal(300)
    320 - sos_vector = np.random.standard_normal(300)
    322 - pad_vector = np.zeros(300)
    329 - model = setup_model(device,len(maps[0]),len(maps[2]),300,embedding_matrix,256,1)
    ```
  
# Running the code
```
Train:
python train1.py \
    --in_data_fn=lang_to_sem_data.json \
    --model_output_dir=experiments/lstm \
    --batch_size=1000 \
    --num_epochs=100 \
    --val_every=5 \
    --force_cpu 
```

