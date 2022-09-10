<h1><strong>CSCI499: Coding Assignment 1</strong></h1> 




<h2>Model architecture with Parameters</h2>
The baseline model after hyperparameter tuning has the following parameters.
<br></br>
<ul>Embedding Layer with dimension of <strong>100</strong></ul>
<ul>One LSTM layer with hidden size of <strong>128</strong></ul>
<ul>Feedforward layer 1 for predicting the Action class</ul>
<ul>Feedforward layer 2 for predicting the Target class</ul>
<hr>
<h2>Loss Function and Optimizer</h2>
<ul>CrossEntropy</ul>
<ul>Adam Optimizer with a learning rate of <strong>0.1</strong></ul>
<hr>
<h2>Results</h2>
<ul><b>Metrics</b></ul>
<ul>The baseline model gives train target accuracy of around 88% with validation target accuracy of 77%. Training more than 20 epochs results in decreasing accuracy for validation set.</ul>
<ul> Increasing the hidden dimension of LSTM to 256 from 128 results in 92% target train accuracy. However, the validation accuracy still remains the same.</ul>
<ul>With respect to the ACTION class, the validation and train accuracy are almost the same i.e. around 98%-99% irrespective of change in parameters.</ul>
<br>
<ul><b>Plots</b></ul>
<ul>Plot for baseline model</ul><img src="plot with 128 lstm .png" ></ul>
<ul>Plot with lstm having hidden size of 256</ul><img src ="plot with 256 lstm.png"></ul>
<br></br>
<b>Legend</b>:<br>
<b>Green = Train</b>
<b>Blue = Validation</b>
<ul></ul>
<hr>
<h2>Notes</h2>
<ul>This code was run on google colab gpu <strong>(Tesla P100)</strong></ul>
<ul>The model was trained for <b>20</b> epochs with validation every <b>5</b> epochs</ul>
<ul>The batch size used was <b>64</b>. Going above 64 resulted in reduced train and validation target accuracy</ul>
<ul> All the parameters and hyperparameters were chosen based on experimentation</ul>
<hr>
<h2>Bonus Task</h2>
<h3>Initialize your LSTM embedding layer with word2vec, GLoVE, or other pretrained lexical embeddings. How does this change affect performance?</h3><br></br>
<ul>Leveraged <b>GLoVE</b> pretrained embeddings having dimensions <b>100</b> and <b>300</b></ul>
<ul>Overall the accuracy did not improve even after using pretrained embeddings as opposed to those made from scratch. This is possible when the population the pretrained model was trained on is very different from the dataset that we are using as it is easier to learn new weights than to unlearn the wrong ones and relearn the right ones. Furthermore, pretrained embeddings are more useful when we don't have sufficient data which is not the case.</ul>
<ul>Results</ul>
<ul><table style="width:100%">
  <tr>
    <th>Embedding Dimension</th>
    <th>Batch Size</th>
    <th>Lstm Dimension</th>
    <th>Train Target Accuracy</th>
    <th>Validation Target Accuracy</th>
  </tr>
  <tr>
    <td>300</td>
    <td>64</td>
    <td>128</td>
    <td>87.27%</td>
    <td>~77%</td>
  </tr>
   <tr>
    <td>100</td>
    <td>64</td>
    <td>128</td>
    <td>84.50%</td>
    <td>~77%</td>
  </tr>
   <tr>
    <td>300</td>
    <td>256</td>
    <td>256</td>
    <td>83.23%</td>
    <td>~77%</td>
  </tr>
   <tr>
    <td>100</td>
    <td>64</td>
    <td>256</td>
    <td>88.90%</td>
    <td>~77%</td>
  </tr>
     <tr>
    <td>300</td>
    <td>64</td>
    <td>256</td>
    <td>90.27%</td>
    <td>~76%</td>
  </tr>
</table></ul>
<ul><strong>Plots</strong></ul>
<ul>Plot for pretrained embeddings of 100 dimensions with 128 LSTM hidden size</ul><img src="embedding 100d 128 lstm.png"></ul>
<ul>Plot for pretrained embeddings of 300 dimensions with 128 LSTM hidden size</ul><img src="embedding 300d lstm 128.png"></ul>
<ul>Plot for pretrained embeddings of 300 dimensions with 256 LSTM hidden size</ul><img src="300 embedding 256 lstm.png"></ul><br>
<b>Legend</b>:<br>
<b>Green = Train</b>
<b>Blue = Validation</b>
<br></br>
<ul>The results were calculated over 20 epochs of training with validation after every 5 epochs.</ul>
<hr>


