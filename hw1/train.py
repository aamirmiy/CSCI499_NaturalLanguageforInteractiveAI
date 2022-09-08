import tqdm
import torch
import argparse
from sklearn.metrics import accuracy_score
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

from utils import (
    get_device,
    read_episodes,
    flatten_list,
    build_tokenizer_table,
    build_output_tables,
    encode_data
)

from model import (
    semanticNet
)

def setup_dataloader(args):
    """
    return:
        - train_loader: torch.utils.data.Dataloader
        - val_loader: torch.utils.data.Dataloader
    """
    # ================== TODO: CODE HERE ================== #
    # Task: Load the training data from provided json file.
    # Perform some preprocessing to tokenize the natural
    # language instructions and labels. Split the data into
    # train set and validataion set and create respective
    # dataloaders.

    # Hint: use the helper functions provided in utils.py
    # ===================================================== #
    train_data, val_data =  read_episodes(args.in_data_fn)
    vocab_to_index, index_to_vocab, len_cutoff = build_tokenizer_table(train_data, vocab_size = 10000)
    actions_to_index, index_to_actions, targets_to_index, index_to_targets = build_output_tables(train_data)
    train_data = flatten_list(train_data)
    val_data = flatten_list(val_data)
    #train_np_x, train_np_y1, train_np_y2 =  encode_data(train_data, vocab_to_index, len_cutoff, targets_to_index, actions_to_index)
    train_np_x, train_np_y =  encode_data(train_data, vocab_to_index, len_cutoff, targets_to_index, actions_to_index)
    #train_dataset = TensorDataset(torch.from_numpy(train_np_x), torch.from_numpy(train_np_y1), torch.from_numpy(train_np_y2))
    train_dataset = TensorDataset(torch.from_numpy(train_np_x), torch.from_numpy(train_np_y))
    #val_np_x, val_np_y1, val_np_y2 = encode_data(val_data, vocab_to_index, len_cutoff, targets_to_index, actions_to_index)
    val_np_x, val_np_y = encode_data(val_data, vocab_to_index, len_cutoff, targets_to_index, actions_to_index)
    #val_dataset = TensorDataset(torch.from_numpy(val_np_x), torch.from_numpy(val_np_y1), torch.from_numpy(val_np_y2))
    val_dataset = TensorDataset(torch.from_numpy(val_np_x), torch.from_numpy(val_np_y))
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size,drop_last=True)
    val_loader = DataLoader(val_dataset, shuffle=True, batch_size=args.batch_size,drop_last=True)

    return train_loader, val_loader, (actions_to_index, index_to_actions, targets_to_index, index_to_targets)


def setup_model(device, vocab_size, output_size1, output_size2, embedding_dim, hidden_dim, n_layers):
    """
    return:
        - model: YourOwnModelClass
    """
    # ================== TODO: CODE HERE ================== #
    # Task: Initialize your model.
    # ===================================================== #
    model = semanticNet(device,vocab_size, output_size1, output_size2, embedding_dim, hidden_dim, n_layers)
    model = model.to(device)
    
    return model


def setup_optimizer(args, model):
    """
    return:
        - action_criterion: loss_fn
        - target_criterion: loss_fn
        - optimizer: torch.optim
    """
    # ================== TODO: CODE HERE ================== #
    # Task: Initialize the loss function for action predictions
    # and target predictions. Also initialize your optimizer.
    # ===================================================== #
    action_criterion = torch.nn.CrossEntropyLoss()
    target_criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    return action_criterion, target_criterion, optimizer


def train_epoch(
    args,
    model,
    loader,
    optimizer,
    action_criterion,
    target_criterion,
    device,
    training=True,
):
    epoch_action_loss = 0.0
    epoch_target_loss = 0.0

    # keep track of the model predictions for computing accuracy
    action_preds = []
    target_preds = []
    action_labels = []
    target_labels = []
    
    # iterate over each batch in the dataloader
    # NOTE: you may have additional outputs from the loader __getitem__, you can modify this
    for (inputs, labels) in loader:
        # put model inputs to device
        inputs, labels = inputs.to(device), labels.to(device)
        model.train()
        
        

        # calculate the loss and train accuracy and perform backprop
        # NOTE: feel free to change the parameters to the model forward pass here + outputs
        actions_out, targets_out= model(inputs)
      
    

        # calculate the action and target prediction loss
        # NOTE: we assume that labels is a tensor of size Bx2 where labels[:, 0] is the
        # action label and labels[:, 1] is the target label
        action_loss = action_criterion(actions_out.squeeze(), labels[:, 0].long())
        target_loss = target_criterion(targets_out.squeeze(), labels[:, 1].long())
        
        loss = action_loss + target_loss

        # step optimizer and compute gradients during training
        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # logging
        epoch_action_loss += action_loss.item()
        epoch_target_loss += target_loss.item()

        # take the prediction with the highest probability
        # NOTE: this could change depending on if you apply Sigmoid in your forward pass
        action_preds_ = actions_out.argmax(-1)
        target_preds_ = targets_out.argmax(-1)

        # aggregate the batch predictions + labels
        action_preds.extend(action_preds_.cpu().numpy())
        target_preds.extend(target_preds_.cpu().numpy())
        action_labels.extend(labels[:, 0].cpu().numpy())
        target_labels.extend(labels[:, 1].cpu().numpy())

    action_acc = accuracy_score(action_preds, action_labels)
    target_acc = accuracy_score(target_preds, target_labels)

    return epoch_action_loss, epoch_target_loss, action_acc, target_acc


def validate(
    args, model, loader, optimizer, action_criterion, target_criterion, device
):
    # set model to eval mode
    model.eval()

    # don't compute gradients
    with torch.no_grad():

        val_action_loss, val_target_loss, action_acc, target_acc = train_epoch(
            args,
            model,
            loader,
            optimizer,
            action_criterion,
            target_criterion,
            device,
            training=False,
        )

    return val_action_loss, val_target_loss, action_acc, target_acc


def train(args, model, loaders, optimizer, action_criterion, target_criterion, device):
    # Train model for a fixed number of epochs
    # In each epoch we compute loss on each sample in our dataset and update the model
    # weights via backpropagation
    model.train()

    train_action_losses=[]
    train_target_losses=[]
    train_action_accuracy=[]
    train_target_accuracy=[]

    val_action_losses = []
    val_target_losses =[]
    val_action_accuracy = []
    val_target_accuracy = []

    for epoch in tqdm.tqdm(range(args.num_epochs)):

        # train single epoch
        # returns loss for action and target prediction and accuracy
        (
            train_action_loss,
            train_target_loss,
            train_action_acc,
            train_target_acc,
        ) = train_epoch(
            args,
            model,
            loaders["train"],
            optimizer,
            action_criterion,
            target_criterion,
            device,
        )

        # some logging
        print(
            f"train action loss : {train_action_loss} | train target loss: {train_target_loss}"
        )
        print(
            f"train action acc : {train_action_acc} | train target acc: {train_target_acc}"
        )
        train_action_losses.append(train_action_loss)
        train_target_losses.append(train_target_loss)
        train_action_accuracy.append(train_action_acc)
        train_target_accuracy.append(train_target_acc)

        # run validation every so often
        # during eval, we run a forward pass through the model and compute
        # loss and accuracy but we don't update the model weights
        if epoch % args.val_every == 0:
            val_action_loss, val_target_loss, val_action_acc, val_target_acc = validate(
                args,
                model,
                loaders["val"],
                optimizer,
                action_criterion,
                target_criterion,
                device,
            )

            print(
                f"val action loss : {val_action_loss} | val target loss: {val_target_loss}"
            )
            print(
                f"val action acc : {val_action_acc} | val target losaccs: {val_target_acc}"
            )
            val_action_losses.append(val_action_loss)
            val_target_losses.append(val_target_loss)
            val_action_accuracy.append(val_action_acc)
            val_target_accuracy.append(val_target_acc)

    # ================== TODO: CODE HERE ================== #
    # Task: Implement some code to keep track of the model training and
    # evaluation loss. Use the matplotlib library to plot
    # 4 figures for 1) training loss, 2) training accuracy,
    # 3) validation loss, 4) validation accuracy
    # ===================================================== #
    epo = range(1,21)
    fig, axs = plt.subplots(2, 2)
    axs[0, 0].plot(epo, val_action_losses, 'b', label='Validation Action Loss')
    axs[0, 0].plot(epo, train_action_losses,'g',label='Training Action Loss')
    axs[0, 0].set_title('Training and Validation loss')
    axs.set(xlabel='Epochs', ylabel='Loss')
    axs[0, 1].plot(epo, val_target_losses, 'b', label='Validation Target Loss')
    axs[0, 1].plot(epo, train_target_losses,'g',label='Training Target Loss')
    axs[0, 1].set_title('Training and Validation loss')
    axs.set(xlabel='Epochs', ylabel='Loss')
    axs[1, 0].plot(epo, val_action_accuracy, 'b', label='Validation Action accuracy')
    axs[1, 0].plot(epo, train_action_accuracy,'g',label='Training Action accuracy')
    axs[1, 0].set_title('Training and Validation Accuracy')
    axs.set(xlabel='Epochs', ylabel='Accuracy')
    axs[1, 1].plot(epo, val_target_accuracy, 'b', label='Validation Target accuracy')
    axs[1, 1].plot(epo, train_target_accuracy,'g',label='Training Action accuracy')
    axs[1, 1].set_title('Training and Validation Accuracy')
    axs.set(xlabel='Epochs', ylabel='Accuracy')
    
   

def main(args):
    
    device = get_device(args.force_cpu)

    # get dataloaders
    train_loader, val_loader, maps = setup_dataloader(args)
    loaders = {"train": train_loader, "val": val_loader}

    # build model
    #setup_model(vocab_size, output_size1, output_size2, embedding_dim, hidden_dim, n_layers)
    model = setup_model(device,2063,len(maps[0]),len(maps[2]),100,128,1)
    
    print(model)
    #model.to(device)
    # get optimizer and loss functions
    action_criterion, target_criterion, optimizer = setup_optimizer(args, model)

    if args.eval:
        val_action_loss, val_target_loss, val_action_acc, val_target_acc = validate(
            args,
            model,
            loaders["val"],
            optimizer,
            action_criterion,
            target_criterion,
            device,
        )
    else:
        train(
            args, model, loaders, optimizer, action_criterion, target_criterion, device
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_data_fn", type=str, help="data file")
    parser.add_argument(
        "--model_output_dir", type=str, help="where to save model outputs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="size of each batch in loader"
    )
    parser.add_argument("--force_cpu", action="store_true", help="debug mode")
    parser.add_argument("--eval", action="store_true", help="run eval")
    parser.add_argument("--num_epochs", default=1000, help="number of training epochs")
    parser.add_argument(
        "--val_every", default=5, help="number of epochs between every eval loop"
    )

    # ================== TODO: CODE HERE ================== #
    # Task (optional): Add any additional command line
    # parameters you may need here
    # ===================================================== #
    args = parser.parse_args()

    main(args)
