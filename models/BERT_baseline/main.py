import math
import time
from abc import ABC

from sklearn.metrics import classification_report
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch import optim
import random
import numpy as np
import torch.nn as nn
from transformers import BertModel

# Specify loss function
loss_fn = nn.CrossEntropyLoss()


# Create the BertClassfier class
class BertClassifier(nn.Module, ABC):
    """Bert Model for Classification Tasks.
    """

    def __init__(self, device, freeze_bert=True, D_out=2):
        """
        @param    bert: a BertModel object
        @param    classifier: a torch.nn.Module classifier
        @param    freeze_bert (bool): Set `False` to fine-tune the BERT model
        """
        super(BertClassifier, self).__init__()
        # Specify hidden size of BERT, hidden size of our classifier, and number of labels
        D_in, H = 768, 50

        # Instantiate BERT model
        self.bert = BertModel.from_pretrained('bert-base-uncased')

        # Instantiate an one-layer feed-forward classifier
        self.classifier = nn.Sequential(
            nn.Linear(D_in, H),
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(H, D_out).to(device)
        )

        # Freeze the BERT model
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, **kwargs):
        """
        Feed input to BERT and the classifier to compute logits.
        @param    input_ids (torch.Tensor): an input tensor with shape (batch_size,
                      max_length)
        @param    attention_mask (torch.Tensor): a tensor that hold attention mask
                      information with shape (batch_size, max_length)
        @return   logits (torch.Tensor): an output tensor with shape (batch_size,
                      num_labels)
        """
        # Feed input to BERT
        outputs = self.bert(**kwargs)

        # Extract the last hidden state of the token `[CLS]` for classification task
        last_hidden_state_cls = outputs[0][:, 0, :]

        # Feed input to classifier to compute logits
        logits = self.classifier(last_hidden_state_cls)

        return logits


class BertBaseline:
    def __init__(self, cfg):
        self.cfg = cfg
        self.domain = cfg.domain
        self.root_path = cfg.path_mapper
        self.bert_type = 'GroNLP/bert-base-dutch-cased'
        self.categories = cfg.domain.aspect_category_mapper
        self.polarities = cfg.domain.sentiment_category_mapper
        self.device = cfg.device
        self.tokenizer = AutoTokenizer.from_pretrained(self.bert_type)
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.batch_size = cfg.domain.params.batch_size
        self.name = 'BertBaseline'
        
    def __call__(self, cats):
        cat_dataset, pol_dataset = self.load_training_data()
        if cats=='polarities':
          self.train_model(pol_dataset, epochs=4, cats=cats)
        else: 
          self.train_model(cat_dataset, epochs=4, cats=cats)
        return self.evaluate(cats)
        

    def find_sentences(self, categories, seeds):
        inv_cat_dict = {}
        for i, cat in enumerate(categories):
            inv_cat_dict[cat] = i

        sentences, cats = [], []

        with open(f'{self.root_path}/train.txt', 'r', encoding="utf8") as f:
            for idx, line in enumerate(f):
                found_cats = []
                for category in categories:
                    text = line.strip()
                    if not set(seeds[category]).isdisjoint(text.split()):
                        found_cats.append(category)
                if len(found_cats) == 1:
                    sentences.append(text)
                    cats.append(inv_cat_dict[found_cats[0]])

        encoded_dict = self.tokenizer(
            sentences,
            padding=True,
            return_tensors='pt',
            max_length=128,
            return_attention_mask=True,
            truncation=True)
        labels_cat = torch.tensor(cats)

        dataset = TensorDataset(
            labels_cat, encoded_dict['input_ids'], encoded_dict['token_type_ids'],
            encoded_dict['attention_mask'])
        return dataset

    def load_training_data(self):
        aspect_seeds = self.cfg.domain.aspect_seed_mapper
        sentiment_seeds = self.cfg.domain.sentiment_seed_mapper

        cat_dataset = self.find_sentences(self.categories, aspect_seeds)
        pol_dataset = self.find_sentences(self.polarities, sentiment_seeds)
        return cat_dataset, pol_dataset

    def initialize_model(self, dataloader, epochs=4, cats='categories', lr=5e-5, beta1=0.9, beta2=0.999):
        """Initialize the Bert Classifier, the optimizer and the learning rate scheduler.
        """
        if cats == 'polarities':
            out_len = len(self.polarities)
        else:
            out_len = len(self.categories)

        # Instantiate Bert Classifier and Tell PyTorch to run the model on GPU
        self.model = BertClassifier(self.device, freeze_bert=False, D_out=out_len).to(self.device)

        # Create the optimizer
        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=lr,  # Default learning rate
                                    betas=(beta1, beta2),
                                    eps=1e-8  # Default epsilon value
                                    )

        # Total number of training steps
        total_steps = len(dataloader) * epochs

        # Set up the learning rate scheduler
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer,
                                                         num_warmup_steps=0,  # Default value
                                                         num_training_steps=total_steps)

    def set_seed(self, value):
        random.seed(value)
        np.random.seed(value)
        torch.manual_seed(value)
        torch.cuda.manual_seed_all(value)

    def train_model(self, dataset, params=None, epochs=None, cats='categories', hyper=False):
        """Train the BertClassifier model.
            """
        device = self.device

        if epochs is None:
            epochs = self.cfg.epochs

        if params:
            learning_rate, beta1, beta2, batch_size = params
            self.batch_size = params[3]
        else:
            learning_rate = self.cfg.domain.params.learning_rate
            batch_size = self.cfg.domain.params.batch_size
            beta1 = self.cfg.domain.params.beta1
            beta2 = self.cfg.domain.params.beta2

        # Prepare dataset
        if hyper:
            data_size = math.floor(len(dataset) * self.cfg.domain.hyper_validation_size)
            train_data, val_data = torch.utils.data.random_split(
                dataset, [data_size, len(dataset) - data_size])
        else:
            train_data, val_data = torch.utils.data.random_split(
                dataset, [len(dataset) - self.cfg.domain.validation_data_size, self.cfg.domain.validation_data_size])

        dataloader = DataLoader(train_data, batch_size=self.batch_size)
        val_dataloader = DataLoader(val_data, batch_size=self.batch_size)

        # Initialize the Bert Classifier
        self.initialize_model(dataloader, epochs=2, cats=cats, lr=learning_rate, beta1=beta1, beta2=beta2)

        model = self.model

        # Start training loop
        print("Start training...\n")
        for epoch_i in range(epochs):
            # =======================================
            #               Training
            # =======================================
            # Print the header of the result table
            print(
                f"{'Epoch':^7} | {'Batch':^7} | {'Train Loss':^12} | {'Val Loss':^10} | {'Val Acc':^9} | {'Elapsed':^9}")
            print("-" * 70)

            # Measure the elapsed time of each epoch
            t0_epoch, t0_batch = time.time(), time.time()

            # Reset tracking variables at the beginning of each epoch
            total_loss, batch_loss, batch_counts = 0, 0, 0

            # Put the model into the training mode
            model.train()

            for step, (labels, input_ids, token_type_ids, attention_mask) in enumerate(dataloader):
                batch_counts += 1
                # Zero out any previously calculated gradients
                model.zero_grad()

                encoded_dict = {
                    'input_ids': input_ids.to(device),
                    'token_type_ids': token_type_ids.to(device),
                    'attention_mask': attention_mask.to(device)
                }

                # Perform a forward pass. This will return logits.
                logits = model(**encoded_dict)

                # Compute loss and accumulate the loss values
                loss = loss_fn(logits, labels.to(device))
                batch_loss += loss.item()
                total_loss += loss.item()

                # Perform a backward pass to calculate gradients
                loss.backward()

                # Clip the norm of the gradients to 1.0 to prevent "exploding gradients"
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                # Update parameters and the learning rate
                self.optimizer.step()
                self.scheduler.step()

                # Print the loss values and time elapsed for every 20 batches
                if (step % 20 == 0 and step != 0) or (step == len(dataloader) - 1):
                    # Calculate time elapsed for 20 batches
                    time_elapsed = time.time() - t0_batch

                    # Print training results
                    print(
                        f"{epoch_i + 1:^7} | {step:^7} | {batch_loss / batch_counts:^12.6f} | {'-':^10} | {'-':^9} | {time_elapsed:^9.2f}")

                    # Reset batch tracking variables
                    batch_loss, batch_counts = 0, 0
                    t0_batch = time.time()

                # Calculate the average loss over the entire training data
            avg_train_loss = total_loss / len(dataloader)

            print("-" * 70)
            # =======================================
            #               Evaluation
            # =======================================
            # Put the model into the evaluation mode. The dropout layers are disabled during
            # the test time.
            model.eval()

            # Tracking variables
            val_accuracy = []
            val_loss = []

            for labels, input_ids, token_type_ids, attention_mask in val_dataloader:
                encoded_dict = {
                    'input_ids': input_ids.to(device),
                    'token_type_ids': token_type_ids.to(device),
                    'attention_mask': attention_mask.to(device)
                }

                # Compute logits
                with torch.no_grad():
                    logits = model(**encoded_dict)

                # Compute loss
                loss = loss_fn(logits, labels.to(device))
                val_loss.append(loss.item())

                # Get the predictions
                preds = torch.argmax(logits, dim=1).flatten()

                # Calculate the accuracy rate
                accuracy = (preds == labels.to(device)).cpu().numpy().mean() * 100
                val_accuracy.append(accuracy)

            # Compute the average accuracy and loss over the validation set.
            val_loss = np.mean(val_loss)
            val_accuracy = np.mean(val_accuracy)

            # Display the epoch training loss and validation loss
            print(
                f"{epoch_i + 1:^7} | {'-':^7} | {avg_train_loss:^12.6f} | {val_loss:^10.6f} | {val_accuracy:^9.2f} | {time_elapsed:^9.2f}")
            print("-" * 70)
            print("\n")

        print("Training complete!")

        return val_loss, val_accuracy

    def hypertuning(self, params):
        cat_dataset, pol_dataset = self.load_training_data()
        pol_loss, pol_acc = self.train_model(pol_dataset, params=params, epochs=4, cats='polarities', hyper=True)
        cat_loss, cat_acc = self.train_model(cat_dataset, params=params, epochs=4, cats='categories', hyper=True)

        return pol_loss + cat_loss, pol_acc + cat_acc

    def evaluate(self, cats='categories'):
        test_sentences = []
        test_cats = []

        cat_dict = {}
        if cats == 'categories' :
            for i, cat in enumerate(self.categories):
                cat_dict[i] = cat
        else:
            for i, cat in enumerate(self.polarities):
                cat_dict[i] = cat

        with open(f'{self.root_path}/test.txt', 'r', encoding="utf8") as f:
            for line in f:
                _, cat, pol, sentence = line.strip().split('\t')
                if cats=='categories':
                    cat = int(cat)
                else:
                    cat = int(pol)

                test_cats.append(cat)
                test_sentences.append(sentence)

        model = self.model
        model.eval()
        device = self.device

        actual_category = []
        predicted_category= []

        iters = 0
        with torch.no_grad():
            for input, cat in tqdm(zip(test_sentences, test_cats)):

                encoded_dict = self.tokenizer([input],
                                              padding=True,
                                              return_tensors='pt',
                                              return_attention_mask=True,
                                              truncation=True).to(device)
                # Compute logits
                logits = model(**encoded_dict)

                actual_category.append(cat_dict[cat])
                predicted_category.append(
                    cat_dict[torch.argmax(logits).item()])
                iters += 1

            predicted = np.array(predicted_category)
            actual = np.array(actual_category)
            print(cats)
            print(classification_report(actual, predicted, digits=4))
            print()

            return classification_report(
                actual, predicted, digits=4, zero_division=0,
                output_dict=True
            )