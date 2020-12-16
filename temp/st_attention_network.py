import torch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import numpy as np
import data_generator

model_batch_size = 16
model_lstm_size = 64
model_location_embedding_size = 64

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class AttentionModel(nn.Module):

    def __init__(self, vocab_size, location_embedding_size, lstm_size):
        super(AttentionModel, self).__init__()

        self.vocab_size = vocab_size
        self.location_embedding_size = location_embedding_size
        self.lstm_size = lstm_size

        self.location_embeddings = nn.Embedding(vocab_size, location_embedding_size)
        self.lstm_network = nn.LSTM(input_size=self.location_embedding_size,
                               hidden_size=self.lstm_size, num_layers=1,batch_first=True)

        self.attn_fc1 = nn.Linear(self.lstm_size , 16)
        self.attn_fc2 = nn.Linear(16,1)

        self.prb_fc = nn.Linear(lstm_size,vocab_size)


    def forward(self, input_seq , input_seq_length):


        input_seq_embedded = self.location_embeddings(input_seq)

        lstm_input = nn.utils.rnn.pack_padded_sequence(input_seq_embedded, input_seq_length, batch_first=True)
        output, _ = self.lstm_network(lstm_input)
        output, _ = nn.utils.rnn.pad_packed_sequence(output,batch_first=True) # B S L

        output_W1 = self.attn_fc1(output) # B S W
        output_W1_a = torch.unsqueeze(output_W1 , dim=1)
        output_W1_b = torch.unsqueeze(output_W1 , dim=2)
        output_W1_sum = output_W1_a + output_W1_b
        output_W1_th = torch.tanh(output_W1_sum)
        output_W2 = self.attn_fc2(output_W1_th)
        output_attn = torch.squeeze(output_W2,dim=3).to(device)

        max_length =  input_seq.shape[1]
        mask_1 = torch.arange(max_length)[None, :].to(device) < input_seq_length[:, None]
        mask_1 = mask_1.type(torch.FloatTensor)
        mask_22 = torch.unsqueeze(mask_1,2).to(device)
        mask_21 = torch.unsqueeze(mask_1,1).to(device)
        attn_mask = torch.einsum('ijk,ikl->ijl',mask_22,mask_21).to(device)
        output_attn = output_attn * attn_mask
        output_refined = torch.einsum('ijk,ikl->ijl',output_attn,output).to(device)
        batch_final_traj_embeddings = torch.transpose(output_refined,0,1)[input_seq_length-1,torch.arange(input_seq.shape[0]),:]
        output_prob = self.prb_fc(output_refined)
        return output_prob , batch_final_traj_embeddings


def loss_fn(pred, target):
    return torch.nn.CrossEntropyLoss()(torch.transpose(pred, 1, 2), target)


def print_all(model):

    dataset_trajectory = data_generator.get_trajectory_dataset()

    for i in range(dataset_trajectory.no_training_batches(model_batch_size)):
        inputs, seq_lengths, targets = dataset_trajectory(i, model_batch_size)
        inputs = torch.LongTensor(inputs).to(device)
        seq_lengths = torch.LongTensor(seq_lengths).to(device)

        targets = torch.LongTensor(targets).to(device)
        output = torch.softmax(model(inputs, seq_lengths)[0], dim=-1)

        #op = torch.transpose(output, 0, 1)
        op = output.cpu().detach().numpy()
        op = np.argmax(op, axis=2)

        #tgt = torch.transpose(targets, 0, 1)
        tgt = targets.cpu().numpy()

        print("PRED:")
        print(op)
        print("GT:")
        print(tgt)
        print("\n")


def train(model, optimizer, loss_fn, epochs=100):
    train_loss_min = 100000.0
    for epoch in range(epochs):

        training_loss = 0.0
        tr_s = data_generator.get_trajectory_dataset()
        model.train()

        for i in range(tr_s.no_training_batches(model_batch_size)):
            inputs, seq_lengths, targets = tr_s(i, model_batch_size)
            inputs = torch.LongTensor(inputs).to(device)
            seq_lengths = torch.LongTensor(seq_lengths).to(device)
            targets = torch.LongTensor(targets).to(device)
            optimizer.zero_grad()
            output, hid = model(inputs, seq_lengths)
            loss = loss_fn(output, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.2)
            optimizer.step()
            training_loss += loss.data.item()

        training_loss /= tr_s.no_training_batches(model_batch_size)

        if training_loss < train_loss_min:
            train_loss_min = training_loss
            torch.save(model.state_dict(),
                       "model_files/LSTM_net_1" + data_generator.dat_suffix[data_generator.dat_ix])
        print(
            'Epoch: {}, Loss: {:.3f}'.format(epoch, training_loss))




def get_lstm_model(load_from_file=True):
    if (load_from_file == False):
        attn_model = AttentionModel(vocab_size=len(data_generator.vocab_to_int) - 3,
                                    location_embedding_size=model_location_embedding_size, lstm_size=model_lstm_size).to(device)

        optimizer = optim.Adam(attn_model.parameters(), lr=0.001)
        train(attn_model, optimizer, loss_fn, epochs=500)

    lstm_model = AttentionModel(vocab_size=len(data_generator.vocab_to_int)-3,
                            location_embedding_size=model_location_embedding_size,lstm_size=model_lstm_size).to(device)
    fwd_model_state_dict = torch.load("model_files/LSTM_net_1"+data_generator.dat_suffix[data_generator.dat_ix])
    lstm_model.load_state_dict(fwd_model_state_dict)

    return lstm_model

#print_all(get_lstm_model(True))
