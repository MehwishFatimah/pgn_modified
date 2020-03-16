import os

'''----------------------------------------------------'''
#input setup
root_in 	 = "/hits/basement/nlp/fatimamh/test_data"
train_folder = os.path.join(root_in, "train")
val_folder 	 = os.path.join(root_in, "val")
test_folder  = os.path.join(root_in, "test")

#vocab setup
vocab_folder = os.path.join(root_in, "dict")
text_vocab_f = os.path.join(vocab_folder, "en_text_vocab_f.json") #full
sum_vocab_f	 = os.path.join(vocab_folder, "en_sum_vocab_f.json") #full
text_vocab_c = os.path.join(vocab_folder, "en_text_vocab_c.json") #condense
sum_vocab_c	 = os.path.join(vocab_folder, "en_sum_vocab_c.json") #condense

'''----------------------------------------------------'''
#out_folder 
root_out 	= "/hits/basement/nlp/fatimamh/test_data"
#"check_point_folder"
log_folder 	= os.path.join(root_out, "log")
best_model 	= os.path.join(log_folder, "best_model.pth.tar")
check_point = "/hits/basement/nlp/fatimamh/test_data/log/train/model_ep_2_b_5.pth.tar"

#output setup
out_folder  = os.path.join(root_out, "out")
s_summaries = os.path.join(out_folder, "test_summaries.csv") #system summaries
scores 	    = os.path.join(out_folder, "summaries_with_scores.csv")
results     = os.path.join(out_folder, "scores.csv")
epoch_loss  = os.path.join(out_folder, "epoch_loss.csv")

'''----------------------------------------------------'''
#data setting
train_docs	= 64
val_docs	= 16
test_docs	= 16
max_text	= 400
max_sum	  	= 100
text_vocab	= 50004 # include 4 special in count
sum_vocab	= 50004 
PAD_index	= 0
UNK_index 	= 1
SP_index  	= 2
EP_index  	= 3

'''----------------------------------------------------'''
# Hyperparameters
emb_dim	 	= 128
hid_dim	 	= 256
batch_size  = 16 # 8 
beam_size	= 8 # for BATCH SIZE8 it is 4

print_every	= 1
plot_every  = 2

epochs	 	= 50 #500000
n_layers	= 1  
lr 			= 0.15
momentum	= 0.9
enc_drop	= 0 
dec_drop	= 0.1
grad_norm	= 2.0

adagrad_init_acc    = 0.1
rand_unif_init_mag  = 0.02
trunc_norm_init_std = 1e-4

pgn			= True
coverage 	= False 
cov_loss_wt = 1.0
eps 		= 1e-12
lr_coverage = 0.15


