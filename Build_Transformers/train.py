import torch
import torch.nn as nn
from torch.utils.data import DataLoader ,random_split
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter

from Transformer import build_transformers
from datasets_tf import BilingualDatasets,causal_mask
from Transformer import build_transformers

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace 
from config import get_weights_file_path,get_config
from pathlib import Path
from tqdm import tqdm




def greedy_decode(model,source,source_mask,tokenizer_src,tokenizer_tgt, max_len, device) :
    sos_idx=tokenizer_tgt.token_to_id["[SOS]"]
    eos_idx=tokenizer_tgt.token_to_id["[EOS]"]

    #Precopute the output of the encoder and reuse it for every token from the decoder
    encoder_output=model.encode(source,source_mask) 
    
    # First output of the decoder from the sos token
    decoder_input=torch.empty(1,1).fill_(sos_idx).type_as(source).to(device)
    while True :
        if decoder_input==max_len :
            break;

        ##Build the mask for the output
        decoder_mask=causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)

        ##Pass it through the model
        out=model.decode(encoder_output,source_mask,decoder_input,decoder_mask)

        ##Get the next/last token
        prob=model.project(out[:,-1])

        ##Word/token with the max proba//Greedy search
        _,next_word=torch.max(prob,dim=1)
        decoder_input=torch.stack([decoder_input,torch.empty(1,1).fill_(next_word).type_as(source).to(device)],dim=1)

        if next_word==eos_idx :
            break

        return decoder_input.unsqueeze(0)




def run_validation(model,validation_ds,tokenizer_src,tokenizer_tgt, max_len,device,print_msg,global_step,writer,num_samples=2):
    model.eval()
    count=0
    sources_txt=[]
    expected=[
    ]
    predicted=[]


    console_wd=80
    with torch.no_grad() :
        for batch in validation_ds :
            count+=1
            encoder_input=batch['encoder_input'].to(device)
            encoder_mask=batch['encoder_mask'].to(device)

            assert encoder_input.size(0)==1 
            
            model_out=greedy_decode(model,encoder_input,encoder_mask,tokenizer_src,tokenizer_tgt,max_len,device)

            sources_txt=batch["src_txt"][0]
            target_text=batch["tgt_text"][0]
            model_out_txt=tokenizer_tgt.decode(model_out.detach().cpu().numpy())

            sources_txt.append(sources_txt)
            predicted.append(model_out_txt)
            expected.append(target_text)


            if count==num_samples :
                break




    return sources_txt,target_text,predicted,expected









def get_all_sentences(ds,lang):

    for item in ds :
        yield item['translation'][lang]



def get_build_token(config , ds, lang):
    tokenizer_path=Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        tokenizer=Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer=Whitespace()

        trainer=WordLevelTrainer(special_tokens=['[UNK]',"[PAD]", "[SOS]" , "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds,lang),trainer=trainer)
    
    else :
         tokenizer=Tokenizer.from_file(str(tokenizer_path))
    
    return tokenizer




def get_ds(config):
    ds_raw=load_dataset('opus_books',f'{config["lang_src"]}-{config["lang_tgt"]}',split='train')

    tokenizer_src=get_build_token(config,ds_raw,config["lang_src"])
    tokenizer_tgt=get_build_token(config,ds_raw,config["lang_tgt"])

    ##SPlit datasets
    train_ds_size=int(0.9*len(ds_raw))
    val_ds_size=len(ds_raw)-train_ds_size
    train_ds_raw,val_ds_raw=random_split(ds_raw,[train_ds_size,val_ds_size])

    train_ds=BilingualDatasets(train_ds_raw,tokenizer_src,tokenizer_tgt, config["lang_src"], config["lang_tgt"],config["seq_len"])
    val_ds=BilingualDatasets(val_ds_raw,tokenizer_src,tokenizer_tgt, config["lang_src"], config["lang_tgt"],config["seq_len"])

 
    max_len_src=0
    max_len_tgt=0

    for item  in ds_raw:
        src_ids=tokenizer_src.encode(item['translation'][config["lang_src"]]).ids
        # print(src_ids)
        tgt_ids=tokenizer_tgt.encode(item['translation'][config["lang_tgt"]]).ids

        max_len_src=max(max_len_src,len(src_ids))
        max_len_tgt=max(max_len_tgt,len(tgt_ids))
    
    print(f'Max length of source sentence : {max_len_src}')
    print(f'Max length of tgt sentence : {max_len_tgt}')


    train_dataloader=DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True)
    val_dataloader=DataLoader(val_ds, batch_size=1, shuffle=True)

    return train_dataloader,val_dataloader,tokenizer_src,tokenizer_tgt



def get_model(config,vocab_src_len,vocab_tgt_len) :
    model=build_transformers(vocab_src_len,vocab_tgt_len,config["seq_len"], config["seq_len"], config['d_model'])
    return model



def train_model(config):
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    Path(config['model_folder']).mkdir(parents=True,exist_ok=True)
    train_dataloader,val_dataloader,tokenizer_src,tokenizer_tgt=get_ds(config)
    model=get_model(config,tokenizer_src.get_vocab_size(),tokenizer_tgt.get_vocab_size()).to(device)

    #TensorBoard
    writer=SummaryWriter(config["experiment_name"])
    optimizer=torch.optim.AdamW(model.parameters(),lr=config["lr"], eps=1e-9)


    init_epoch=0
    global_step=0
    if config["preload"] :
        model_filename=get_weights_file_path(config,config["preload"])
        print((f" Preloading Model {model_filename} "))
        state=torch.load(model_filename)
        init_epoch=state["epoch"] +1 
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step=state["global_step"]


    loss_fn=nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id("[PAD]"),label_smoothing=0.1)


    for epoch in range(init_epoch, config['num_epoch']) :
        model.train()
        batch_iterator=tqdm(train_dataloader, desc=f"Processing epoch {epoch:02d}")
        for batch in batch_iterator :
            encoder_input=batch["encoder_input"].to(device) #(Batch,seq_len)
            decoder_input=batch["decoder_input"].to(device) #(Batch, seq_len)
            encoder_mask=batch["encoder_mask"].to(device) #(Batch, 1,1, seq_lenh)
            decoder_mask=batch["decoder_mask"].to(device)#(Batch, 1,seq_len, seq_lenh)


            #Run through the transformer
            encoder_output=model.encode(encoder_input,encoder_mask)
            decoder_output=model.decode(encoder_output,encoder_mask,decoder_input,decoder_mask) #(Batchn seq_len, d_model)
            proj_output=model.proj(decoder_output) #(Batch, seq_len , vocan_size)

            label=batch["label"].to(device) #(Batch,seq_len)

            
            loss=loss_fn(proj_output.view(-1,tokenizer_tgt.get_vocab_size()),label.view(-1))
            batch_iterator.set_postfix(f'loss : f"{loss.item():6.3.f}"')


            ##Write to tensor board 
            writer.add_scalar('train loss', loss.item(), global_step)
            writer.flush()
            
            #Backpropagate
            loss.backward()

            #Update the weight
            optimizer.step()
            optimizer.zero_grad()


            global_step+=1


            ##Save the model

        run_validation(model,val_dataloader,tokenizer_src,tokenizer_tgt,config["seq_len"], device, lambda msg :batch_iterator.write(msg), global_step,writer )
        model_filename=get_weights_file_path(config, f'{epoch:02d}')
        torch.save({
            "epoch" : epoch,
            "model_state_dict" : model.state_dict(),
            "optimizer_state_dict" : optimizer.state_dict(),
            "global_setp" : global_step,


        }, model_filename)



if __name__=='__main__' :
    config=get_config()
    train_model(config)
