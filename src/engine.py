import tqdm as tqdm
import torch
import config

def train_fn(model, data_loader, optimizer):
    model.train()

    fn_loss = 0

    tk0 = tqdm(data_loader, total=len(data_loader))
    for batch in tk0:
        for key, value in data.items():
            batch[key] = value.to(config.DEVICE)
        optimizer.zero_grad()

        _, loss = model(**batch)   

        loss.backwards()

        optimizer.step()

        fn_loss += loss.item()

    return fn_loss / len(data_loader)    

def eval_mode(model, data_loader):

    fn_loss = 0
    fin_preds = []
    tk0 = tqdm(data_loader, total=len(data_loader))
    model.eval()
    with torch.inference_mode():

        for batch in tk0:
            for key, value in batch.items():
                batch[key] = value.to(config.DEVICE)

            preds, loss = model(**batch) 
           
            fin_preds.append(preds)  

            fn_loss += loss.item()

    return fn_loss, fn_loss / len(data_loader)    




    