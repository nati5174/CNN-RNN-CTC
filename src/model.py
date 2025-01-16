import torch
from torch import nn
from torch.nn import functional as F

class CaptchaModel(nn.Module):

    def __init__(self, nums_chars):
        super(CaptchaModel, self).__init__()

        self.convd1 = nn.Conv2d(3, 128, kernel_size=(3, 3), padding=(1, 1))
        self.max1 = nn.MaxPool2d(kernel_size = (2, 2))
        self.convd2 = nn.Conv2d(128, 64, kernel_size=(3, 3), padding=(1, 1))
        self.max2 = nn.MaxPool2d(kernel_size = (2, 2)) #(1, 64, 18, 75) where 1 is batch, 64 is the filters, 18 hight, 75 width?

        self.linear1 = nn.Linear(1152, 64)
        self.dp1 = nn.Dropout(0.2) #some neurons will be zeroed out, randmy during training to prevent overfitting

        self.lstm  = nn.LSTM(input_size=64, hidden_size=32, num_layers=2, bidirectional=True, dropout=0.25, batch_first=True) # this will output a size of 64 bc bidriectiona;==2. thus 2 hidden states 32+32

        self.linear2 = nn.Linear(64, nums_chars + 1) 
        
    def forward(self, images, targets=None):

        bs, c, h, w = images.size() #(5, 3, 75, 300)
        x = F.relu(self.convd1(images))  #(5, 128, 75, 300)
        x = self.max1(x)                #(5, 128, 37, 150)
        x = F.relu(self.convd2(x))  #(5, 64, 37, 150)
        x = self.max2(x)            #(5, 64, 18, 75)
        x = x.permute(0, 3, 1, 2)   #(5, 75, 18, 64)                         #reorder so that we can go from (batcth, filters, hiehgt, width) to (batch, hieght, filters, width)
        x = x.view(bs, x.size(1),-1)  #(5, 75, 1152)  # (0, 3, 1*2)
        x = self.linear1(x)            #(5, 75, 64)
        x = self.dp1(X)                #(5, 75, 64)
        x, _ = self.lstm(x)            # (5, 75, 64)
        x = self.linear2(x) #output shoulf be (5, 75, 20)
        x = x.permute(1,0,2) #now (75, 5, 20)
        if targets is not None:
                        log_probs = F.log_softmax(x, 2)
                        input_lengths = torch.full(
                                size=(bs,), fill_value=log_probs.size(0), dtype=torch.int32
                                                            ) # vills outa tensor wit array of size bs, with all elements being log_probs.size(0)
                        output_lengths = torch.ful(size=(bs,), fill_value=targets.size(1), dtype=torch.int32)

                        loss = nn.CTCLoss(blank = 0)(log_probs, targets, input_lengths,output_lengths)

                        return x, loss

        return x, None




if __name__ == "__main__":
    cm = CaptchaModel(19)
    img = torch.rand(5, 3, 75, 300) #*bactch, channels-rgb, height, width)
    target = torch.randint(5, 20, (5, 5))
    x, loss = cm(img, target)

         
