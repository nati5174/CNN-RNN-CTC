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

        self.lstm  = nn.LSTM(input_size=64, hidden_size=32, num_layers=2, bidirectional=True, dropout=0.25, batch_first=True)

        self.linear2 = nn.Linear(64, nums_chars + 1)
        
    def forward(self, images, targets=None):

        bs, c, h, w = images.size()
        x = F.relu(self.convd1(images))
        x = self.max1(x)
        x = F.relu(self.convd2(x))
        x = self.max2(x)
        x = x.permute(0, 3, 1, 2) #reorder so that we can go from (batcth, filters, hiehgt, width) to (batch, hieght, filters, width)
        x = x.view(bs, x.size(1),-1) # (0, 3, 1*2)
        x = self.linear1(x)
        x = self.dp1(X)
        x, _ = self.lstm(x)
        x = self.output(x)
        return x, None




if __name__ == "__main__":
    cm = CaptchaModel(19)
    img = torch.rand(1, 3, 75, 300)
    target = torch.randint(1, 3, 75, 300)
    x, loss = cm(img, target)

         
