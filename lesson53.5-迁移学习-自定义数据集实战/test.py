import  torch
from    torch import optim, nn
import  visdom
import  torchvision
from    torch.utils.data import DataLoader

# from    resnet import ResNet18
from    torchvision.models import resnet18
class Flatten(nn.Module):

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        shape = torch.prod(torch.tensor(x.shape[1:])).item()
        return x.view(-1, shape)

if __name__ == '__main__':
    trained_model = resnet18(pretrained=True)
        # get params of 1st to 2nd last layers 
    model = nn.Sequential(*list(trained_model.children())[:-1], #[b, 512, 1, 1]
                            Flatten(), # [b, 512, 1, 1] => [b, 512]
                            nn.Linear(512, 5)
                            )
    print(*list(trained_model.children())[:-1])
