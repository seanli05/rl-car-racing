import os
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.optim as optim
from models.dataset import Episodes, split_episodes

IMG_WIDTH = 96
IMG_HEIGHT = 96

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            # in: 96x96x3
            nn.Conv2d(in_channels=3, out_channels=16, stride=2, kernel_size=3, padding=1),
            # 48x48x16
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 24x24x16

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            # 24x24x32
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
            #12x12x32
        )

        self.linear_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(12*12*32, 512),
            nn.ReLU(),
            nn.Linear(512, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.linear_head(x)
        return x


if __name__ == "__main__":
    device = torch.accelerator.current_accelerator() if torch.accelerator.is_available() else "cpu"
    print(f"Using {device} device")
    model = CNN().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)


    train_files, test_files = split_episodes("data/plain", test_frac=0.2, seed=0)
    train = Episodes(train_files)
    test = Episodes(test_files)
    print(f"train: {len(train_files)} episodes, {len(train):,} frames | "
          f"test: {len(test_files)} episodes, {len(test):,} frames")

    train_loader = DataLoader(dataset=train, batch_size=128, shuffle=True)
    test_loader = DataLoader(dataset=test, batch_size=256, shuffle=False)


    num_epochs = 7
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            predicted = model.forward(images)
            loss = criterion(predicted, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.7f}")


    torch.save(model.state_dict(), "firsttest.pth")