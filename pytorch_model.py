import torch
import torch.nn as nn

class CustomModel(nn.Module):
    def __init__(self, num_classes=25):
        super(CustomModel, self).__init__()

        self.conv1 = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.batch_norm1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64, 32, kernel_size=(3, 3), stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.batch_norm2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 32, kernel_size=(3, 3), stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.batch_norm3 = nn.BatchNorm2d(32)

        self.conv4 = nn.Conv2d(32, 16, kernel_size=(3, 3), stride=1, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.batch_norm4 = nn.BatchNorm2d(16)

        self.conv5 = nn.Conv2d(16, 16, kernel_size=(3, 3), stride=1, padding=1)
        self.pool5 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.batch_norm5 = nn.BatchNorm2d(16)


        self.flatten = nn.Flatten()

        self.fc1_input_size = 16 * (target_size_custom[0] // 32) * (target_size_custom[1] // 32)
        self.fc1 = nn.Linear(self.fc1_input_size, 256)
        self.dropout1 = nn.Dropout(0.25)
        self.batch_norm_fc1 = nn.BatchNorm1d(256)

        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(0.25)
        self.batch_norm_fc2 = nn.BatchNorm1d(128)

        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.batch_norm1(self.pool1(torch.relu(self.conv1(x))))
        x = self.batch_norm2(self.pool2(torch.relu(self.conv2(x))))
        x = self.batch_norm3(self.pool3(torch.relu(self.conv3(x))))
        x = self.batch_norm4(self.pool4(torch.relu(self.conv4(x))))
        x = self.batch_norm5(self.pool5(torch.relu(self.conv5(x))))

        x = self.flatten(x)

        x = self.batch_norm_fc1(self.dropout1(torch.relu(self.fc1(x))))
        x = self.batch_norm_fc2(self.dropout2(torch.relu(self.fc2(x))))
        x = self.fc3(x)

        return x
