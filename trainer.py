from models.factory import SafetyValueFunctionFactory
import torch
from torch import nn
from typing import Callable, List
from math import ceil
import matplotlib.pyplot as plt
from datetime import datetime
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

device = torch.device("cpu")

nn_net_config = {
    "name": "feedforward",
    "eps": 1e-2,
    "layers": [64, 64],
    "activation": nn.ReLU(),
    "device": device,
    "input_size": 2,
    "bounded": True,
}

model = SafetyValueFunctionFactory.create(**nn_net_config)
x_vals = torch.linspace(-1, 1, 100)
y_vals = torch.linspace(-1, 1, 100)
x_grid, y_grid = torch.meshgrid(x_vals, y_vals,indexing="ij")

input_data = torch.stack([x_grid.flatten(), y_grid.flatten()], dim=-1)

def function(data:torch.Tensor):
    x = data[:, 0]
    y = data[:, 1]
    output = -1*x**5*y**5
    output = torch.clamp(output,min = -1, max = 0)

    return output

def plot(max_x1:float, max_x2:float,
        resolution:int, 
        value_net:Callable[[torch.Tensor], torch.Tensor],
        levels:List[float] = [0.0]):
    x1_low = -max_x1*1.1
    x1_high = max_x1*1.1
    x2_low = -max_x2*1.1
    x2_high = max_x2*1.1
    linspace_x1 = torch.linspace(x1_low, x1_high, ceil(resolution*(x1_high-x1_low)))
    linspace_x2 = torch.linspace(x2_low, x2_high, ceil(resolution*(x2_high-x2_low)))
    linspaces = [linspace_x1, linspace_x2]
    mesh = torch.meshgrid(*linspaces,indexing="xy")
    inputs = torch.stack([m.flatten() for m in mesh],dim=-1)
    outputs = value_net(inputs).detach().cpu().numpy()
    outputs = outputs.reshape(mesh[0].shape)
    fig = plt.figure()

    ax = fig.add_subplot(111, projection='3d')
    
    surf = ax.plot_surface(mesh[0],mesh[1],outputs,cmap='coolwarm')
    ax.plot([-max_x1, -max_x1], [-max_x2, max_x2],[0,0], "r")
    ax.plot([max_x1, max_x1], [-max_x2, max_x2],[0,0], "r")
    ax.plot([-max_x1, max_x1], [-max_x2, -max_x2],[0,0], "r")
    ax.plot([-max_x1, max_x1], [max_x2, max_x2], [0,0],"r")
    ax.set_xlim(-max_x1*1.1, max_x1*1.1)
    ax.set_ylim(-max_x2*1.1, max_x2*1.1)
   
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("Value function") 
    ax.set_title("Value function landscape")
    fig.colorbar(surf)
    plt.savefig("test" + datetime.now().strftime("%Y%m%d-%H%M%S") + ".pdf")


targets = function(input_data).unsqueeze(-1)
batch_size = 64
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
dataset = TensorDataset(input_data, targets)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
loss_fn = nn.MSELoss()
epochs = 500
pbar = tqdm(range(epochs))
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    epoch_loss = 0
    pbar.update(1)
    for inputs, target in loader:
        inputs = inputs.to(device)
        target = target.to(device)
        output = model(inputs)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    pbar.set_postfix(loss=epoch_loss/len(loader))
    pass
model.eval()
with torch.no_grad():
    inputs = input_data.to(device)
    outputs = model(inputs)
    loss = loss_fn(outputs, targets)
    print(f"Test loss: {loss.item()}")
plot(1, 1, 1000, model)