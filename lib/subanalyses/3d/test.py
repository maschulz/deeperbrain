import numpy
import torch
from data import get_dataloaders
from peng import PengNet
from sklearn.metrics import r2_score
from tqdm import tqdm

dataloaders, regression, output_dim = get_dataloaders(8000, 0, 8, 'ageC')

state_dict = torch.load(STATE_PATH, map_location=torch.device('cpu'), )
model = PengNet(Config()).cpu()
model.load_state_dict(state_dict['state_dict'])

Y, Y_pred = [], []
for x, y in tqdm(dataloaders['test']):
    Y_pred.append(model(x).view(len(y)).detach().numpy())
    Y.append(y.numpy())
Y, Y_pred = numpy.concatenate(Y), numpy.concatenate(Y_pred)

R2 = []
for i in range(2000):
    idx = numpy.random.choice(numpy.arange(len(Y)), size=len(Y), replace=True)
    R2.append(r2_score(Y[idx], Y_pred[idx]))

print(r2_score(Y, Y_pred), numpy.percentile(R2, 2.5), numpy.percentile(R2, 97.5))