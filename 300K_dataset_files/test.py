import torch
import os

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from model import ModelCNN_fvs, ModelResNet50_fvs, ModelResNet34_fvs, ModelDenseNet121_fvs, ModelSwinT_fvs, ModelEfficientNetB0_fvs
from loss_fcns import RMSELoss
from Call_dataset import MyDataset
from save_validation import save_prediction

in_instances = ['fvs', 'fls', 'x0', 'dx', 'Ch']
in_channels = 3
model = ModelCNN_fvs(in_instances, in_channels).to(device)
model_weight = 'saved_best_model_pretrained.pth'
# If measured data is used no target (true) included
# case = "measured"
case = "synthetic"

if __name__== "__main__":
    criterion = RMSELoss()
    #load model
    model.load_state_dict(torch.load(model_weight, map_location=device, weights_only=True))
    model.eval()

    # Choose test mode: if synthetic data is used the target (true) is included.
    if case == "synthetic":
        data_folder="testing_synthetic"
        test_dataset=MyDataset(in_instances,
                               in_channels,
                               os.path.join(data_folder,"input"),
                               os.path.join(data_folder,"output"))
        test_loss =0
        idx = 0
        for data in test_dataset:
            # Unpack all inputs and targets
            *all_inputs, target = data
            # Move everything to device
            all_inputs = [inp.unsqueeze(0).to(device) for inp in all_inputs]
            target = target.unsqueeze(0).to(device)
            predict = model(*all_inputs)
            input_fvs = all_inputs[0].squeeze(0).cpu().detach() if len(all_inputs) > 0 else None
            input_fls = all_inputs[1].squeeze(0).cpu().detach() if len(all_inputs) > 1 else None
            input_x0 = all_inputs[2].flatten().cpu().detach() if len(all_inputs) > 2 else None
            input_dx = all_inputs[3].flatten().cpu().detach() if len(all_inputs) > 3 else None
            input_Ch = all_inputs[4].flatten().cpu().detach() if len(all_inputs) > 4 else None

            save_prediction(input_fvs=input_fvs,
                            input_fls=input_fls,
                            predict=predict.squeeze(0).cpu().detach(),
                            file_id=idx,
                            folder_save_result=os.path.join(data_folder,"predict"),
                            input_x0=[input_x0[0]],
                            input_dx=[input_dx[0]],
                            input_Ch=[input_Ch[0]],
                            target=target.squeeze(0).cpu().detach()
                            )
            idx +=1

    elif case == "measured":
        data_folder="testing_measured"
        test_dataset = MyDataset(in_instances,
                                 in_channels,
                                 os.path.join(data_folder,"input")
                                 )
        idx = 0
        for data in test_dataset:
            # Unpack all inputs and targets
            *all_inputs, = data
            # Move everything to device
            all_inputs = [inp.unsqueeze(0).to(device) for inp in all_inputs]
            predict = model(*all_inputs)

            input_fvs = all_inputs[0].squeeze(0).cpu().detach() if len(all_inputs) > 0 else None
            input_fls = all_inputs[1].squeeze(0).cpu().detach() if len(all_inputs) > 1 else None
            input_x0 = all_inputs[2].flatten().cpu().detach() if len(all_inputs) > 2 else None
            input_dx = all_inputs[3].flatten().cpu().detach() if len(all_inputs) > 3 else None
            input_Ch = all_inputs[4].flatten().cpu().detach() if len(all_inputs) > 4 else None

            save_prediction(input_fvs=input_fvs,
                            input_fls=input_fls,
                            predict=predict.squeeze(0).cpu().detach(),
                            file_id=idx,
                            folder_save_result=os.path.join(data_folder,"predict"),
                            input_x0=[input_x0[0]],
                            input_dx=[input_dx[0]],
                            input_Ch=[input_Ch[0]],
                            )
            idx += 1